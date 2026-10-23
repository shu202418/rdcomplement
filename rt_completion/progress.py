"""Safe, deterministic state for resuming inside a completion training epoch.

The sampler intentionally uses a dedicated CPU RNG. Its order does not depend
on DataLoader workers, prefetch, or how many earlier samples have been read.
This ordering is for new step checkpoints; legacy epoch checkpoints did not
record enough state to reproduce their earlier DataLoader shuffle.
"""

import math
import operator
import random

import numpy as np
import torch
from torch.utils.data import Sampler

from .evaluation import GroupMetrics, HeldoutDiagnostics


def capture_rng_state(mask_generator=None):
    """Capture RNGs using only primitives and tensors (weights_only safe).

    Reading the CUDA state is deliberately conditional on initialization, so a
    CPU training checkpoint does not initialize a CUDA context as a side effect.
    """
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [],
        "mask": mask_generator.get_state().clone() if mask_generator is not None else None,
    }


def restore_rng_state(state, mask_generator=None):
    """Restore a state captured above, including a dedicated mask RNG if given."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["algorithm"],
        numpy_state["keys"].detach().cpu().numpy().astype(np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore CUDA RNG state without an available CUDA device")
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved RNG state")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if mask_generator is not None and state.get("mask") is not None:
        mask_generator.set_state(state["mask"].cpu())


class DeterministicBatchSampler(Sampler):
    """Resume shuffled batches without loading/skipping any preceding frames.

    ``seed`` must incorporate the epoch. ``start_step`` is the number of batches
    already consumed, not successful optimizer updates: skipped nonfinite
    batches must still advance this position. DataLoader should also receive
    its own generator to keep its worker seeding off the global torch RNG.
    """

    def __init__(self, dataset_length, batch_size, seed, start_step=0, drop_last=False):
        self.dataset_length = operator.index(dataset_length)
        self.batch_size = operator.index(batch_size)
        self.seed = operator.index(seed)
        self.start_step = operator.index(start_step)
        self.drop_last = bool(drop_last)
        if self.dataset_length < 0:
            raise ValueError("dataset_length must be nonnegative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.total_steps = (self.dataset_length // self.batch_size if self.drop_last
                            else math.ceil(self.dataset_length / self.batch_size))
        if not 0 <= self.start_step <= self.total_steps:
            raise ValueError("start_step must lie between zero and total_steps")

    def __iter__(self):
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        order = torch.randperm(self.dataset_length, generator=generator).tolist()
        for step in range(self.start_step, self.total_steps):
            offset = step * self.batch_size
            yield order[offset:offset + self.batch_size]

    def __len__(self):
        return self.total_steps - self.start_step


_INTEGER_FIELDS = ("frames", "eligible_frames", "count", "accepted_count", "cover68", "cover95")
_FLOAT_FIELDS = ("error_sum", "accepted_error_sum", "sigma_sum")
_ARRAY_FIELDS = ("edges", "bin_count", "bin_error", "bin_squared_error", "bin_sigma")


def _diagnostics_state(diagnostics):
    state = {name: int(getattr(diagnostics, name)) for name in _INTEGER_FIELDS}
    state.update({name: float(getattr(diagnostics, name)) for name in _FLOAT_FIELDS})
    state.update({name: torch.from_numpy(getattr(diagnostics, name).copy())
                  for name in _ARRAY_FIELDS})
    return state


def _restore_diagnostics(state):
    diagnostics = HeldoutDiagnostics()
    for name in _INTEGER_FIELDS:
        setattr(diagnostics, name, int(state[name]))
    for name in _FLOAT_FIELDS:
        setattr(diagnostics, name, float(state[name]))
    for name in _ARRAY_FIELDS:
        value = state[name].detach().cpu().numpy().copy()
        expected = getattr(diagnostics, name)
        if value.shape != expected.shape:
            raise ValueError(f"Invalid held-out diagnostic shape for {name}: {value.shape}")
        setattr(diagnostics, name, value.astype(expected.dtype, copy=False))
    return diagnostics


def group_metrics_state(metrics):
    """Serialize streaming metrics explicitly, without pickling Python objects."""
    return {
        "version": 1,
        "samples": int(metrics.samples),
        "totals": {str(key): value for key, value in metrics.scalar_totals().items()},
        "heldout": _diagnostics_state(metrics.heldout),
        "parts": {str(part): _diagnostics_state(diagnostic)
                  for part, diagnostic in metrics.parts.items()},
    }


def restore_group_metrics_state(state):
    """Rebuild accumulators so epoch summaries also include pre-resume batches."""
    if state.get("version") != 1:
        raise ValueError("Unsupported group metrics checkpoint version")
    metrics = GroupMetrics()
    metrics.samples = int(state["samples"])
    metrics.totals = {str(key): float(value) for key, value in state["totals"].items()}
    metrics.heldout = _restore_diagnostics(state["heldout"])
    metrics.parts = {str(part): _restore_diagnostics(value)
                     for part, value in state["parts"].items()}
    return metrics
