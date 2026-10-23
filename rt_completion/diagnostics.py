"""Bounded, append-only diagnostics for numerically unsafe training batches.

These files are debugging artifacts, not supervision or radar export caches.
Only the explicitly allowed minibatch tensors are retained; model weights and
dense SMPL-X meshes are never copied here.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid

import torch


_BATCH_KEYS = {
    "raw": ("raw", "rt"),
    "corrupted": ("corrupted",),
    "mask": ("mask", "masked"),
    "joints": ("joints",),
    "history_valid": ("history_valid",),
    "indicators": ("indicators", "indicator"),
}


def _json_value(value):
    """Produce strict-JSON, weights-only-safe data without NaN literals."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return _json_value(value.item() if value.numel() == 1 else value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    # NumPy scalar metrics provide item(), without requiring NumPy here.
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def _allowed_batch(batch):
    selected = {}
    for name, alternatives in _BATCH_KEYS.items():
        for key in alternatives:
            if key in batch:
                selected[name] = batch[key]
                break
    return selected


def tensor_statistics(value):
    """Return finite-only ranges so an all-NaN batch is still serializable."""
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    count = int(finite.sum().item())
    result = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "finite_count": count,
        "nonfinite_count": tensor.numel() - count,
        "min": None,
        "max": None,
        "mean": None,
    }
    if count:
        values = tensor[finite].to(dtype=torch.float64)
        result.update(min=float(values.min().item()), max=float(values.max().item()),
                      mean=float(values.mean().item()))
    return _json_value(result)


@contextmanager
def _locked_log(path):
    # The trainer is single-process today. Locking also prevents competing
    # resumed invocations from exceeding the snapshot limit or mixing lines.
    import fcntl

    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_numerics_event(output, event, batch=None, max_snapshots=3,
                         max_snapshot_bytes=128 * 1024 * 1024):
    """Append a strict-JSON event and optionally retain a small CPU minibatch.

    ``event`` should identify the 1-based epoch and step, reason, AMP mode,
    gradient scale, and bad parameter names. Non-finite metric values become
    JSON null. Tensor summaries are computed even after the snapshot budget is
    exhausted. No caller-provided artifact is overwritten.

    Snapshot payloads contain only ordinary dictionaries, primitive values,
    and CPU tensors and can be read with ``torch.load(..., weights_only=True)``.
    This function returns the append-only JSONL path.
    """
    if max_snapshots < 0 or max_snapshot_bytes < 0:
        raise ValueError("Diagnostic snapshot limits must be nonnegative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "numerics_events.jsonl"
    record = _json_value(dict(event))
    record["event_id"] = uuid.uuid4().hex
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    selected = _allowed_batch(batch or {})
    record["tensor_stats"] = {
        key: tensor_statistics(value) for key, value in selected.items()
        if isinstance(value, torch.Tensor)
    }
    size = sum(value.numel() * value.element_size() for value in selected.values()
               if isinstance(value, torch.Tensor))
    record["snapshot_tensor_bytes"] = size
    record["snapshot"] = None
    with _locked_log(log_path) as stream:
        snapshot_dir = output / "debug_failures"
        existing = sum(1 for _ in snapshot_dir.glob("failure_*.pt"))
        if not selected:
            record["snapshot_skipped"] = "no_batch"
        elif existing >= max_snapshots:
            record["snapshot_skipped"] = "snapshot_limit"
        elif size > max_snapshot_bytes:
            record["snapshot_skipped"] = "byte_limit"
        else:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            # UUID avoids accidental overwrites even when epoch/step repeat on
            # resume. Epoch and step are formatting-only, never used as paths.
            epoch = int(record.get("epoch") or 0)
            step = int(record.get("step") or 0)
            name = f"failure_epoch{epoch:03d}_step{step:08d}_{record['event_id']}.pt"
            snapshot_path = snapshot_dir / name
            record["snapshot"] = str(snapshot_path.relative_to(output))
            cpu_batch = {
                key: value.detach().cpu().clone() if isinstance(value, torch.Tensor)
                else _json_value(value)
                for key, value in selected.items()
            }
            try:
                with snapshot_path.open("xb") as snapshot_stream:
                    torch.save({"event": record, "batch": cpu_batch}, snapshot_stream)
            except Exception as error:
                # Preserve the numeric event even if a snapshot cannot be
                # serialized (for example insufficient available disk space).
                record["snapshot"] = None
                record["snapshot_skipped"] = f"write_failed: {type(error).__name__}"
        stream.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
        stream.flush()
    return log_path
