"""Repeatable mask groups and diagnostics on held-out MEASURED returns only."""

import hashlib
import math

import numpy as np
import torch

from .losses import corrupt_rt
from .selection import select_additions


MASK_GROUPS = ("random", "single_part", "continuous_part", "spatial_block")


def mask_seed(indicator, seed, group):
    payload = f"{int(seed)}:{group}:" + ":".join(str(int(x)) for x in indicator)
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "little") % (2**63 - 1)


def fixed_corruption(raw, joints, indicators, mask_config, bounds, seed, group):
    """Masks are independent of epoch, shuffle, batch size and worker count."""
    if group not in MASK_GROUPS:
        raise ValueError(f"Unknown fixed mask group: {group}")
    settings = dict(mask_config, mode="mixed_parts", radius_jitter=0., joint_jitter_m=0.)
    corrupted, masked, infos = [], [], []
    for sample in range(raw.shape[0]):
        rng = torch.Generator(device=raw.device).manual_seed(
            mask_seed(indicators[sample].tolist(), seed, group))
        value, heldout, info = corrupt_rt(
            raw[sample:sample + 1], settings, rng,
            joints=joints[sample:sample + 1], bounds=bounds,
            group=group, return_info=True)
        corrupted.append(value)
        masked.append(heldout)
        infos.append(info)
    info = {}
    for key in ("part_index", "duration", "eligible"):
        if key in infos[0]:
            info[key] = torch.cat([item[key] for item in infos], dim=0)
    info["kind"] = [group] * raw.shape[0]
    return torch.cat(corrupted), torch.cat(masked), info


class HeldoutDiagnostics:
    """Streaming, voxel-weighted risk/coverage; never retain dense predictions.

    Coverage checks are conditional on the chosen synthetic masks and recorded
    noisy measurements. They do NOT establish calibration at natural zeros.
    """
    def __init__(self):
        self.frames = self.eligible_frames = self.count = self.accepted_count = 0
        self.error_sum = self.accepted_error_sum = self.sigma_sum = 0.
        self.cover68 = self.cover95 = 0
        self.edges = np.linspace(-3., 2., 21)
        self.bin_count = np.zeros(20, np.int64)
        self.bin_error = np.zeros(20, np.float64)
        self.bin_squared_error = np.zeros(20, np.float64)
        self.bin_sigma = np.zeros(20, np.float64)

    @torch.no_grad()
    def update(self, error, sigma, masked, accepted, accepted_error):
        self.frames += int(masked.shape[0])
        self.eligible_frames += int(masked.flatten(1).any(1).sum())
        errors = error[masked].float()
        deviations = sigma[masked].float()
        self.count += errors.numel()
        self.error_sum += float(errors.sum())
        self.sigma_sum += float(deviations.sum())
        self.cover68 += int((errors <= deviations).sum())
        self.cover95 += int((errors <= 1.95996398454 * deviations).sum())
        chosen = accepted & masked
        self.accepted_count += int(chosen.sum())
        self.accepted_error_sum += float(accepted_error[chosen].sum())
        if errors.numel():
            edges = torch.as_tensor(self.edges[1:-1], device=errors.device, dtype=torch.float32)
            indices = torch.bucketize(deviations.clamp_min(1e-12).log().contiguous(), edges)
            self.bin_count += torch.bincount(indices, minlength=20).cpu().numpy()
            self.bin_error += torch.bincount(indices, weights=errors, minlength=20).double().cpu().numpy()
            self.bin_squared_error += torch.bincount(indices, weights=errors.square(), minlength=20).double().cpu().numpy()
            self.bin_sigma += torch.bincount(indices, weights=deviations, minlength=20).double().cpu().numpy()

    def summary(self):
        count = max(1, self.count)
        risk, calibration = [], []
        cumulative_count, cumulative_error = 0, 0.
        for index, number in enumerate(self.bin_count):
            if not number:
                continue
            cumulative_count += int(number)
            cumulative_error += float(self.bin_error[index])
            upper = math.exp(float(self.edges[index + 1]))
            risk.append({"sigma_log_upper": upper, "coverage": cumulative_count / count,
                         "mae_log": cumulative_error / cumulative_count})
            calibration.append({"sigma_log_upper": upper, "voxels": int(number),
                                "mean_sigma_log": float(self.bin_sigma[index] / number),
                                "rmse_log": math.sqrt(float(self.bin_squared_error[index] / number))})
        return {
            "frames": self.frames, "eligible_frames": self.eligible_frames,
            "eligible_frame_fraction": self.eligible_frames / max(1, self.frames),
            "total_masked_voxels": self.count,
            "masked_mae_log": self.error_sum / count if self.count else None,
            "masked_coverage68": self.cover68 / count if self.count else None,
            "masked_coverage95": self.cover95 / count if self.count else None,
            "masked_sigma_log": self.sigma_sum / count if self.count else None,
            "accepted_heldout_voxels": self.accepted_count,
            "accepted_heldout_coverage": self.accepted_count / count if self.count else None,
            "accepted_heldout_mae_log": (self.accepted_error_sum / self.accepted_count
                                          if self.accepted_count else None),
            "risk_coverage": risk, "uncertainty_bins": calibration,
        }


class GroupMetrics:
    def __init__(self):
        self.totals, self.samples = {}, 0
        self.heldout = HeldoutDiagnostics()
        self.parts = {}

    def scalar_totals(self):
        """Copy scalar accumulators to the host in one transfer, not per key.

        Checkpoints contain Python floats only, including when these totals
        have been accumulated on CUDA. Legacy progress files contain floats
        too, so a resumed accumulator can be moved lazily on its next update.
        """
        if not self.totals:
            return {}
        keys = list(self.totals)
        device = next((value.device for value in self.totals.values()
                       if torch.is_tensor(value)), torch.device("cpu"))
        values = torch.stack([torch.as_tensor(self.totals[key], device=device,
                                             dtype=torch.float64).reshape(())
                              for key in keys])
        return dict(zip(keys, values.detach().cpu().tolist()))

    @torch.no_grad()
    def update(self, metrics, model, original, corrupted, masked, prediction, info, export_settings,
               *, detailed=True):
        size = original.shape[0]
        self.samples += size
        keys = [key for key in metrics if not key.endswith("_sum")]
        if keys:
            device = next((metrics[key].device for key in keys if torch.is_tensor(metrics[key])),
                          torch.device("cpu"))
            # Stack first, then promote once. Double-precision accumulation
            # preserves small increments across hundreds of thousands of steps.
            current = torch.stack([torch.as_tensor(metrics[key], device=device).detach().reshape(())
                                   for key in keys]).to(dtype=torch.float64)
            previous = torch.stack([torch.as_tensor(self.totals.get(key, 0.), device=device,
                                                   dtype=torch.float64).reshape(())
                                    for key in keys])
            self.totals.update(zip(keys, (previous + current * size).unbind()))
        if not detailed:
            return
        error = (prediction["intensity"] - model.normalize(original[:, -1:])).abs()
        selected = select_additions(model, corrupted, prediction, export_settings)
        accepted_error = (model.normalize(selected["values"])
                          - model.normalize(original[:, -1:])).abs()
        self.heldout.update(error, selected["sigma"], masked,
                            selected["accepted"], accepted_error)
        if "part_index" in info:
            for batch, part in enumerate(info["part_index"].tolist()):
                if part < 0:
                    continue
                diagnostic = self.parts.setdefault(str(part), HeldoutDiagnostics())
                sl = slice(batch, batch + 1)
                diagnostic.update(error[sl], selected["sigma"][sl], masked[sl],
                                  selected["accepted"][sl], accepted_error[sl])

    def summary(self):
        result = {key: value / max(1, self.samples) for key, value in self.scalar_totals().items()}
        diagnostic_samples = self.heldout.frames
        scope = ("disabled" if not diagnostic_samples else
                 "full" if diagnostic_samples == self.samples else "sampled")
        result.update(diagnostics_scope=scope, diagnostic_samples=diagnostic_samples)
        if diagnostic_samples:
            diagnostic = self.heldout.summary()
            if self.parts:
                diagnostic["parts"] = {part: value.summary()
                                       for part, value in sorted(self.parts.items())}
            if scope == "full":
                # Preserve validation's voxel-weighted held-out MAE and all
                # existing selection/calibration keys exactly.
                result.update(diagnostic)
            else:
                # A resumed old run or periodic diagnostics describe only a
                # subset. Do not replace the all-batch scalar MAE with its MAE
                # or present partial coverage as a complete-epoch statistic.
                result["sampled_diagnostics"] = diagnostic
        return result
