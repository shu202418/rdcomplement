"""RT-only historical scatter transfer guided by predicted body geometry.

SMPL-X labels may supervise the joint predictor during training, but this
module consumes only raw RT and network predictions.  Geometry determines
where measured historical returns may move; it never invents their amplitude.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import _body_prior, articulated_warp


def _triplet(value, name, cast=float):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three values in X,Y,Z order")
    result = tuple(cast(item) for item in value)
    if any(item < 0 for item in result):
        raise ValueError(f"{name} values must be non-negative")
    return result


def _splat_kernel(bounds, shape, sigma_m, radius_voxels, device):
    radius = _triplet(radius_voxels, "splat_radius_voxels", int)
    sigma = _triplet(sigma_m, "splat_sigma_m", float)
    axes = []
    for dim, (rad, sig, (low, high), size) in enumerate(zip(radius, sigma, bounds, shape)):
        spacing = float(high - low) / max(1, int(size) - 1)
        offsets = torch.arange(-rad, rad + 1, device=device, dtype=torch.float32) * spacing
        if rad and sig <= 0:
            raise ValueError(f"splat_sigma_m[{dim}] must be positive when its radius is nonzero")
        axes.append(torch.ones_like(offsets) if rad == 0 else torch.exp(-0.5 * (offsets / sig) ** 2))
    kernel = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return kernel.view(1, 1, *kernel.shape), radius


def _local_splat(intensity, support, quality, bounds, sigma_m, radius_voxels):
    """Spread one frame's already-warped measured evidence by a tiny metric kernel."""
    kernel, padding = _splat_kernel(
        bounds, intensity.shape[-3:], sigma_m, radius_voxels, intensity.device)
    denominator = F.conv3d(support.float(), kernel, padding=padding)
    numerator = F.conv3d((intensity.float() * support.float()), kernel, padding=padding)
    quality_sum = F.conv3d((quality.float() * support.float()), kernel, padding=padding)
    return (numerator / denominator.clamp_min(1e-8),
            denominator.clamp(0, 1),
            quality_sum / denominator.clamp_min(1e-8))


def _settings(config, frames):
    values = {
        "min_support_frames": int(config.get("min_support_frames", 2)),
        "support_threshold": float(config.get("support_threshold", 0.20)),
        "body_threshold": float(config.get("body_threshold", 0.20)),
        "confidence_threshold": float(config.get("confidence_threshold", 0.20)),
        "consistency_scale_log": float(config.get("consistency_scale_log", 0.35)),
        "uncertainty_scale_m": float(config.get("uncertainty_scale_m", 0.15)),
        "max_joint_sigma_m": float(config.get("max_joint_sigma_m", 0.30)),
        "max_intensity_cv": float(config.get("max_intensity_cv", 1.0)),
        "min_intensity": float(config.get("min_intensity", 1e-6)),
        "splat_sigma_m": config.get("splat_sigma_m", [0.05, 0.05, 0.10]),
        "splat_radius_voxels": config.get("splat_radius_voxels", [1, 1, 1]),
        "use_part_frames": bool(config.get("use_part_frames", True)),
    }
    if not 1 <= values["min_support_frames"] <= max(1, frames - 1):
        raise ValueError("min_support_frames exceeds the available historical frames")
    for name in ("support_threshold", "body_threshold", "confidence_threshold"):
        if not 0 <= values[name] <= 1:
            raise ValueError(f"{name} must lie in [0,1]")
    for name in ("consistency_scale_log", "uncertainty_scale_m", "max_joint_sigma_m"):
        if values[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if values["max_intensity_cv"] < 0 or values["min_intensity"] < 0:
        raise ValueError("max_intensity_cv and min_intensity must be non-negative")
    return values


def validate_transfer_config(config, frames):
    """Validate and normalize public Scheme-B export settings."""
    return _settings(config or {}, frames)


@torch.no_grad()
def transfer_historical_returns(raw, prediction, history_valid, bounds, capsule_radius_m,
                                config=None):
    """Transport measured history using RT-predicted joints and abstain if uncertain.

    Returns dense tensors only for the export batch.  ``values`` are weighted
    combinations of measured historical RT amplitudes; current observations are
    never included and the accepted mask is restricted to current zero voxels.
    """
    if raw.ndim != 5 or raw.shape[1] < 2:
        raise ValueError("raw must have shape B,T,X,Y,Z with at least one history frame")
    batch, frames = raw.shape[:2]
    if history_valid is None or tuple(history_valid.shape) != (batch, frames):
        raise ValueError("history_valid must have shape B,T")
    joints = prediction.get("joints_sequence")
    variances = prediction.get("joint_variance_sequence")
    if joints is None or tuple(joints.shape) != (batch, frames, 22, 3):
        raise ValueError("prediction must contain frame-only joints_sequence B,T,22,3")
    if variances is None or tuple(variances.shape) != (batch, frames, 22, 3):
        raise ValueError("prediction must contain joint_variance_sequence B,T,22,3")
    if capsule_radius_m <= 0:
        raise ValueError("capsule_radius_m must be positive")
    options = _settings(config or {}, frames)
    raw = raw.float()
    joints, variances = joints.float(), variances.float().clamp_min(0)
    target_joints, target_variance = joints[:, -1], variances[:, -1]
    target_sigma = target_variance.mean((1, 2)).sqrt()
    current_body = (_body_prior(target_joints, raw.shape[-3:], bounds, capsule_radius_m)
                    >= options["body_threshold"])

    logs, intensities, supports, qualities, uncertainty_qualities = [], [], [], [], []
    for time in range(frames - 1):
        observed = raw[:, time:time + 1] > options["min_intensity"]
        source_body = (_body_prior(joints[:, time], raw.shape[-3:], bounds, capsule_radius_m)
                       >= options["body_threshold"])
        observed &= source_body
        source_sigma = variances[:, time].mean((1, 2)).sqrt()
        pair_sigma = (.5 * (source_sigma.square() + target_sigma.square())).sqrt()
        pose_allowed = pair_sigma <= options["max_joint_sigma_m"]
        pose_quality = torch.exp(-pair_sigma / options["uncertainty_scale_m"])
        warped = articulated_warp(
            raw[:, time:time + 1] * observed,
            joints[:, time], target_joints, bounds, capsule_radius_m,
            source_support=observed.float(),
            joint_variance=torch.stack((variances[:, time], target_variance), 1),
            body_threshold=options["body_threshold"],
            source_body_mask=source_body,
            use_part_frames=options["use_part_frames"],
        )
        intensity = warped["values"] / warped["support"].clamp_min(1e-8)
        intensity, support, quality = _local_splat(
            intensity, warped["support"], warped["quality"], bounds,
            options["splat_sigma_m"], options["splat_radius_voxels"])
        usable = ((support >= options["support_threshold"])
                  & warped["in_bounds"] & current_body
                  & history_valid[:, time, None, None, None, None].bool()
                  & pose_allowed[:, None, None, None, None]
                  & (intensity > options["min_intensity"]))
        intensities.append(intensity)
        logs.append(intensity.clamp_min(0).log1p())
        supports.append(usable)
        qualities.append(quality.clamp(0, 1))
        uncertainty_qualities.append(pose_quality[:, None, None, None, None].expand_as(intensity))

    logs = torch.stack(logs, 1)
    intensities = torch.stack(intensities, 1)
    observed = torch.stack(supports, 1)
    quality = torch.stack(qualities, 1)
    uncertainty_quality = torch.stack(uncertainty_qualities, 1)
    count = observed.sum(1)

    sorted_logs = logs.masked_fill(~observed, float("inf")).sort(dim=1).values
    lower = ((count - 1).clamp_min(0) // 2).unsqueeze(1)
    upper = (count // 2).unsqueeze(1)
    center = .5 * (sorted_logs.gather(1, lower).squeeze(1)
                   + sorted_logs.gather(1, upper).squeeze(1))
    center = torch.where(count > 0, center, torch.zeros_like(center))
    residual = torch.where(observed, (logs - center.unsqueeze(1)).abs(), torch.zeros_like(logs))
    robust = (options["consistency_scale_log"]
              / residual.clamp_min(options["consistency_scale_log"]))
    weight = observed * quality * uncertainty_quality * robust
    pseudo_log = (logs * weight).sum(1) / weight.sum(1).clamp_min(1e-8)
    disagreement = residual.sum(1) / count.clamp_min(1)
    alignment_quality = (quality * observed).sum(1) / count.clamp_min(1)
    pose_quality = (uncertainty_quality * observed).sum(1) / count.clamp_min(1)

    mean_intensity = (intensities * observed).sum(1) / count.clamp_min(1)
    variance_intensity = torch.where(
        observed, (intensities - mean_intensity.unsqueeze(1)).square(),
        torch.zeros_like(intensities)).sum(1) / count.clamp_min(1)
    intensity_cv = variance_intensity.sqrt() / mean_intensity.clamp_min(1e-8)
    confidence = ((count.float() / max(1, frames - 1))
                  * alignment_quality * pose_quality
                  * torch.exp(-disagreement / options["consistency_scale_log"]))
    accepted = ((raw[:, -1:] == 0) & current_body
                & (count >= options["min_support_frames"])
                & (confidence >= options["confidence_threshold"])
                & (intensity_cv <= options["max_intensity_cv"]))
    values = pseudo_log.expm1().clamp_min(0) * accepted
    return {
        "accepted": accepted,
        "values": values,
        "confidence": confidence.clamp(0, 1) * accepted,
        "support_count": count,
        "alignment_quality": alignment_quality.clamp(0, 1),
        "intensity_disagreement_log": disagreement,
        "intensity_cv": intensity_cv,
        "joint_sigma_m": target_sigma[:, None, None, None, None].expand_as(values),
        "joints": target_joints,
    }


@torch.no_grad()
def budget_transfer(result, max_added_voxels):
    """Apply only a safety cap; confidence, not quota filling, decides additions."""
    budget = int(max_added_voxels)
    if budget < 0:
        raise ValueError("max_added_voxels must be non-negative")
    accepted = torch.zeros_like(result["accepted"])
    for batch in range(result["accepted"].shape[0]):
        indices = result["accepted"][batch].flatten().nonzero().flatten()
        if len(indices) > budget:
            score = (result["confidence"][batch] * result["values"][batch]
                     / (1 + result["intensity_cv"][batch])).flatten()[indices]
            indices = indices[score.topk(budget).indices]
        accepted[batch].view(-1)[indices] = True
    output = dict(result)
    output["accepted"] = accepted
    return output
