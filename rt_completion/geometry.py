"""All spatial arrays use X,Y,Z; grid_sample alone takes coordinates Z,Y,X."""

import torch
import torch.nn.functional as F


# Torso, neck, head; shoulders; upper/lower arms; thighs/calves; feet.
CAPSULES = ((0, 9), (9, 12), (12, 15), (9, 16), (9, 17),
            (16, 18), (18, 20), (17, 19), (19, 21),
            (1, 4), (4, 7), (2, 5), (5, 8), (7, 10), (8, 11))
RADIUS_MULTIPLIERS = (1.8, 0.8, 1.1, 0.9, 0.9, 0.8, 0.65, 0.8, 0.65,
                      1.0, 0.8, 1.0, 0.8, 0.7, 0.7)


def metric_grid(shape, bounds, device, dtype=torch.float32):
    axes = [torch.linspace(float(lo), float(hi), int(n), device=device, dtype=dtype)
            for n, (lo, hi) in zip(shape, bounds)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def capsule_heatmaps(joints, shape, bounds, radius=0.16):
    # Compute in FP32 even during mixed precision. Never render full SMPL-X here.
    joints = joints.float()
    grid = metric_grid(shape, bounds, joints.device).reshape(1, -1, 3)
    heatmaps = []
    for (start, end), multiplier in zip(CAPSULES, RADIUS_MULTIPLIERS):
        a = joints[:, start:start + 1]
        vector = joints[:, end:end + 1] - a
        projection = ((grid - a) * vector).sum(-1, keepdim=True)
        projection = (projection / vector.square().sum(-1, keepdim=True).clamp_min(1e-8)).clamp(0, 1)
        distance_sq = (grid - a - projection * vector).square().sum(-1)
        heatmaps.append(torch.exp(-distance_sq / (2 * (radius * multiplier)**2)))
    return torch.stack(heatmaps, 1).reshape(joints.shape[0], len(CAPSULES), *shape)


def warp_by_root(volume, previous_root, current_root, bounds):
    """Translation-only alignment, not an articulated or physically exact warp."""
    shape = volume.shape[-3:]
    unit_bounds = [(-1, 1)] * 3
    grid = metric_grid(shape, unit_bounds, volume.device)
    spans = volume.new_tensor([hi - lo for lo, hi in bounds])
    delta = 2 * (current_root - previous_root) / spans
    sampling_grid = grid.unsqueeze(0) - delta[:, None, None, None, :]
    return F.grid_sample(volume.float(), sampling_grid[..., [2, 1, 0]].float(),
                         mode="bilinear", padding_mode="zeros", align_corners=True)


@torch.no_grad()
def temporal_target(raw, joints, history_valid, bounds, radius, min_support=2):
    """Weak positive evidence only: two distinct, aligned historical returns.

    Zero measurements are unknown, not negative GT. Gate historical and current
    volumes by GT capsules FOR SUPERVISION ONLY, to reduce background/ghosting.
    """
    batch, frames = raw.shape[:2]
    shape = raw.shape[-3:]
    coarse_shape = tuple(max(2, (n + 3) // 4) for n in shape)
    priors = capsule_heatmaps(joints.reshape(-1, 22, 3), coarse_shape, bounds, radius)
    priors = F.interpolate(priors.amax(1, keepdim=True), size=shape,
                           mode="trilinear", align_corners=True).reshape(batch, frames, *shape)
    total = torch.zeros_like(raw[:, -1:])
    count = torch.zeros_like(total)
    for time in range(frames - 1):
        history = raw[:, time:time + 1] * (priors[:, time:time + 1] > 0.25)
        aligned = warp_by_root(history, joints[:, time, 0], joints[:, -1, 0], bounds)
        support = (aligned > 1e-6) & history_valid[:, time, None, None, None, None].bool()
        total += aligned * support
        count += support
    mask = (count >= min_support) & (raw[:, -1:] == 0) & (priors[:, -1:] > 0.25)
    return total / count.clamp_min(1), mask


def _body_prior(joints, shape, bounds, radius):
    """Maximum capsule proximity without allocating all 15 dense heatmaps."""
    grid = metric_grid(shape, bounds, joints.device).reshape(1, -1, 3)
    best = torch.full((joints.shape[0], grid.shape[1]), float("inf"),
                      device=joints.device, dtype=torch.float32)
    for (start, end), multiplier in zip(CAPSULES, RADIUS_MULTIPLIERS):
        a = joints[:, start:start + 1].float()
        vector = joints[:, end:end + 1].float() - a
        projection = (((grid - a) * vector).sum(-1, keepdim=True)
                      / vector.square().sum(-1, keepdim=True).clamp_min(1e-8)).clamp(0, 1)
        distance = (grid - a - projection * vector).square().sum(-1)
        best = torch.minimum(best, distance / (radius * multiplier)**2)
    return torch.exp(-.5 * best).reshape(joints.shape[0], 1, *shape)


def _joint_variance_scalar(variance, batch):
    if variance is None:
        return None
    variance = variance.float()
    # A pair of source/target diagonal variances is also accepted. A single
    # B,22,3 array describes source localization variance in square meters.
    if variance.ndim == 4 and variance.shape[1:3] == (2, 22):
        variance = variance.mean(-1).sum(1)
    elif variance.ndim == 3 and variance.shape[1:] == (2, 22):
        variance = variance.sum(1)
    elif variance.ndim == 3 and variance.shape[1:] == (22, 3):
        variance = variance.mean(-1)
    if variance.shape != (batch, 22):
        raise ValueError("joint_variance must have shape B,22, B,22,3 or B,2,22[,3]")
    return variance.clamp_min(0)


def _rotate_between_axes(offset, source_axis, destination_axis):
    """Minimal rotation, with a finite deterministic 180-degree fallback."""
    cross = torch.cross(source_axis, destination_axis, dim=-1)
    cosine = (source_axis * destination_axis).sum(-1, keepdim=True).clamp(-1, 1)
    cross_offset = torch.cross(cross, offset, dim=-1)
    rotated = offset + cross_offset + torch.cross(cross, cross_offset, dim=-1) / (1 + cosine).clamp_min(1e-4)
    # Two endpoints do not determine axial twist. Near antiparallel axes the
    # minimal rotation is ambiguous; choose an orthogonal axis reproducibly.
    basis = F.one_hot(source_axis.abs().argmin(-1), num_classes=3).float()
    axis = F.normalize(torch.cross(source_axis, basis, dim=-1), dim=-1, eps=1e-8)
    flipped = 2 * axis * (axis * offset).sum(-1, keepdim=True) - offset
    return torch.where(cosine < -.9999, flipped, rotated)


def articulated_warp(volume, source_joints, target_joints, bounds, radius=0.16,
                     source_support=None, joint_variance=None, body_threshold=0.25,
                     source_body_mask=None):
    """Inverse-warp a source volume using 15 approximately rigid body parts.

    Each target voxel selects its nearest *normalized* target capsule. Its
    offset is rotated into the corresponding source bone and scaled only along
    the bone. This is a minimal-twist approximation, NOT a recovered full body
    rotation: two endpoints cannot determine rotation around a bone's axis.
    The discrete part assignment is piecewise differentiable; the selected
    sampling coordinates and sampled values retain gradients to both poses.

    Only one composite sampling grid is used, not 15 full warped volumes.
    Support is sampled as a separate channel using the same grid as values and
    remains soft (0..1), so callers can reject interpolation dust with an
    explicit support threshold. ``source_body_mask`` may reuse an already
    computed B,1,X,Y,Z boolean capsule ROI for the same source pose, bounds,
    radius and threshold; it does not replace the observed ``source_support``.
    Degenerate bones and source/target length ratios outside [.5,2] are rejected.
    ``quality`` is a heuristic geometric weight, not a calibrated probability.
    Spatial tensors use X,Y,Z. All geometry and grid_sample run in FP32.
    """
    if (volume.ndim != 5 or source_joints.shape != target_joints.shape
            or source_joints.shape[1:] != (22, 3) or source_joints.shape[0] != volume.shape[0]):
        raise ValueError("expected volume B,C,X,Y,Z and source/target joints B,22,3")
    if radius <= 0 or not 0 <= body_threshold <= 1:
        raise ValueError("radius must be positive and body_threshold must be in [0,1]")
    batch, _, *shape = volume.shape
    with torch.autocast(device_type=volume.device.type, enabled=False):
        source, target = source_joints.float(), target_joints.float()
        points = metric_grid(shape, bounds, volume.device).reshape(1, -1, 3).expand(batch, -1, -1)
        count = points.shape[1]
        best = torch.full((batch, count), float("inf"), device=volume.device)
        sample_points = points.clone()
        selected_quality = torch.zeros_like(best)
        selected_valid = torch.zeros_like(best, dtype=torch.bool)
        variance = _joint_variance_scalar(joint_variance, batch)
        for (start, end), multiplier in zip(CAPSULES, RADIUS_MULTIPLIERS):
            a_t, a_s = target[:, start:start + 1], source[:, start:start + 1]
            bone_t = target[:, end:end + 1] - a_t
            bone_s = source[:, end:end + 1] - a_s
            length_t = bone_t.norm(dim=-1, keepdim=True)
            length_s = bone_s.norm(dim=-1, keepdim=True)
            axis_t = bone_t / length_t.clamp_min(1e-8)
            axis_s = bone_s / length_s.clamp_min(1e-8)
            ratio = length_s / length_t.clamp_min(1e-8)
            valid_bone = ((length_t > 1e-4) & (length_s > 1e-4)
                          & (ratio >= .5) & (ratio <= 2)).squeeze(-1)
            offset = points - a_t
            longitudinal = (offset * axis_t).sum(-1, keepdim=True)
            projection = (longitudinal / length_t.clamp_min(1e-8)).clamp(0, 1)
            distance = (offset - projection * bone_t).square().sum(-1)
            normalized_distance = distance / (radius * multiplier)**2
            better = (normalized_distance < best) & valid_bone
            radial = offset - longitudinal * axis_t
            mapped = (a_s + _rotate_between_axes(radial, axis_t, axis_s)
                      + longitudinal * ratio * axis_s)
            # Moderate length uncertainty reduces weight without imposing a
            # hard equality on bone lengths or suppressing genuine rotation.
            bone_quality = torch.exp(-ratio.clamp_min(1e-8).log().abs().squeeze(-1) / .5)
            if variance is not None:
                local_variance = .5 * (variance[:, start] + variance[:, end])
                bone_quality = bone_quality / (1 + local_variance[:, None] / (radius * multiplier)**2)
            best = torch.where(better, normalized_distance, best)
            sample_points = torch.where(better[..., None], mapped, sample_points)
            selected_quality = torch.where(better, bone_quality.expand_as(best), selected_quality)
            selected_valid |= better
        target_body = torch.exp(-.5 * best) >= body_threshold
        lows = points.new_tensor([pair[0] for pair in bounds])
        spans = points.new_tensor([pair[1] - pair[0] for pair in bounds])
        grid = 2 * (sample_points - lows) / spans - 1
        in_bounds = (grid.abs() <= 1 + 1e-6).all(-1) & target_body & selected_valid
        sampling_grid = grid.reshape(batch, *shape, 3)[..., [2, 1, 0]]
        # Source ROI prevents unrelated measured clutter from entering through
        # trilinear interpolation at capsule borders.
        if source_body_mask is None:
            source_body = _body_prior(source, shape, bounds, radius) >= body_threshold
        else:
            if (source_body_mask.shape != (batch, 1, *shape)
                    or source_body_mask.dtype != torch.bool
                    or source_body_mask.device != volume.device):
                raise ValueError("source_body_mask must be a boolean B,1,X,Y,Z tensor on the volume device")
            source_body = source_body_mask
        if source_support is None:
            support = (volume.float().abs().sum(1, keepdim=True) > 0).float()
        else:
            if source_support.shape != (batch, 1, *shape):
                raise ValueError("source_support must have shape B,1,X,Y,Z")
            support = source_support.float().clamp(0, 1)
        support = support * source_body
        # Sampling is independent across channels. Concatenating the observed
        # support avoids a second grid_sample launch and grid traversal while
        # preserving the differentiable grid for feature/pose optimization.
        sampled = F.grid_sample(torch.cat((volume.float(), support), dim=1), sampling_grid,
                                mode="bilinear", padding_mode="zeros", align_corners=True)
        warped_values, warped_support = sampled[:, :-1], sampled[:, -1:]
        valid = in_bounds.reshape(batch, 1, *shape)
        return {"values": warped_values * valid,
                "support": warped_support.clamp(0, 1) * valid,
                "in_bounds": valid,
                "quality": selected_quality.reshape(batch, 1, *shape) * valid}


@torch.no_grad()
def temporal_target_v2(raw, joints, history_valid, bounds, radius, config):
    """Stop-gradient, body-aligned weak supervision from distinct past frames.

    Returns raw-unit ``intensity`` plus ``valid_mask``, heuristic ``confidence``
    and integer ``support_count``; all shapes are B,1,X,Y,Z. A missing/invalid
    target is ignored, NEVER treated as measured zero. Original current-frame
    observations always take precedence over this teacher. ``history_valid``
    must exclude repeated window-padding frames; identical intensities in truly
    different frames are still independent temporal supports.
    """
    batch, frames = raw.shape[:2]
    if joints.shape != (batch, frames, 22, 3) or history_valid.shape != (batch, frames):
        raise ValueError("expected joints B,T,22,3 and history_valid B,T")
    min_support = int(config.get("temporal_min_support", 2))
    support_threshold = float(config.get("temporal_support_threshold", .25))
    body_threshold = float(config.get("temporal_body_threshold", .25))
    consistency_scale = float(config.get("temporal_consistency_scale", .35))
    alignment_scale = float(config.get("temporal_alignment_scale", .5))
    min_intensity = float(config.get("temporal_min_intensity", 1e-6))
    if min_support < 1 or not 0 < support_threshold <= 1 or consistency_scale <= 0 or alignment_scale <= 0:
        raise ValueError("invalid temporal teacher support/scale settings")
    empty = torch.zeros_like(raw[:, -1:], dtype=torch.float32)
    if frames < 2:
        return {"intensity": empty, "valid_mask": empty.bool(), "confidence": empty.clone(),
                "support_count": empty.long()}
    aligned_logs, supports, qualities = [], [], []
    for time in range(frames - 1):
        observed = raw[:, time:time + 1].float() > min_intensity
        source_body = _body_prior(joints[:, time].float(), raw.shape[-3:], bounds, radius) >= body_threshold
        observed &= source_body
        warped = articulated_warp(raw[:, time:time + 1].float() * observed,
                                  joints[:, time], joints[:, -1], bounds, radius,
                                  source_support=observed.float(), body_threshold=body_threshold,
                                  source_body_mask=source_body)
        usable = ((warped["support"] >= support_threshold) & warped["in_bounds"]
                  & history_valid[:, time, None, None, None, None].bool())
        # Divide sparse interpolated amplitude by interpolated observed support;
        # otherwise a half-voxel displacement artificially halves the signal.
        intensity = warped["values"] / warped["support"].clamp_min(1e-8)
        usable &= intensity > min_intensity
        aligned_logs.append(intensity.clamp_min(0).log1p())
        supports.append(usable)
        # quality already contains a length-change penalty; rescaling allows
        # the teacher's tolerance to be tuned independently of feature fusion.
        qualities.append(warped["quality"].clamp_min(1e-8).pow(.5 / alignment_scale))
    logs, observed, quality = map(lambda items: torch.stack(items, 1),
                                  (aligned_logs, supports, qualities))
    count = observed.sum(1)
    # An even number of histories uses the mean of the central pair, rather
    # than torch.median's lower median (which would bias two-frame targets).
    sorted_logs = logs.masked_fill(~observed, float("inf")).sort(dim=1).values
    lower = ((count - 1).clamp_min(0) // 2).unsqueeze(1)
    upper = (count // 2).unsqueeze(1)
    center = .5 * (sorted_logs.gather(1, lower).squeeze(1)
                   + sorted_logs.gather(1, upper).squeeze(1))
    center = torch.where(count > 0, center, torch.zeros_like(center))
    residual = (logs - center.unsqueeze(1)).abs()
    robust_weight = (consistency_scale / residual.clamp_min(consistency_scale)) * observed * quality
    pseudo_log = (logs * robust_weight).sum(1) / robust_weight.sum(1).clamp_min(1e-8)
    disagreement = (residual * observed).sum(1) / count.clamp_min(1)
    confidence = ((count.float() / max(frames - 1, 1))
                  * (quality * observed).sum(1) / count.clamp_min(1)
                  * torch.exp(-disagreement / consistency_scale))
    current_body = _body_prior(joints[:, -1].float(), raw.shape[-3:], bounds, radius) >= body_threshold
    valid = (count >= min_support) & (raw[:, -1:] == 0) & current_body
    return {"intensity": pseudo_log.expm1().clamp_min(0) * valid,
            "valid_mask": valid, "confidence": confidence.clamp(0, 1) * valid,
            "support_count": count}
