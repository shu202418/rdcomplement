"""Self-supervision + weak temporal positives; no dense radar GT is invented."""

import torch
import torch.nn.functional as F

from .geometry import CAPSULES, RADIUS_MULTIPLIERS, temporal_target


def _legacy_corrupt_rt(raw, config, generator):
    corrupted = raw.clone()
    observed = raw[:, -1] > 0
    random_mask = torch.rand(observed.shape, device=raw.device, generator=generator) < config["drop_probability"]
    corrupted[:, -1].masked_fill_(observed & random_mask, 0)
    # Remove one spatial cuboid in ALL frames, centred on an actual target
    # return. This also teaches recovery when temporal copying is impossible.
    for batch in range(raw.shape[0]):
        if torch.rand((), device=raw.device, generator=generator).item() >= config["block_probability"]:
            continue
        coords = observed[batch].nonzero()
        if not len(coords):
            continue
        choice = torch.randint(len(coords), (), device=raw.device, generator=generator).item()
        centre = coords[choice].tolist()
        slices = tuple(slice(max(0, c - n // 2), min(s, c + (n + 1) // 2))
                       for c, n, s in zip(centre, config["block_shape"], raw.shape[-3:]))
        corrupted[(batch, slice(None), *slices)] = 0
    # A sparse/small validation frame can otherwise receive no random mask,
    # giving a misleading zero restoration error. Guarantee one held-out return
    # whenever corruption is enabled and the target contains observations.
    if config["drop_probability"] > 0 or config["block_probability"] > 0:
        for batch in range(raw.shape[0]):
            if not (observed[batch] & (corrupted[batch, -1] == 0)).any():
                coords = observed[batch].nonzero()
                if len(coords):
                    choice = torch.randint(len(coords), (), device=raw.device, generator=generator).item()
                    corrupted[(batch, -1, *coords[choice].tolist())] = 0
    masked = observed & (corrupted[:, -1] == 0)
    return corrupted, masked.unsqueeze(1)


MASK_GROUPS = ("random", "single_part", "continuous_part", "spatial_block")


def _metric_points(indices, shape, bounds):
    """Convert XYZ indices to metres; notably, Z need not have XY spacing."""
    lo = indices.new_tensor([pair[0] for pair in bounds], dtype=torch.float32)
    span = indices.new_tensor([pair[1] - pair[0] for pair in bounds], dtype=torch.float32)
    denominator = indices.new_tensor([max(1, n - 1) for n in shape], dtype=torch.float32)
    return lo + indices.float() * span / denominator


def _part_membership(points, joints, radii):
    """N observed positions x 15 bone capsules, without allocating a dense mask."""
    starts = joints[[pair[0] for pair in CAPSULES]].float()
    vectors = joints[[pair[1] for pair in CAPSULES]].float() - starts
    offset = points[:, None, :] - starts[None]
    projection = ((offset * vectors[None]).sum(-1)
                  / vectors.square().sum(-1).clamp_min(1e-8)).clamp(0, 1)
    distance_sq = (offset - projection[..., None] * vectors[None]).square().sum(-1)
    return distance_sq <= radii.square()[None]


def corrupt_rt(raw, config, generator, *, joints=None, bounds=None, group=None, return_info=False):
    """Artificially hold out *measured* returns, never invent dense empty GT.

    The historical API is preserved. ``mode: mixed_parts`` enables a single
    named corruption per sample. Training labels determine the capsule mask,
    but are never returned as forward-model inputs. A part group with no
    measured support is marked ineligible instead of silently becoming random.
    ``radius_jitter`` is relative; ``joint_jitter_m`` is in metres. Set both to
    zero for the fixed validation groups and use a per-sample seeded generator.
    """
    if config.get("mode", "legacy") != "mixed_parts" and group is None:
        corrupted, masked = _legacy_corrupt_rt(raw, config, generator)
        if not return_info:
            return corrupted, masked
        batch = raw.shape[0]
        info = {"kind": ["legacy"] * batch,
                "part_index": torch.full((batch,), -1, device=raw.device, dtype=torch.long),
                "duration": torch.full((batch,), raw.shape[1], device=raw.device, dtype=torch.long),
                "eligible": masked.flatten(1).any(1)}
        return corrupted, masked, info
    if group is not None and group not in MASK_GROUPS:
        raise ValueError(f"Unknown RT mask group: {group}")
    if raw.ndim != 5:
        raise ValueError("RT must have shape [B,T,X,Y,Z]")
    batch_size, frames = raw.shape[:2]
    corrupted = raw.clone()
    observed = raw[:, -1] > 0
    info = {"kind": [],
            "part_index": torch.full((batch_size,), -1, device=raw.device, dtype=torch.long),
            "duration": torch.zeros(batch_size, device=raw.device, dtype=torch.long)}
    weights = config.get("group_weights", {"random": .35, "single_part": .3,
                                          "continuous_part": .25, "spatial_block": .1})
    probabilities = raw.new_tensor([float(weights.get(name, 0.)) for name in MASK_GROUPS])
    if group is None and (not torch.isfinite(probabilities).all() or
                          (probabilities < 0).any() or probabilities.sum() <= 0):
        raise ValueError("mask.group_weights must be finite, nonnegative, and have positive total")
    if joints is not None and joints.shape != (batch_size, frames, 22, 3):
        raise ValueError("Mask joints must have shape [B,T,22,3]")
    for batch in range(batch_size):
        kind = group or MASK_GROUPS[int(torch.multinomial(probabilities, 1, generator=generator))]
        info["kind"].append(kind)
        coords = observed[batch].nonzero()
        if kind == "random":
            probability = float(config.get("drop_probability", .25))
            selected = torch.rand(len(coords), device=raw.device, generator=generator) < probability
            if probability > 0 and len(coords) and not selected.any():
                selected[torch.randint(len(coords), (), device=raw.device, generator=generator)] = True
            chosen = coords[selected]
            if len(chosen):
                corrupted[batch, -1, chosen[:, 0], chosen[:, 1], chosen[:, 2]] = 0
            info["duration"][batch] = 1
        elif kind == "spatial_block":
            info["duration"][batch] = frames
            if not len(coords):
                continue
            centre = coords[int(torch.randint(len(coords), (), device=raw.device, generator=generator))].tolist()
            slices = tuple(slice(max(0, c - n // 2), min(s, c + (n + 1) // 2))
                           for c, n, s in zip(centre, config.get("block_shape", [15, 15, 9]), raw.shape[-3:]))
            corrupted[(batch, slice(None), *slices)] = 0
        else:
            if joints is None or bounds is None:
                raise ValueError("Part masks require radar-coordinate joints and metric bounds")
            if not len(coords):
                continue
            joint = joints[batch].detach().float().clone()
            jitter_m = float(config.get("joint_jitter_m", 0.))
            if jitter_m > 0:
                joint += torch.randn(joint.shape, device=raw.device, generator=generator) * jitter_m
            radius = float(config.get("part_radius", .16))
            relative_jitter = float(config.get("radius_jitter", 0.))
            factor = 1. + (2 * torch.rand((), device=raw.device, generator=generator) - 1) * relative_jitter
            radii = raw.new_tensor(RADIUS_MULTIPLIERS) * radius * factor.clamp_min(.05)
            current_points = _metric_points(coords, raw.shape[-3:], bounds)
            membership = _part_membership(current_points, joint[-1], radii)
            eligible_parts = membership.any(0).nonzero().flatten()
            if not len(eligible_parts):
                continue
            choice = int(torch.randint(len(eligible_parts), (), device=raw.device, generator=generator))
            part = int(eligible_parts[choice])
            info["part_index"][batch] = part
            if kind == "single_part":
                duration = 1
            else:
                minimum = min(frames, max(1, int(config.get("continuous_min_frames", 2))))
                maximum = min(frames, max(minimum, int(config.get("continuous_max_frames", frames))))
                duration = int(torch.randint(minimum, maximum + 1, (), device=raw.device, generator=generator))
            info["duration"][batch] = duration
            for time in range(frames - duration, frames):
                indices = (raw[batch, time] > 0).nonzero()
                if not len(indices):
                    continue
                points = _metric_points(indices, raw.shape[-3:], bounds)
                selected = _part_membership(points, joint[time], radii)[:, part]
                indices = indices[selected]
                corrupted[batch, time, indices[:, 0], indices[:, 1], indices[:, 2]] = 0
    masked = (observed & (corrupted[:, -1] == 0)).unsqueeze(1)
    info["eligible"] = masked.flatten(1).any(1)
    if return_info:
        return corrupted, masked, info
    return corrupted, masked


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1)


def _legacy_completion_loss(output, raw, masked, joints, history_valid, model, config):
    current = raw[:, -1:]
    observed = current > 0
    target = model.normalize(current)
    weights = masked.float() + (observed & ~masked).float() * config["observed_weight"]
    error = output["intensity"] - target
    logvar = output["log_variance"]
    nll = 0.5 * (error.square() * (-logvar).exp() + logvar)
    reconstruction = weighted_mean(nll, weights)
    positive_bce = F.softplus(-output["reflection_logits"])
    reflection = weighted_mean(positive_bce, weights)

    temporal = output["intensity"].sum() * 0
    temporal_mask = torch.zeros_like(observed)
    if config["temporal_weight"] > 0:
        pseudo, temporal_mask = temporal_target(
            raw, joints, history_valid, model.bounds, model.radius, config["temporal_min_support"])
        # Robust L1: temporal returns are only approximate targets, not noise-
        # free truth used to calibrate predicted uncertainty.
        temporal = weighted_mean((output["intensity"] - model.normalize(pseudo)).abs() + positive_bce,
                                 temporal_mask.float())
    # Weak sparsity regularization, NOT a supervised "empty voxel" label.
    unknown = (~observed & ~temporal_mask).float()
    sparsity = weighted_mean(output["reflection_logits"].sigmoid(), unknown)
    joint_loss = F.smooth_l1_loss(output["joints"], joints[:, -1], beta=0.05)
    total = (reconstruction + config["reflection_weight"] * reflection
             + config["temporal_weight"] * temporal + config["joint_weight"] * joint_loss
             + config["sparsity_weight"] * sparsity)
    metrics = {
        "loss": total.detach(), "reconstruction_nll": reconstruction.detach(),
        "reflection": reflection.detach(), "temporal": temporal.detach(),
        "sparsity": sparsity.detach(), "joint_loss": joint_loss.detach(),
        "masked_mae_log": weighted_mean(error.abs(), masked.float()).detach(),
        "joint_mpjpe_m": (output["joints"] - joints[:, -1]).norm(dim=-1).mean().detach(),
        "masked_voxels": masked.sum().detach(), "temporal_voxels": temporal_mask.sum().detach(),
    }
    return total, metrics


def _joint_objectives(prediction, variance, labels, valid_frames, variance_floor):
    valid = valid_frames[..., None] & torch.isfinite(labels).all(-1)
    target = torch.nan_to_num(labels.detach())
    weights = valid[..., None].expand_as(prediction).float()
    error = prediction.float() - target
    smooth = weighted_mean(F.smooth_l1_loss(prediction.float(), target, beta=.05, reduction="none"), weights)
    if variance is None:
        nll = prediction.sum() * 0
    else:
        variance = variance.float().clamp_min(variance_floor).expand_as(prediction)
        nll = weighted_mean(.5 * (error.square() / variance + variance.log()), weights)
    return smooth, nll


def completion_loss(output, raw, masked, joints, history_valid, model, config, *, temporal_teacher=None):
    """V2 weak supervision; the original baseline remains byte-for-byte in its path."""
    use_v2 = config.get("temporal_alignment") == "articulated" or "joints_sequence" in output
    if not use_v2:
        if temporal_teacher is not None:
            raise ValueError("A cached articulated teacher is only supported for V2 supervision")
        return _legacy_completion_loss(output, raw, masked, joints, history_valid, model, config)
    current = raw[:, -1:]
    observed = current > 0
    # Recompute held-out validity from the original current measurements. A
    # synthetic mask cannot turn a naturally unknown zero into real GT.
    masked = masked.bool() & observed
    target = model.normalize(current).detach()
    weights = masked.float() + (observed & ~masked).float() * config.get("observed_weight", .05)
    error = output["intensity"].float() - target
    logvar = output["log_variance"].float().clamp(-12, 12)
    reconstruction = weighted_mean(.5 * (error.square() * (-logvar).exp() + logvar), weights)
    positive_bce = F.softplus(-output["reflection_logits"].float())
    reflection = weighted_mean(positive_bce, weights)

    zero = output["intensity"].sum() * 0
    temporal = zero
    temporal_mask = torch.zeros_like(observed)
    confidence = torch.zeros_like(current)
    support_count = torch.zeros_like(current)
    if config.get("temporal_weight", 0.) > 0:
        if config.get("temporal_alignment", "articulated") == "articulated":
            if temporal_teacher is None:
                from .geometry import temporal_target_v2
                pseudo = temporal_target_v2(raw, joints.detach(), history_valid, model.bounds, model.radius, config)
            else:
                pseudo = temporal_teacher
                for name in ("intensity", "valid_mask", "confidence", "support_count"):
                    if name not in pseudo or pseudo[name].shape != current.shape:
                        raise ValueError(f"Invalid cached temporal teacher shape/key: {name}")
                    if pseudo[name].device != raw.device:
                        raise ValueError("Cached temporal teacher must be on the same device as RT")
            pseudo_intensity = pseudo["intensity"].detach()
            confidence = torch.nan_to_num(pseudo["confidence"].detach().float()).clamp(0, 1)
            support_count = pseudo["support_count"].detach().float()
            temporal_mask = (pseudo["valid_mask"].detach().bool() & ~observed
                             & torch.isfinite(pseudo_intensity))
        else:
            pseudo_intensity, temporal_mask = temporal_target(
                raw, joints, history_valid, model.bounds, model.radius,
                config.get("temporal_min_support", 2))
            confidence = temporal_mask.float()
        # Real current returns (even artificially hidden ones) always outrank
        # a pseudo target, independent of the target builder implementation.
        temporal_mask = temporal_mask & ~observed
        pseudo_target = model.normalize(torch.nan_to_num(pseudo_intensity)).detach()
        residual = F.smooth_l1_loss(output["intensity"].float(), pseudo_target,
                                   beta=config.get("temporal_huber_beta", .1), reduction="none")
        temporal = ((residual + positive_bce) * temporal_mask * confidence).sum() / temporal_mask.sum().clamp_min(1)
    unknown = (~observed & ~temporal_mask).float()
    sparsity = weighted_mean(output["reflection_logits"].sigmoid(), unknown)

    valid_frames = history_valid.bool().clone()
    valid_frames[:, -1] = True
    floor = float(config.get("joint_variance_floor", 1e-4))
    current_loss, _ = _joint_objectives(output["joints"][:, None], None, joints[:, -1:],
                                       valid_frames[:, -1:], floor)
    joint_terms = [current_loss]
    joint_nll_terms = []
    for prefix in ("", "temporal_"):
        key = prefix + "joints_sequence"
        if key not in output:
            continue
        smooth, nll = _joint_objectives(output[key], output.get(prefix + "joint_variance_sequence"),
                                         joints, valid_frames, floor)
        joint_terms.append(smooth)
        joint_nll_terms.append(nll)
    joint_loss = torch.stack(joint_terms).mean()
    joint_nll = torch.stack(joint_nll_terms).mean() if joint_nll_terms else zero

    # Independently supervise historical quality with artificially held-out
    # current measurements, not by multiplying the task loss by learned trust.
    # These are coarse conditional-mean proxy labels, not physical SNR labels.
    quality_loss = zero
    quality_count = zero.detach()
    quality_keys = ("history_quality_logits", "aligned_history_intensity", "history_support")
    if all(key in output for key in quality_keys):
        logits, aligned, support = (output[key] for key in quality_keys)
        shape = logits.shape[-3:]
        target_support = F.adaptive_avg_pool3d(masked.float(), shape)
        heldout_mean = F.adaptive_avg_pool3d(target * masked, shape) / target_support.clamp_min(1e-8)
        heldout_mean = heldout_mean[:, None]
        valid_quality = ((target_support[:, None] > 0) &
                         (support.detach() > config.get("quality_support_threshold", .01)))
        quality_target = torch.exp(-(aligned.detach() - heldout_mean).abs()
                                   / max(float(config.get("quality_error_scale", .15)), 1e-6)).detach()
        quality_loss = weighted_mean(F.binary_cross_entropy_with_logits(logits.float(), quality_target.float(),
                                                                       reduction="none"), valid_quality.float())
        quality_count = valid_quality.sum().detach()

    total = (reconstruction + config.get("reflection_weight", .1) * reflection
             + config.get("temporal_weight", .1) * temporal
             + config.get("joint_weight", 1.) * joint_loss
             + config.get("joint_uncertainty_weight", .01) * joint_nll
             + config.get("sparsity_weight", .03) * sparsity
             + config.get("quality_weight", .05) * quality_loss)
    sigma = (.5 * logvar.detach()).exp()
    standardized = error.detach().abs() / sigma.clamp_min(1e-8)
    masked_float = masked.float()
    coverage68_sum = ((standardized <= 1.).float() * masked_float).sum()
    coverage95_sum = ((standardized <= 1.96).float() * masked_float).sum()
    finite_current = torch.isfinite(joints[:, -1]).all(-1)
    joint_distance = (output["joints"].detach() - torch.nan_to_num(joints[:, -1])).norm(dim=-1)
    metrics = {
        "loss": total.detach(), "reconstruction_nll": reconstruction.detach(),
        "reflection": reflection.detach(), "temporal": temporal.detach(),
        "sparsity": sparsity.detach(), "joint_loss": joint_loss.detach(), "joint_nll": joint_nll.detach(),
        "masked_mae_log": weighted_mean(error.abs(), masked_float).detach(),
        "joint_mpjpe_m": weighted_mean(joint_distance, finite_current.float()).detach(),
        "masked_voxels": masked.sum().detach(), "temporal_voxels": temporal_mask.sum().detach(),
        "masked_abs_error_sum": (error.detach().abs() * masked_float).sum(),
        "masked_coverage68_sum": coverage68_sum, "masked_coverage95_sum": coverage95_sum,
        "masked_sigma_sum": (sigma * masked_float).sum(),
        "coverage68": coverage68_sum / masked.sum().clamp_min(1),
        "coverage95": coverage95_sum / masked.sum().clamp_min(1),
        "sigma_log": weighted_mean(sigma, masked_float),
        "temporal_confidence": weighted_mean(confidence, temporal_mask.float()).detach(),
        "temporal_support": weighted_mean(support_count, temporal_mask.float()).detach(),
        "temporal_coverage": (temporal_mask.sum() / (~observed).sum().clamp_min(1)).detach(),
        "history_quality_loss": quality_loss.detach(), "history_quality_voxels": quality_count,
    }
    if "history_null" in output:
        metrics["history_null"] = output["history_null"].detach().float().mean()
    return total, metrics
