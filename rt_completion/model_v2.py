"""RT-only articulated temporal completion with an explicit current-only fallback.

Reliability values here are task-trained scores, not calibrated physical radar
probabilities.  Neither joint labels nor training-mask geometry enter forward().
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import articulated_warp, capsule_heatmaps, metric_grid
from .model import ConvGRU3D, GatedGeometry, block


class LocalHistoryFusion(nn.Module):
    """Softmax over real histories AND an always-valid, zero-valued null entry."""

    def __init__(self, channels):
        super().__init__()
        inner = max(4, channels // 2)
        self.score = nn.Sequential(
            nn.Conv3d(3 * channels + 3, inner, 1), nn.SiLU(),
            nn.Conv3d(inner, 1, 1),
        )
        self.null_score = nn.Conv3d(channels, 1, 1)
        self.value = nn.Conv3d(channels, channels, 1, bias=False)

    def forward(self, current, histories, supports, qualities, validities):
        scores, values, quality_logits = [], [], []
        count = len(histories)
        for index, (history, support, quality, valid) in enumerate(
                zip(histories, supports, qualities, validities)):
            # Age is dimensionless and bounded; it is evidence, not a hard rule
            # that the most recent frame must receive the largest weight.
            age = current.new_full(support.shape, (count - index) / max(1, count))
            inputs = torch.cat((current, history, (current - history).abs(),
                                support.to(current.dtype), quality.to(current.dtype), age), 1)
            learned = self.score(inputs).float()
            score = learned + quality.float().clamp_min(1e-6).log()
            scores.append(score.masked_fill(~valid, -torch.inf))
            quality_logits.append(learned)
            values.append(self.value(history))
        scores.append(self.null_score(current).float())
        attention = torch.softmax(torch.stack(scores, 1), dim=1)
        # Crucially: do not renormalize the real-frame weights after removing
        # null. Their total mass can be zero, allowing genuine abstention.
        history_delta = torch.zeros_like(current)
        for index, value in enumerate(values):
            history_delta = history_delta + attention[:, index].to(value.dtype) * value
        mass = attention[:, :-1].sum(1)
        if quality_logits:
            quality_logits = torch.stack(quality_logits, 1)
        else:
            quality_logits = current.new_empty((current.shape[0], 0, 1, *current.shape[-3:]))
        return current + history_delta, mass, attention, quality_logits


class ReliabilityRTCompletionNet(nn.Module):
    def __init__(self, data, model):
        super().__init__()
        if model.get("encoder", "dense") != "dense":
            raise ValueError("reliability_v2 currently supports encoder=dense only")
        channels = int(model["base_channels"])
        self.bounds = data["bounds"]
        self.scale = float(data["intensity_scale"])
        self.cap = float(data["intensity_cap"])
        self.radius = float(model["capsule_radius_m"])
        self.variance_floor = float(model.get("joint_variance_floor", 1e-4))
        self.support_threshold = float(model.get("feature_support_threshold", .01))
        self.feature_radius_scale = float(model.get("feature_radius_scale", 1.))
        if self.feature_radius_scale < 0:
            raise ValueError("feature_radius_scale must be nonnegative")
        self.encoder1 = block(2, channels, stride=2)
        self.encoder2 = block(channels, channels * 2, stride=2)
        self.encoder3 = block(channels * 2, channels * 4, stride=2)
        self.gru = ConvGRU3D(channels * 4)
        self.joint_logits = nn.Conv3d(channels * 4, 22, 1)
        self.history1 = LocalHistoryFusion(channels)
        self.history2 = LocalHistoryFusion(channels * 2)
        self.history3 = LocalHistoryFusion(channels * 4)
        self.temporal_delta = nn.Conv3d(channels * 4, channels * 4, 1)
        self.fuse3 = GatedGeometry(channels * 4)
        self.fuse2 = GatedGeometry(channels * 2)
        self.fuse1 = GatedGeometry(channels)
        self.decode2 = block(channels * 6, channels * 2)
        self.decode1 = block(channels * 3, channels)
        self.decode0 = block(channels + 2, channels)
        self.head = nn.Conv3d(channels, 3, 1)
        # Conservative numerical initialization, not a calibrated physical
        # prior. Sparse inputs can create large local GroupNorm activations;
        # a default random head could pair huge intensity with tiny variance
        # and make heteroscedastic NLL explode before learning begins.
        nn.init.normal_(self.head.weight, mean=0., std=.001)
        with torch.no_grad():
            self.head.bias.copy_(self.head.bias.new_tensor(
                [-2., math.log(math.expm1(.1)), 2 * math.log(.5)]))

    def _joint_distribution(self, feature):
        logits = self.joint_logits(feature)
        # Casting inputs to float is insufficient under outer CUDA autocast:
        # matmul could still choose FP16 and corrupt second-moment subtraction.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            heat = logits.float().flatten(2).softmax(-1)
            grid = metric_grid(feature.shape[-3:], self.bounds, feature.device).reshape(-1, 3)
            mean = heat @ grid
            variance = (heat @ grid.square() - mean.square()).clamp_min(0) + self.variance_floor
        return mean, variance

    def _feature_radius(self, shape):
        # A coarse voxel represents a finite spatial cell, not an infinitesimal
        # point. Without dilation e.g. 1 m coarse Z spacing can miss every .16 m
        # body capsule, permanently starving the coarsest history/GRU gate.
        half_diagonal_squared = sum(
            (.5 * (high - low) / max(1, size - 1)) ** 2
            for size, (low, high) in zip(shape, self.bounds))
        return math.sqrt(self.radius ** 2 + self.feature_radius_scale ** 2 * half_diagonal_squared)

    def forward(self, raw, history_valid=None, force_no_history=False):
        if raw.ndim != 5 or raw.shape[1] < 1:
            raise ValueError("Expected nonempty RT with shape [B,T,X,Y,Z]")
        batch, times = raw.shape[:2]
        if history_valid is None:
            valid_frames = torch.ones((batch, times), dtype=torch.bool, device=raw.device)
        else:
            if tuple(history_valid.shape) != (batch, times):
                raise ValueError("history_valid must have shape [B,T]")
            valid_frames = history_valid.to(device=raw.device, dtype=torch.bool).clone()
        valid_frames[:, -1] = True
        if force_no_history:
            valid_frames[:, :-1] = False
        encoded = torch.stack((torch.log1p(raw.clamp(0, self.cap) / self.scale),
                               (raw != 0).to(raw.dtype)), dim=2)
        features, supports, intensities = [], [], []
        joints, variances, temporal_joints, temporal_variances = [], [], [], []
        hidden = None
        for time in range(times):
            first = self.encoder1(encoded[:, time])
            second = self.encoder2(first)
            third = self.encoder3(second)
            frame_features = (first, second, third)
            support = encoded[:, time, 1:2]
            intensity_sum = encoded[:, time, :1]
            observed_density = support
            frame_support, frame_intensity = [], []
            for feature in frame_features:
                # Receptive-field support, not a test of whether the central
                # raw voxel is zero. A missing voxel may have valid context.
                support = F.max_pool3d(support, 3, stride=2, padding=1)
                support = F.max_pool3d(support, 3, stride=1, padding=1)
                # This local conditional-mean proxy deliberately has a smaller
                # footprint than feature support. Warp its numerator AND count
                # to avoid interpolating missing intensities as actual zeros.
                kernel = tuple(3 if size >= 3 else 1 for size in intensity_sum.shape[-3:])
                padding = tuple(value // 2 for value in kernel)
                intensity_sum = F.avg_pool3d(intensity_sum, kernel, stride=2, padding=padding,
                                             count_include_pad=False)
                observed_density = F.avg_pool3d(observed_density, kernel, stride=2, padding=padding,
                                                count_include_pad=False)
                frame_support.append(support)
                frame_intensity.append((intensity_sum, observed_density))
            features.append(frame_features)
            supports.append(frame_support)
            intensities.append(frame_intensity)
            mean, variance = self._joint_distribution(third)
            joints.append(mean)
            variances.append(variance)
            old_hidden = torch.zeros_like(third) if hidden is None else hidden
            candidate = self.gru(third, old_hidden)
            hidden = torch.where(valid_frames[:, time, None, None, None, None], candidate, old_hidden)
            temporal_mean, temporal_variance = self._joint_distribution(hidden)
            temporal_joints.append(temporal_mean)
            temporal_variances.append(temporal_variance)

        current_joints, current_variance = joints[-1], variances[-1]
        fused, masses, attentions, quality_logits = [], [], {}, []
        aligned_intensities, aligned_supports = [], []
        aligned_coarse_features = []
        for scale, fusion in enumerate((self.history1, self.history2, self.history3)):
            current = features[-1][scale]
            warped_features, warped_supports, qualities, validities, warped_intensity = [], [], [], [], []
            intensity_supports = []
            for time in range(times - 1):
                # Sharing a warp with the intensity diagnostic avoids another
                # expensive articulated resampling pass at every scale.
                source = torch.cat((features[time][scale], *intensities[time][scale]), 1)
                result = articulated_warp(
                    source, joints[time], current_joints, self.bounds,
                    self._feature_radius(current.shape[-3:]),
                    source_support=supports[time][scale],
                    joint_variance=torch.stack((variances[time], current_variance), 1),
                )
                warped_features.append(result["values"][:, :-2].to(current.dtype))
                density = result["values"][:, -1:].clamp(0, 1)
                warped_intensity.append(result["values"][:, -2:-1] / density.clamp_min(1e-6))
                intensity_supports.append(density)
                support = result["support"].float().clamp(0, 1)
                quality = result["quality"].float().clamp(0, 1)
                valid = (result["in_bounds"].bool() & (support > self.support_threshold)
                         & valid_frames[:, time, None, None, None, None])
                warped_supports.append(support)
                qualities.append(quality)
                validities.append(valid)
            value, mass, attention, learned_quality = fusion(
                current, warped_features, warped_supports, qualities, validities)
            fused.append(value)
            masses.append(mass)
            attentions[f"scale{scale + 1}"] = attention
            quality_logits.append(learned_quality)
            if scale == 2:
                aligned_coarse_features = warped_features
            empty = current.new_empty((batch, 0, 1, *current.shape[-3:]))
            aligned_intensities.append(torch.stack(warped_intensity, 1) if warped_intensity else empty)
            diagnostic_supports = [support * valid.to(support.dtype)
                                   for support, valid in zip(intensity_supports, validities)]
            aligned_supports.append(torch.stack(diagnostic_supports, 1) if diagnostic_supports else empty)

        # The original chronological GRU above provides auxiliary per-frame
        # supervision only. Reusing it here would let a rejected bad history
        # bypass the local gate whenever SOME OTHER history was accepted.
        # Instead gate EACH aligned history's recurrent state update before it
        # can contaminate the trusted state or its geometric prediction.
        trusted_history = torch.zeros_like(features[-1][2])
        for time, aligned in enumerate(aligned_coarse_features):
            proposed = self.gru(aligned, trusted_history)
            accepted = attentions["scale3"][:, time].to(proposed.dtype)
            trusted_history = trusted_history + accepted * (proposed - trusted_history)
        trusted_current = self.gru(features[-1][2], trusted_history)
        current_recurrent = self.gru(features[-1][2], torch.zeros_like(features[-1][2]))
        trusted_joints, trusted_variance = self._joint_distribution(trusted_current)
        fused[2] = fused[2] + masses[2].to(fused[2].dtype) * self.temporal_delta(
            trusted_current - current_recurrent)
        current_capsules = capsule_heatmaps(current_joints, features[-1][1].shape[-3:],
                                            self.bounds, self.radius)
        temporal_capsules = capsule_heatmaps(trusted_joints, features[-1][1].shape[-3:],
                                             self.bounds, self.radius)
        capsule_scales = []
        for scale in range(3):
            size = features[-1][scale].shape[-3:]
            base = F.interpolate(current_capsules, size=size, mode="trilinear", align_corners=True)
            temporal = F.interpolate(temporal_capsules, size=size, mode="trilinear", align_corners=True)
            capsule_scales.append(base + masses[scale] * (temporal - base))
        coarse = self.fuse3(fused[2], capsule_scales[2])
        up2 = F.interpolate(coarse, size=fused[1].shape[-3:], mode="trilinear", align_corners=True)
        up2 = self.decode2(torch.cat((up2, self.fuse2(fused[1], capsule_scales[1])), 1))
        up1 = F.interpolate(up2, size=fused[0].shape[-3:], mode="trilinear", align_corners=True)
        up1 = self.decode1(torch.cat((up1, self.fuse1(fused[0], capsule_scales[0])), 1))
        up0 = F.interpolate(up1, size=raw.shape[-3:], mode="trilinear", align_corners=True)
        decoded = self.decode0(torch.cat((up0, encoded[:, -1]), 1))
        logits, intensity, log_variance = self.head(decoded).float().chunk(3, 1)
        global_mass = masses[2].mean((2, 3, 4)).unsqueeze(-1)
        final_joints = current_joints + global_mass * (trusted_joints - current_joints)
        final_variance = (current_variance + global_mass * (trusted_variance - current_variance)
                          + global_mass * (1 - global_mass) * (trusted_joints - current_joints).square())
        history_mass = F.interpolate(masses[0], size=raw.shape[-3:], mode="trilinear", align_corners=True)
        return {
            "reflection_logits": logits,
            "intensity": F.softplus(intensity),
            "log_variance": log_variance.clamp(-6, 4),
            "joints": final_joints,
            "joint_variance": final_variance,
            "body_prior": F.interpolate(capsule_scales[1].amax(1, keepdim=True),
                                         size=raw.shape[-3:], mode="trilinear", align_corners=True),
            "joints_sequence": torch.stack(joints, 1),
            "joint_variance_sequence": torch.stack(variances, 1),
            "temporal_joints_sequence": torch.stack(temporal_joints, 1),
            "temporal_joint_variance_sequence": torch.stack(temporal_variances, 1),
            "current_only_joints": current_joints,
            "current_only_joint_variance": current_variance,
            "trusted_joints": trusted_joints,
            "trusted_joint_variance": trusted_variance,
            "history_attention": attentions,
            "history_mass": history_mass,
            "history_null": 1 - history_mass,
            "history_quality_logits": quality_logits[0],
            "aligned_history_intensity": aligned_intensities[0],
            "history_support": aligned_supports[0],
        }
