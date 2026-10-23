"""Runnable dense-encoder baseline; no claim that Conv3d is a Sparse CNN."""

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import capsule_heatmaps, metric_grid


def block(inputs, outputs, stride=1):
    groups = next(g for g in (8, 4, 2, 1) if outputs % g == 0)
    return nn.Sequential(
        nn.Conv3d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, outputs), nn.SiLU(),
        nn.Conv3d(outputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(groups, outputs), nn.SiLU(),
    )


class ConvGRU3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv3d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv3d(channels * 2, channels, 3, padding=1)

    def forward(self, value, hidden):
        if hidden is None:
            hidden = torch.zeros_like(value)
        reset, update = self.gates(torch.cat((value, hidden), 1)).sigmoid().chunk(2, 1)
        candidate = self.candidate(torch.cat((value, reset * hidden), 1)).tanh()
        return (1 - update) * hidden + update * candidate


class GatedGeometry(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.project = nn.Conv3d(15, channels, 1)
        self.gate = nn.Conv3d(2 * channels, channels, 1)

    def forward(self, features, capsules):
        geometry = self.project(capsules.to(features.dtype))
        weight = self.gate(torch.cat((features, geometry), 1)).sigmoid()
        return features + weight * geometry


class RTCompletionNet(nn.Module):
    def __init__(self, data, model):
        super().__init__()
        self.architecture = model.get("architecture", "baseline")
        if self.architecture not in ("baseline", "reliability_v2"):
            raise ValueError(f"Unknown RT completion architecture: {self.architecture}")
        self.bounds = data["bounds"]
        self.intensity_scale = float(data["intensity_scale"])
        self.intensity_cap = float(data["intensity_cap"])
        self.radius = float(model["capsule_radius_m"])
        if self.architecture == "reliability_v2":
            # A separate namespace deliberately prevents accidentally loading a
            # baseline checkpoint into the materially different V2 network.
            from .model_v2 import ReliabilityRTCompletionNet
            self.v2 = ReliabilityRTCompletionNet(data, model)
            return
        if model.get("encoder", "dense") != "dense":
            raise ValueError("This implementation supports encoder=dense; spconv is not installed")
        channels = int(model["base_channels"])
        self.bounds = data["bounds"]
        self.intensity_scale = float(data["intensity_scale"])
        self.intensity_cap = float(data["intensity_cap"])
        self.radius = float(model["capsule_radius_m"])
        # Shared weights over time, low-resolution features to limit memory.
        self.encoder1 = block(2, channels, stride=2)
        self.encoder2 = block(channels, channels * 2, stride=2)
        self.encoder3 = block(channels * 2, channels * 4, stride=2)
        self.gru = ConvGRU3D(channels * 4)
        self.joint_logits = nn.Conv3d(channels * 4, 22, 1)
        self.fuse3 = GatedGeometry(channels * 4)
        self.fuse2 = GatedGeometry(channels * 2)
        self.fuse1 = GatedGeometry(channels)
        self.decode2 = block(channels * 6, channels * 2)
        self.decode1 = block(channels * 3, channels)
        self.decode0 = block(channels + 2, channels)
        self.head = nn.Conv3d(channels, 3, 1)

    def normalize(self, value):
        return torch.log1p(value.clamp(0, self.intensity_cap) / self.intensity_scale)

    def denormalize(self, value):
        maximum = value.new_tensor(self.intensity_cap / self.intensity_scale).log1p()
        return torch.expm1(value.clamp(0, maximum)) * self.intensity_scale

    def forward(self, raw, history_valid=None, force_no_history=False):
        if self.architecture == "reliability_v2":
            return self.v2(raw, history_valid=history_valid, force_no_history=force_no_history)
        if force_no_history:
            raise ValueError("force_no_history is supported only by architecture=reliability_v2")
        if raw.ndim != 5:
            raise ValueError("Expected RT with shape [B,T,X,Y,Z]")
        encoded = torch.stack((self.normalize(raw), (raw != 0).to(raw.dtype)), dim=2)
        hidden = None
        # Only the current-frame skips enter the decoder; all four frames enter
        # the recurrent state. No GT joints, masks or SMPL-X arguments exist.
        for time in range(raw.shape[1]):
            skip1 = self.encoder1(encoded[:, time])
            skip2 = self.encoder2(skip1)
            skip3 = self.encoder3(skip2)
            hidden = self.gru(skip3, hidden)
        heat = self.joint_logits(hidden).float().flatten(2).softmax(-1)
        grid = metric_grid(hidden.shape[-3:], self.bounds, raw.device).reshape(-1, 3)
        joints = heat @ grid
        capsules2 = capsule_heatmaps(joints, skip2.shape[-3:], self.bounds, self.radius)
        capsules3 = F.interpolate(capsules2, size=hidden.shape[-3:], mode="trilinear", align_corners=True)
        capsules1 = F.interpolate(capsules2, size=skip1.shape[-3:], mode="trilinear", align_corners=True)
        hidden = self.fuse3(hidden, capsules3)
        up2 = F.interpolate(hidden, size=skip2.shape[-3:], mode="trilinear", align_corners=True)
        up2 = self.decode2(torch.cat((up2, self.fuse2(skip2, capsules2)), dim=1))
        up1 = F.interpolate(up2, size=skip1.shape[-3:], mode="trilinear", align_corners=True)
        up1 = self.decode1(torch.cat((up1, self.fuse1(skip1, capsules1)), dim=1))
        up0 = F.interpolate(up1, size=raw.shape[-3:], mode="trilinear", align_corners=True)
        decoded = self.decode0(torch.cat((up0, encoded[:, -1]), dim=1))
        logits, intensity, log_variance = self.head(decoded).float().chunk(3, dim=1)
        return {
            "reflection_logits": logits,
            "intensity": F.softplus(intensity),  # log1p intensity space
            "log_variance": log_variance.clamp(-6, 4),
            "joints": joints,  # radar-space meters, 22 SMPL-X-regressed joints
            "body_prior": F.interpolate(capsules2.amax(1, keepdim=True), size=raw.shape[-3:],
                                         mode="trilinear", align_corners=True),
        }
