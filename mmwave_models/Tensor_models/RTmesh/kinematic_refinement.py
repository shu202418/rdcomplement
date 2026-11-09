"""Kinematic refinement blocks for the RT-Mesh baseline.

The module keeps the expensive RT encoder unchanged and adds three lightweight
priors inspired by temporal LiDAR HMR methods: an RT-energy trajectory, an
explicit SMPL-X skeleton, and multiple translation candidates.  It is kept in
its own file so it can be unit-tested without loading the SMPL-X body models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# SMPL-X body-joint tree for the first 22 joints.  The root is its own parent.
SMPLX_22_PARENTS = (0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def joints_to_links(joints: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """Convert absolute joints ``(B, J, 3)`` to parent-relative links."""
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"expected joints shaped (B, J, 3), got {tuple(joints.shape)}")
    if joints.shape[1] != parents.numel():
        raise ValueError("joint count does not match the kinematic tree")
    links = joints - joints[:, parents]
    links[:, 0] = joints[:, 0]
    return links


def links_to_joints(links: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """Reconstruct absolute joints by walking the parent-before-child tree."""
    joints = [links[:, 0]]
    for joint_id in range(1, links.shape[1]):
        joints.append(joints[int(parents[joint_id])] + links[:, joint_id])
    return torch.stack(joints, dim=1)


class RTEnergyTrajectory(nn.Module):
    """Extract a differentiable per-frame signal centroid from a 4D RT clip."""

    def __init__(self, offset=(-3.0, 0.5, 0.0), scale=(6.0, 5.5, 3.0)):
        super().__init__()
        self.register_buffer("offset", torch.tensor(offset, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, rt: torch.Tensor) -> torch.Tensor:
        if rt.ndim != 5:
            raise ValueError(f"expected RT shaped (B, T, X, Y, Z), got {tuple(rt.shape)}")
        energy = rt.float().abs().clamp_max(1.0e6)
        axes = []
        for dim, size in enumerate(rt.shape[2:]):
            reduce_dims = tuple(i for i in range(2, 5) if i != dim + 2)
            marginal = energy.sum(dim=reduce_dims)
            coord = torch.linspace(0.0, 1.0, size, device=rt.device, dtype=energy.dtype)
            axes.append((marginal * coord).sum(-1) / marginal.sum(-1).clamp_min(1.0e-6))
        normalized = torch.stack(axes, dim=-1)
        return normalized * self.scale + self.offset


class KinematicRefinementHead(nn.Module):
    """Skeleton- and velocity-aware refinement for a fused RT-Mesh feature."""

    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 128, num_joints: int = 22):
        super().__init__()
        if num_joints != len(SMPLX_22_PARENTS):
            raise ValueError("the current refinement head expects the 22-joint SMPL-X body tree")
        self.num_joints = num_joints
        self.register_buffer("parents", torch.tensor(SMPLX_22_PARENTS, dtype=torch.long))
        self.trajectory = RTEnergyTrajectory()
        self.trajectory_encoder = nn.GRU(3, hidden_dim, batch_first=True)

        self.coarse_joint_head = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_joints * 3),
        )
        self.link_residual_head = nn.Sequential(
            nn.LayerNorm(feature_dim + hidden_dim),
            nn.Linear(feature_dim + hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_joints * 3),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)
        )
        self.candidate_score = nn.Sequential(
            nn.LayerNorm(feature_dim + hidden_dim),
            nn.Linear(feature_dim + hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4),
        )
        self.pose_residual = nn.Linear(feature_dim + hidden_dim + num_joints * 3, 63)
        self.root_residual = nn.Linear(feature_dim + hidden_dim + num_joints * 3, 3)

        # Start exactly from the published baseline and let training opt into
        # refinement smoothly instead of perturbing predictions at step zero.
        nn.init.zeros_(self.link_residual_head[-1].weight)
        nn.init.zeros_(self.link_residual_head[-1].bias)
        nn.init.zeros_(self.pose_residual.weight)
        nn.init.zeros_(self.pose_residual.bias)
        nn.init.zeros_(self.root_residual.weight)
        nn.init.zeros_(self.root_residual.bias)
        self.translation_gate = nn.Parameter(torch.tensor(-8.0))

    def forward(
        self,
        feature: torch.Tensor,
        rt: torch.Tensor,
        base_pose: torch.Tensor,
        base_root_orient: torch.Tensor,
        base_translation: torch.Tensor,
    ):
        trajectory = self.trajectory(rt).to(feature.dtype)
        temporal_features, last_hidden = self.trajectory_encoder(trajectory)
        temporal_context = last_hidden[-1]
        context = torch.cat((feature, temporal_context), dim=-1)

        coarse_joints = self.coarse_joint_head(feature).view(-1, self.num_joints, 3)
        coarse_links = joints_to_links(coarse_joints, self.parents)
        link_residual = self.link_residual_head(context).view(-1, self.num_joints, 3)
        link_residual[:, 0] = 0.0
        refined_links = coarse_links + link_residual
        refined_joints = links_to_joints(refined_links, self.parents)

        predicted_velocity = self.velocity_head(temporal_features[:, 1:])
        current_signal_center = trajectory[:, -1]
        extrapolated_center = trajectory[:, -2] + predicted_velocity[:, -1]
        skeleton_root = refined_joints[:, 0]
        candidates = torch.stack(
            (base_translation, current_signal_center, extrapolated_center, skeleton_root), dim=1
        )
        candidate_weights = F.softmax(self.candidate_score(context), dim=-1)
        candidate_translation = (candidates * candidate_weights.unsqueeze(-1)).sum(dim=1)
        gate = torch.sigmoid(self.translation_gate)
        refined_translation = base_translation + gate * (candidate_translation - base_translation)

        kinematic_feature = torch.cat((context, refined_joints.flatten(1)), dim=-1)
        refined_pose = base_pose + self.pose_residual(kinematic_feature)
        refined_root_orient = base_root_orient + self.root_residual(kinematic_feature)

        auxiliary = {
            "coarse_joints": coarse_joints,
            "refined_joints": refined_joints,
            "refined_links": refined_links,
            "predicted_velocity": predicted_velocity,
            "rt_trajectory": trajectory,
            "translation_candidates": candidates,
            "candidate_weights": candidate_weights,
        }
        return refined_pose, refined_root_orient, refined_translation, auxiliary
