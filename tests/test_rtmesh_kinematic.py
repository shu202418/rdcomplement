import torch

from mmwave_models.Tensor_models.RTmesh.kinematic_refinement import (
    KinematicRefinementHead,
    RTEnergyTrajectory,
    SMPLX_22_PARENTS,
    joints_to_links,
    links_to_joints,
)


def test_skeleton_link_round_trip():
    parents = torch.tensor(SMPLX_22_PARENTS)
    joints = torch.randn(3, 22, 3)
    restored = links_to_joints(joints_to_links(joints, parents), parents)
    torch.testing.assert_close(restored, joints)


def test_rt_energy_trajectory_follows_signal_peak():
    rt = torch.zeros(1, 4, 5, 7, 3)
    rt[0, :, 4, 6, 2] = 10.0
    trajectory = RTEnergyTrajectory()(rt)
    expected = torch.tensor([[3.0, 6.0, 3.0]]).expand(4, -1).unsqueeze(0)
    torch.testing.assert_close(trajectory, expected)


def test_kinematic_head_shapes_and_safe_initialization():
    head = KinematicRefinementHead(feature_dim=32, hidden_dim=16)
    feature = torch.randn(2, 32, requires_grad=True)
    rt = torch.rand(2, 4, 9, 7, 5)
    pose = torch.randn(2, 63)
    root = torch.randn(2, 3)
    translation = torch.randn(2, 3)

    refined_pose, refined_root, refined_trans, auxiliary = head(
        feature, rt, pose, root, translation
    )

    # Zero-initialized residual heads preserve the baseline at initialization.
    torch.testing.assert_close(refined_pose, pose)
    torch.testing.assert_close(refined_root, root)
    assert refined_trans.shape == (2, 3)
    assert auxiliary["refined_joints"].shape == (2, 22, 3)
    assert auxiliary["predicted_velocity"].shape == (2, 3, 3)
    torch.testing.assert_close(
        auxiliary["candidate_weights"].sum(dim=-1), torch.ones(2)
    )

    loss = refined_pose.square().mean() + auxiliary["predicted_velocity"].square().mean()
    loss.backward()
    assert feature.grad is not None
