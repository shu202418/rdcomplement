"""Articulated feature warp and detached temporal-teacher CPU regressions."""

import unittest

import torch

from rt_completion.geometry import articulated_warp, temporal_target_v2


BOUNDS = [[-1., 1.]] * 3


def straight_body():
    # All nondegenerate segments lie on one axis, making translation and
    # observation-support assertions independent of part-overlap ties.
    joints = torch.zeros(1, 22, 3)
    joints[0, :, 0] = torch.linspace(-.7, .7, 22)
    return joints


class ArticulatedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_translation_all_axes_and_soft_support(self):
        source = straight_body()
        volume = torch.zeros(1, 1, 9, 9, 9)
        volume[0, 0, 4, 4, 4] = 2
        for axis in range(3):
            target = source.clone()
            target[:, :, axis] += .25
            result = articulated_warp(volume, source, target, BOUNDS, radius=.3)
            expected = [4, 4, 4]
            expected[axis] += 1
            self.assertAlmostEqual(float(result["values"][(0, 0, *expected)]), 2, places=5)
            self.assertAlmostEqual(float(result["support"][(0, 0, *expected)]), 1, places=5)
            self.assertTrue(result["in_bounds"][(0, 0, *expected)])
        target = source.clone()
        target[:, :, 1] += .125
        result = articulated_warp(volume, source, target, BOUNDS, radius=.3)
        self.assertAlmostEqual(float(result["support"][0, 0, 4, 4, 4]), .5, places=5)
        self.assertAlmostEqual(float(result["support"][0, 0, 4, 5, 4]), .5, places=5)

    def test_rotation_of_limb_not_root_translation(self):
        # The skeleton rotates globally around the origin while the root stays
        # fixed: the observed point must follow rotation, not remain stationary.
        source = straight_body()
        source[:, 0] = 0
        target = source.clone()
        target[..., 0], target[..., 1] = -source[..., 1], source[..., 0]
        volume = torch.zeros(1, 1, 9, 9, 9)
        volume[0, 0, 5, 4, 4] = 3
        result = articulated_warp(volume, source, target, BOUNDS, radius=.2)
        self.assertAlmostEqual(float(result["values"][0, 0, 4, 5, 4]), 3, places=5)
        self.assertAlmostEqual(float(result["values"][0, 0, 5, 4, 4]), 0, places=5)

    def test_feature_and_pose_gradients_are_finite(self):
        source = straight_body().requires_grad_()
        target = (straight_body() + torch.tensor([0., .06, .03])).requires_grad_()
        volume = torch.rand(1, 2, 9, 9, 9, requires_grad=True)
        result = articulated_warp(volume, source, target, BOUNDS, radius=.3,
                                  source_support=torch.ones(1, 1, 9, 9, 9))
        (result["values"].square().sum() + result["quality"].sum()).backward()
        for value in (volume, source, target):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_degenerate_antiparallel_and_extreme_bones_finite(self):
        source = straight_body()
        volume = torch.ones(1, 1, 9, 9, 9)
        for target in (torch.zeros_like(source), -source, source * 20):
            result = articulated_warp(volume, source, target, BOUNDS, radius=.3)
            for value in result.values():
                self.assertTrue(torch.isfinite(value).all())
            if torch.equal(target, torch.zeros_like(source)) or torch.equal(target, source * 20):
                self.assertFalse(result["support"].any())
        variance = torch.full((1, 22, 3), .01)
        uncertain = articulated_warp(volume, source, source, BOUNDS, radius=.3, joint_variance=variance)
        exact = articulated_warp(volume, source, source, BOUNDS, radius=.3)
        self.assertLess(float(uncertain["quality"].sum()), float(exact["quality"].sum()))


class TemporalTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        raw = torch.zeros(1, 4, 9, 9, 9)
        raw[:, :3, 4, 4, 4] = 2
        joints = straight_body()[:, None].expand(-1, 4, -1, -1).clone()
        history = torch.tensor([[1., 1., 1., 0.]])
        return raw, joints, history

    def test_support_count_padding_and_observation_precedence(self):
        raw, joints, history = self.fixture()
        result = temporal_target_v2(raw, joints, history, BOUNDS, .3, {})
        self.assertTrue(result["valid_mask"][0, 0, 4, 4, 4])
        self.assertEqual(int(result["support_count"][0, 0, 4, 4, 4]), 3)
        self.assertAlmostEqual(float(result["intensity"][0, 0, 4, 4, 4]), 2, places=5)
        history[:, :2] = 0
        padded = temporal_target_v2(raw, joints, history, BOUNDS, .3, {})
        self.assertFalse(padded["valid_mask"].any())
        self.assertEqual(int(padded["support_count"][0, 0, 4, 4, 4]), 1)
        history[:, :2] = 1
        raw[:, -1, 4, 4, 4] = 1e-9  # Every original nonzero is preserved.
        measured = temporal_target_v2(raw, joints, history, BOUNDS, .3, {})
        self.assertFalse(measured["valid_mask"].any())
        self.assertFalse(measured["confidence"].any())

    def test_conflict_reduces_confidence_and_teacher_detaches(self):
        raw, joints, history = self.fixture()
        consistent = temporal_target_v2(raw, joints, history, BOUNDS, .3, {})
        raw[:, 0, 4, 4, 4] = 20
        raw.requires_grad_()
        joints.requires_grad_()
        conflict = temporal_target_v2(raw, joints, history, BOUNDS, .3, {})
        self.assertTrue(conflict["valid_mask"][0, 0, 4, 4, 4])
        self.assertLess(float(conflict["confidence"][0, 0, 4, 4, 4]),
                        float(consistent["confidence"][0, 0, 4, 4, 4]))
        for value in conflict.values():
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value).all())

    def test_tiny_interpolation_support_is_not_positive_evidence(self):
        raw, joints, history = self.fixture()
        # Shift current skeleton almost one whole voxel. The old voxel retains
        # only 1% interpolation weight, below the explicit 25% support gate.
        joints[:, -1, :, 1] += .2475
        result = temporal_target_v2(raw, joints, history, BOUNDS, .3, {})
        self.assertEqual(int(result["support_count"][0, 0, 4, 4, 4]), 0)
        self.assertFalse(result["valid_mask"][0, 0, 4, 4, 4])
        self.assertTrue(result["valid_mask"][0, 0, 4, 5, 4])
        self.assertAlmostEqual(float(result["intensity"][0, 0, 4, 5, 4]), 2, places=5)

    def test_empty_and_single_frame_are_finite(self):
        raw, joints, history = self.fixture()
        for count in (1, 4):
            result = temporal_target_v2(torch.zeros_like(raw[:, :count]), joints[:, :count],
                                        history[:, :count], BOUNDS, .3, {})
            for value in result.values():
                self.assertTrue(torch.isfinite(value).all())
            self.assertFalse(result["valid_mask"].any())


if __name__ == "__main__":
    unittest.main()
