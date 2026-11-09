"""CPU regressions for S1 V3 rigid geometry, masks and surface supervision."""

import unittest

import torch

from rt_completion.data import deterministic_subset
from rt_completion.geometry import articulated_warp
from rt_completion.losses import completion_loss, corrupt_rt
from rt_completion.model import RTCompletionNet


class V3SurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model_config(self):
        data = {"bounds": [[-1., 1.]] * 3, "intensity_scale": 1., "intensity_cap": 10.}
        model = {
            "architecture": "reliability_v2", "base_channels": 2,
            "capsule_radius_m": .3, "part_rigid_frames": True,
            "surface_prior": {"enabled": True, "num_points": 8, "heatmap_radius_m": .2},
        }
        return data, model

    def test_fraction_is_exact_reproducible_and_not_prefix(self):
        values = list(range(100))
        first = deterministic_subset(values, .1, 42)
        self.assertEqual(len(first), 10)
        self.assertEqual(first, deterministic_subset(values, .1, 42))
        self.assertNotEqual(first, values[:10])

    def test_part_frame_resolves_axial_twist(self):
        source = torch.full((1, 22, 3), 10.)
        target = source.clone()
        source[:, 16] = torch.tensor([0., 0., 0.])
        source[:, 18] = torch.tensor([1., 0., 0.])
        source[:, 9] = torch.tensor([0., 1., 0.])
        target[:, 16] = source[:, 16]
        target[:, 18] = source[:, 18]
        target[:, 9] = torch.tensor([0., 0., 1.])
        volume = torch.zeros(1, 1, 9, 9, 9)
        volume[0, 0, 6, 5, 4] = 2.
        result = articulated_warp(volume, source, target, [[-1., 1.]] * 3,
                                  radius=.3, use_part_frames=True)
        self.assertAlmostEqual(float(result["values"][0, 0, 6, 4, 5]), 2., places=4)
        self.assertAlmostEqual(float(result["values"][0, 0, 6, 5, 4]), 0., places=4)

    def test_surface_forward_loss_and_backward(self):
        data, settings = self.model_config()
        model = RTCompletionNet(data, settings)
        raw = torch.zeros(1, 4, 9, 11, 9)
        raw[:, :, 4, 5, 4] = 1.
        joints = torch.zeros(1, 4, 22, 3)
        surfaces = torch.rand(1, 4, 8, 3) * 1.5 - .75
        corrupted = raw.clone()
        corrupted[:, -1, 4, 5, 4] = 0
        masked = raw[:, -1:] > 0
        output = model(corrupted, torch.tensor([[1., 1., 1., 0.]]))
        self.assertEqual(tuple(output["surface_points"].shape), (1, 8, 3))
        self.assertEqual(tuple(output["surface_sequence"].shape), (1, 4, 8, 3))
        self.assertEqual(tuple(output["surface_prior"].shape), (1, 1, 9, 11, 9))
        config = {"temporal_alignment": "articulated", "temporal_weight": 0.,
                  "observed_weight": .05, "reflection_weight": 1., "joint_weight": 1.,
                  "joint_uncertainty_weight": .01, "surface_weight": 1.,
                  "sparsity_weight": .03}
        loss, metrics = completion_loss(
            output, raw, masked, joints, torch.tensor([[1., 1., 1., 0.]]), model,
            config, surface_points=surfaces)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(metrics["surface_mpe_m"]), 0)
        self.assertGreater(float(model.v2.surface_logits.weight.grad.abs().sum()), 0)

    def test_natural_missing_masks_only_remove_observations(self):
        raw = torch.zeros(1, 4, 9, 9, 9)
        raw[:, :, 4, 3:7, 4] = torch.tensor([.01, .05, .2, 1.])
        joints = torch.zeros(1, 4, 22, 3)
        for group in ("weak_return", "range_shadow"):
            first = corrupt_rt(
                raw, {"mode": "mixed_parts", "weak_drop_probability": 1.,
                      "shadow_azimuth_width_rad": .5, "shadow_height_m": 2.},
                torch.Generator().manual_seed(7), joints=joints,
                bounds=[[-1., 1.]] * 3, group=group, return_info=True)
            second = corrupt_rt(
                raw, {"mode": "mixed_parts", "weak_drop_probability": 1.,
                      "shadow_azimuth_width_rad": .5, "shadow_height_m": 2.},
                torch.Generator().manual_seed(7), joints=joints,
                bounds=[[-1., 1.]] * 3, group=group, return_info=True)
            torch.testing.assert_close(first[0], second[0])
            self.assertFalse((first[1] & (raw[:, -1:] == 0)).any())
            self.assertTrue(first[1].any())


if __name__ == "__main__":
    unittest.main()
