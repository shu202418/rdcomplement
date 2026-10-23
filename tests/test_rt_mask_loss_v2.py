"""Small CPU checks for motion-following masks and V2 supervision boundaries."""

import unittest
from unittest.mock import patch

import torch

from rt_completion.losses import completion_loss, corrupt_rt


class DummyModel:
    bounds = [[-1., 1.]] * 3
    radius = .16

    @staticmethod
    def normalize(value):
        return value.clamp_min(0).log1p()


class MaskLossV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def moving_fixture(self):
        raw = torch.zeros(1, 4, 9, 9, 9)
        joints = torch.zeros(1, 4, 22, 3)
        centres = [1, 2, 3, 7]
        for time, x in enumerate(centres):
            raw[0, time, x, 4, 4] = 1.
            joints[0, time, :, :] = torch.tensor([float(x), 4., 4.])
            if time < 3:
                raw[0, time, 7, 4, 4] = .5  # fixed-world decoy
        config = {"mode": "mixed_parts", "part_radius": .2, "drop_probability": .5,
                  "continuous_min_frames": 4, "continuous_max_frames": 4,
                  "block_shape": [3, 3, 3]}
        return raw, joints, config, [[0., 8.]] * 3

    def test_continuous_part_mask_follows_motion_not_fixed_world_box(self):
        raw, joints, config, bounds = self.moving_fixture()
        result, masked, info = corrupt_rt(raw, config, torch.Generator().manual_seed(3),
                                          joints=joints, bounds=bounds, group="continuous_part", return_info=True)
        for time, x in enumerate([1, 2, 3, 7]):
            self.assertEqual(float(result[0, time, x, 4, 4]), 0.)
        for time in range(3):
            self.assertEqual(float(result[0, time, 7, 4, 4]), .5)
        self.assertEqual(int(masked.sum()), 1)
        self.assertEqual(info["kind"], ["continuous_part"])
        self.assertEqual(int(info["duration"][0]), 4)
        self.assertTrue(info["eligible"][0])
        self.assertGreaterEqual(int(info["part_index"][0]), 0)

    def test_single_part_only_current_and_no_support_is_not_random_fallback(self):
        raw, joints, config, bounds = self.moving_fixture()
        result, _, info = corrupt_rt(raw, config, torch.Generator().manual_seed(3),
                                     joints=joints, bounds=bounds, group="single_part", return_info=True)
        torch.testing.assert_close(result[:, :-1], raw[:, :-1])
        self.assertEqual(int(info["duration"][0]), 1)
        result, masked, info = corrupt_rt(raw, config, torch.Generator().manual_seed(3),
                                          joints=joints + 100, bounds=bounds,
                                          group="single_part", return_info=True)
        torch.testing.assert_close(result, raw)
        self.assertEqual(int(masked.sum()), 0)
        self.assertFalse(info["eligible"][0])
        self.assertEqual(int(info["part_index"][0]), -1)

    def test_seeded_groups_deterministic_and_only_measured_targets(self):
        raw, joints, config, bounds = self.moving_fixture()
        for group in ("random", "single_part", "continuous_part", "spatial_block"):
            first = corrupt_rt(raw, config, torch.Generator().manual_seed(43),
                               joints=joints, bounds=bounds, group=group, return_info=True)
            second = corrupt_rt(raw, config, torch.Generator().manual_seed(43),
                                joints=joints, bounds=bounds, group=group, return_info=True)
            torch.testing.assert_close(first[0], second[0])
            torch.testing.assert_close(first[1], second[1])
            self.assertFalse((first[1] & (raw[:, -1:] == 0)).any())
            self.assertEqual(first[2]["kind"], [group])
            torch.testing.assert_close(first[2]["part_index"], second[2]["part_index"])

    def loss_fixture(self):
        raw = torch.zeros(1, 4, 3, 3, 3)
        raw[:, -1, 1, 1, 1] = 1.
        masked = raw[:, -1:] > 0
        joints = torch.zeros(1, 4, 22, 3)
        history = torch.tensor([[0., 1., 1., 0.]])
        output = {
            "intensity": torch.full_like(raw[:, -1:], .3, requires_grad=True),
            "reflection_logits": torch.zeros_like(raw[:, -1:], requires_grad=True),
            "log_variance": torch.zeros_like(raw[:, -1:], requires_grad=True),
            "joints": torch.full((1, 22, 3), .05, requires_grad=True),
            "joints_sequence": torch.full_like(joints, .05, requires_grad=True),
            "joint_variance_sequence": torch.full_like(joints, .02, requires_grad=True),
        }
        config = {"temporal_alignment": "articulated", "temporal_weight": .1,
                  "observed_weight": .05, "reflection_weight": .1,
                  "joint_weight": 1., "joint_uncertainty_weight": .01,
                  "sparsity_weight": .03}
        pseudo = {"intensity": torch.ones_like(raw[:, -1:], requires_grad=True),
                  "valid_mask": torch.ones_like(masked),
                  "confidence": torch.full_like(raw[:, -1:], .5, requires_grad=True),
                  "support_count": torch.full_like(raw[:, -1:], 2.)}
        return raw, masked, joints, history, output, config, pseudo

    def test_real_observation_precedence_and_pseudo_targets_stop_gradient(self):
        raw, masked, joints, history, output, config, pseudo = self.loss_fixture()
        with patch("rt_completion.geometry.temporal_target_v2", return_value=pseudo, create=True):
            loss, metrics = completion_loss(output, raw, masked, joints, history, DummyModel(), config)
        self.assertEqual(int(metrics["temporal_voxels"]), 26)
        self.assertEqual(int(metrics["masked_voxels"]), 1)
        altered = dict(pseudo)
        altered["intensity"] = pseudo["intensity"].detach().clone()
        altered["intensity"][raw[:, -1:] > 0] = 100.
        with patch("rt_completion.geometry.temporal_target_v2", return_value=altered, create=True):
            _, modified = completion_loss(output, raw, masked, joints, history, DummyModel(), config)
        torch.testing.assert_close(metrics["temporal"], modified["temporal"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(pseudo["intensity"].grad)
        self.assertIsNone(pseudo["confidence"].grad)
        self.assertTrue(torch.isfinite(output["joint_variance_sequence"].grad).all())
        self.assertTrue(torch.isfinite(output["joints_sequence"].grad).all())
        self.assertEqual(float(output["joints_sequence"].grad[:, 0].abs().sum()), 0.)
        self.assertGreater(float(output["joints_sequence"].grad[:, -1].abs().sum()), 0.)
        self.assertAlmostEqual(float(metrics["masked_abs_error_sum"]),
                               abs(.3 - float(torch.tensor(1.).log1p())), places=6)

    def test_confidence_weights_do_not_cancel_in_normalization(self):
        raw, masked, joints, history, output, config, pseudo = self.loss_fixture()
        values = []
        for confidence in (1., .1):
            pseudo["confidence"] = torch.full_like(raw[:, -1:], confidence)
            with patch("rt_completion.geometry.temporal_target_v2", return_value=pseudo, create=True):
                _, metrics = completion_loss(output, raw, masked, joints, history, DummyModel(), config)
            values.append(float(metrics["temporal"]))
        self.assertAlmostEqual(values[1], .1 * values[0], places=6)

    def test_missing_labels_and_empty_evidence_are_finite(self):
        raw, masked, joints, history, output, config, pseudo = self.loss_fixture()
        raw.zero_()
        masked.fill_(False)
        joints.fill_(float("nan"))
        pseudo["valid_mask"].fill_(False)
        with patch("rt_completion.geometry.temporal_target_v2", return_value=pseudo, create=True):
            loss, metrics = completion_loss(output, raw, masked, joints, history, DummyModel(), config)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
        self.assertEqual(int(metrics["masked_voxels"]), 0)
        self.assertEqual(int(metrics["temporal_voxels"]), 0)

    def test_quality_proxy_loss_uses_heldout_truth_but_detaches_quality_label(self):
        raw, masked, joints, history, output, config, pseudo = self.loss_fixture()
        output.update(history_quality_logits=torch.zeros(1, 3, 1, 1, 1, 1, requires_grad=True),
                      aligned_history_intensity=torch.zeros(1, 3, 1, 1, 1, 1, requires_grad=True),
                      history_support=torch.ones(1, 3, 1, 1, 1, 1))
        with patch("rt_completion.geometry.temporal_target_v2", return_value=pseudo, create=True):
            loss, metrics = completion_loss(output, raw, masked, joints, history, DummyModel(), config)
        loss.backward()
        self.assertEqual(int(metrics["history_quality_voxels"]), 3)
        self.assertGreater(float(output["history_quality_logits"].grad.abs().sum()), 0.)
        self.assertIsNone(output["aligned_history_intensity"].grad)


if __name__ == "__main__":
    unittest.main()
