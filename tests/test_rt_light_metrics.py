"""Light training reporting keeps scalar means without export/calibration work."""

import io
import unittest
from unittest.mock import Mock, patch

import torch

from rt_completion.evaluation import GroupMetrics
from rt_completion.progress import group_metrics_state, restore_group_metrics_state


class IdentityModel:
    @staticmethod
    def normalize(value):
        return value


def inputs(size=2):
    original = torch.ones(size, 4, 2, 2, 2)
    masked = torch.ones(size, 1, 2, 2, 2, dtype=torch.bool)
    prediction = {"intensity": torch.zeros_like(original[:, -1:])}
    selected = {"sigma": torch.ones_like(prediction["intensity"]),
                "accepted": masked.clone(), "values": prediction["intensity"].clone()}
    return original, masked, prediction, selected


def update(accumulator, loss=1., mae=1., detailed=True, size=2):
    original, masked, prediction, selected = inputs(size)
    metrics = {"loss": torch.tensor(loss), "masked_mae_log": torch.tensor(mae),
               "joint_mpjpe_m": torch.tensor(.125), "masked_abs_error_sum": torch.tensor(100.)}
    with patch("rt_completion.evaluation.select_additions", return_value=selected):
        accumulator.update(metrics, IdentityModel(), original, original, masked, prediction,
                           {"part_index": torch.zeros(size, dtype=torch.long)}, {}, detailed=detailed)


class LightMetricsTests(unittest.TestCase):
    def test_light_does_not_select_or_normalize_or_compute_diagnostics(self):
        original, masked, prediction, _ = inputs()
        model = Mock()
        model.normalize.side_effect = AssertionError("Dense diagnostics must not run")
        collector = GroupMetrics()
        with patch("rt_completion.evaluation.select_additions",
                   side_effect=AssertionError("Top-k selection must not run")):
            collector.update({"loss": torch.tensor(2.)}, model, original, original, masked,
                             prediction, {"part_index": torch.zeros(2)}, {}, detailed=False)
        self.assertEqual(collector.summary(), {"loss": 2., "diagnostics_scope": "disabled",
                                               "diagnostic_samples": 0})
        self.assertEqual(collector.parts, {})
        self.assertEqual(collector.heldout.frames, 0)

    def test_full_and_light_scalar_means_match_and_sum_keys_remain_excluded(self):
        full, light = GroupMetrics(), GroupMetrics()
        for collector, detailed in ((full, True), (light, False)):
            update(collector, loss=1., detailed=detailed, size=2)
            update(collector, loss=4., detailed=detailed, size=1)
        full_report, light_report = full.summary(), light.summary()
        for key in ("loss", "masked_mae_log", "joint_mpjpe_m"):
            self.assertEqual(full_report[key], light_report[key])
        self.assertEqual(light_report["loss"], 2.)
        self.assertNotIn("masked_abs_error_sum", light_report)
        self.assertEqual(full_report["diagnostics_scope"], "full")
        self.assertEqual(full_report["diagnostic_samples"], 3)
        self.assertEqual(full_report["total_masked_voxels"], 24)
        self.assertEqual(full_report["masked_coverage68"], 1.)
        self.assertNotIn("total_masked_voxels", light_report)

    def test_light_round_trip_is_weights_only_safe_and_preserves_scalar_updates(self):
        original = GroupMetrics()
        update(original, loss=2., detailed=False)
        state = group_metrics_state(original)
        self.assertTrue(all(isinstance(value, float) for value in state["totals"].values()))
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = restore_group_metrics_state(torch.load(buffer, weights_only=True))
        self.assertEqual(restored.summary(), original.summary())
        update(original, loss=5., detailed=False, size=1)
        update(restored, loss=5., detailed=False, size=1)
        self.assertEqual(restored.summary(), original.summary())
        self.assertEqual(restored.summary()["loss"], 3.)

    def test_old_full_checkpoint_resumed_light_marks_partial_without_overwriting_scalar_mae(self):
        original = GroupMetrics()
        update(original, mae=.25, detailed=True)
        legacy_state = group_metrics_state(original)
        self.assertEqual(legacy_state["version"], 1)
        resumed = restore_group_metrics_state(legacy_state)
        update(resumed, mae=.75, detailed=False)
        report = resumed.summary()
        self.assertEqual(report["diagnostics_scope"], "sampled")
        self.assertEqual(report["diagnostic_samples"], 2)
        self.assertEqual(report["masked_mae_log"], .5)
        self.assertEqual(report["sampled_diagnostics"]["masked_mae_log"], 1.)
        self.assertEqual(report["sampled_diagnostics"]["frames"], 2)
        self.assertNotIn("masked_coverage68", report)
        self.assertIn("parts", report["sampled_diagnostics"])
        round_trip = restore_group_metrics_state(group_metrics_state(resumed))
        self.assertEqual(report, round_trip.summary())

    def test_periodic_diagnostics_report_only_sampled_coverage(self):
        collector = GroupMetrics()
        update(collector, detailed=False)
        update(collector, detailed=True)
        update(collector, detailed=False)
        report = collector.summary()
        self.assertEqual(report["diagnostics_scope"], "sampled")
        self.assertEqual(report["diagnostic_samples"], 2)
        self.assertNotIn("accepted_heldout_coverage", report)
        self.assertEqual(report["sampled_diagnostics"]["accepted_heldout_coverage"], 1.)


if __name__ == "__main__":
    unittest.main()
