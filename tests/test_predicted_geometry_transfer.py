"""CPU tests for Scheme-B RT-only measured-return transfer."""

import unittest

import torch

from rt_completion.predicted_geometry_transfer import (
    budget_transfer, transfer_historical_returns)
from rt_completion.pipeline import prediction_to_records


BOUNDS = [[-1.0, 1.0]] * 3


def body_pose():
    """A compact non-degenerate 22-joint body inside the synthetic grid."""
    return torch.tensor([
        [0, 0, 0], [-.12, 0, -.10], [.12, 0, -.10], [0, 0, .20],
        [-.12, 0, -.35], [.12, 0, -.35], [0, 0, .40],
        [-.12, 0, -.65], [.12, 0, -.65], [0, 0, .55],
        [-.12, .12, -.75], [.12, .12, -.75], [0, 0, .70],
        [-.10, 0, .65], [.10, 0, .65], [0, 0, .88],
        [-.25, 0, .65], [.25, 0, .65], [-.48, 0, .58],
        [.48, 0, .58], [-.70, 0, .50], [.70, 0, .50],
    ], dtype=torch.float32)


def inputs(source_shift=0.0, current_value=0.0, valid=None, variance=1e-4):
    raw = torch.zeros(1, 4, 9, 9, 9)
    source_x = 4 + int(round(source_shift / .25))
    raw[0, 0, source_x, 4, 5] = .8
    raw[0, 1, source_x, 4, 5] = 1.0
    raw[0, 2, source_x, 4, 5] = 1.2
    raw[0, 3, 4, 4, 5] = current_value
    current = body_pose()
    historical = current.clone()
    historical[:, 0] += source_shift
    joints = torch.stack((historical, historical, historical, current), 0).unsqueeze(0)
    prediction = {
        "joints_sequence": joints,
        "joint_variance_sequence": torch.full_like(joints, variance),
    }
    history_valid = (torch.ones(1, 4) if valid is None
                     else torch.tensor([valid], dtype=torch.float32))
    return raw, prediction, history_valid


def settings(**updates):
    result = {
        "min_support_frames": 2,
        "support_threshold": .9,
        "body_threshold": .01,
        "confidence_threshold": .1,
        "consistency_scale_log": .35,
        "uncertainty_scale_m": .15,
        "max_joint_sigma_m": .3,
        "max_intensity_cv": 1.,
        "min_intensity": 1e-6,
        "splat_sigma_m": [.05, .05, .10],
        "splat_radius_voxels": [0, 0, 0],
        "use_part_frames": True,
    }
    result.update(updates)
    return result


class PredictedGeometryTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def run_transfer(self, source_shift=0., current_value=0., valid=None,
                     variance=1e-4, **options):
        raw, prediction, history = inputs(source_shift, current_value, valid, variance)
        return transfer_historical_returns(
            raw, prediction, history, BOUNDS, .18, settings(**options))

    def test_identity_uses_measured_history_amplitude(self):
        result = self.run_transfer()
        index = (0, 0, 4, 4, 5)
        self.assertTrue(result["accepted"][index])
        self.assertEqual(int(result["support_count"][index]), 3)
        self.assertGreater(float(result["confidence"][index]), .1)
        value = float(result["values"][index])
        self.assertGreater(value, .8)
        self.assertLess(value, 1.2)

    def test_predicted_motion_transports_scatter_to_current_pose(self):
        result = self.run_transfer(source_shift=-.25)
        self.assertTrue(result["accepted"][0, 0, 4, 4, 5])
        self.assertFalse(result["accepted"][0, 0, 3, 4, 5])

    def test_current_measurement_is_never_overwritten(self):
        result = self.run_transfer(current_value=.4)
        self.assertFalse(result["accepted"][0, 0, 4, 4, 5])
        self.assertEqual(float(result["values"][0, 0, 4, 4, 5]), 0.)

    def test_padding_and_pose_uncertainty_can_force_abstention(self):
        padded = self.run_transfer(valid=[1, 0, 0, 1])
        self.assertFalse(padded["accepted"].any())
        uncertain = self.run_transfer(variance=.04, max_joint_sigma_m=.05)
        self.assertFalse(uncertain["accepted"].any())

    def test_budget_is_only_a_cap(self):
        result = self.run_transfer()
        capped = budget_transfer(result, 0)
        self.assertFalse(capped["accepted"].any())
        self.assertEqual(float(capped["values"].max()), float(result["values"].max()))

    def test_invalid_support_requirement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "available historical"):
            self.run_transfer(min_support_frames=4)

    def test_pipeline_record_keeps_scheme_b_evidence(self):
        raw, prediction, history = inputs()

        class Model:
            bounds = BOUNDS
            radius = .18

        records = prediction_to_records(Model(), raw, prediction, {
            "mode": "predicted_geometry",
            "max_added_voxels": 8,
            "predicted_geometry": settings(),
        }, history_valid=history)
        record = records[0]
        self.assertEqual(len(record["indices"]), 1)
        self.assertEqual(int(record["support_count"][0]), 3)
        self.assertIn("alignment_quality", record)
        self.assertIn("intensity_disagreement_log", record)
        self.assertGreater(float(record["values"][0]), .8)


if __name__ == "__main__":
    unittest.main()
