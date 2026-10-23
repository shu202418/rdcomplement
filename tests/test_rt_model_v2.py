"""Small CPU checks of V2 architecture and its no-history safety contract."""

import unittest
from unittest.mock import patch

import torch

from rt_completion.model import RTCompletionNet
from rt_completion.model_v2 import LocalHistoryFusion
from rt_completion.geometry import articulated_warp


class ReliabilityModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, architecture="reliability_v2"):
        return ({"bounds": [[-1., 1.]] * 3, "intensity_scale": 1., "intensity_cap": 10.},
                {"architecture": architecture, "base_channels": 2,
                 "capsule_radius_m": .3, "joint_variance_floor": 1e-4})

    def model(self):
        return RTCompletionNet(*self.config())

    def test_legacy_state_schema_and_strict_reload(self):
        data, config = self.config("baseline")
        config.pop("architecture")
        torch.manual_seed(42)
        original = RTCompletionNet(data, config)
        explicit = RTCompletionNet(data, {**config, "architecture": "baseline"})
        explicit.load_state_dict(original.state_dict(), strict=True)
        self.assertIn("encoder1.0.weight", original.state_dict())
        self.assertNotIn("v2.encoder1.0.weight", original.state_dict())
        raw = torch.rand(1, 4, 9, 11, 9)
        for name, value in original(raw).items():
            self.assertTrue(torch.equal(value, explicit(raw)[name]), name)
        with self.assertRaisesRegex(ValueError, "only"):
            original(raw, force_no_history=True)

    def test_odd_shape_finite_backward_and_joint_moments(self):
        model = self.model()
        raw = torch.rand(2, 4, 9, 11, 9)
        result = model(raw, torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]]))
        self.assertEqual(tuple(result["intensity"].shape), (2, 1, 9, 11, 9))
        for name in ("joints_sequence", "joint_variance_sequence", "temporal_joints_sequence",
                     "temporal_joint_variance_sequence"):
            self.assertEqual(tuple(result[name].shape), (2, 4, 22, 3))
            self.assertTrue(torch.isfinite(result[name]).all(), name)
        self.assertTrue((result["joint_variance_sequence"] >= 1e-4).all())
        loss = (result["intensity"].mean() + result["reflection_logits"].square().mean()
                + result["log_variance"].square().mean() + result["joints_sequence"].square().mean()
                + result["temporal_joints_sequence"].square().mean()
                + result["joint_variance_sequence"].mean())
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_attention_sum_and_all_invalid_exact_null(self):
        result = self.model()(torch.rand(1, 4, 9, 11, 9), torch.zeros(1, 4))
        for attention in result["history_attention"].values():
            torch.testing.assert_close(attention.sum(1), torch.ones_like(attention[:, 0]))
            self.assertTrue(torch.equal(attention[:, :-1], torch.zeros_like(attention[:, :-1])))
            self.assertTrue(torch.equal(attention[:, -1], torch.ones_like(attention[:, -1])))
        self.assertTrue(torch.equal(result["history_mass"], torch.zeros_like(result["history_mass"])))
        self.assertTrue(torch.equal(result["history_null"], torch.ones_like(result["history_null"])))

    def test_strict_current_only_fallback_has_no_history_bypass(self):
        model = self.model().eval()
        raw = torch.rand(1, 4, 9, 11, 9)
        changed = raw.clone()
        changed[:, :-1] = torch.rand_like(changed[:, :-1]) * 5
        keys = ("reflection_logits", "intensity", "log_variance", "joints", "joint_variance",
                "body_prior", "history_mass", "history_null")
        with torch.no_grad():
            forced = model(raw, torch.ones(1, 4), force_no_history=True)
            other = model(changed, torch.ones(1, 4), force_no_history=True)
            invalid = model(raw, torch.zeros(1, 4))
            other_invalid = model(changed, torch.zeros(1, 4))
        for key in keys:
            self.assertTrue(torch.equal(forced[key], other[key]), key)
            self.assertTrue(torch.equal(invalid[key], other_invalid[key]), key)
            self.assertTrue(torch.equal(forced[key], invalid[key]), key)
        differentiable = raw.clone().requires_grad_()
        output = model(differentiable, force_no_history=True)
        output["intensity"].sum().backward()
        self.assertTrue(torch.equal(differentiable.grad[:, :-1],
                                    torch.zeros_like(differentiable.grad[:, :-1])))
        self.assertGreater(float(differentiable.grad[:, -1].abs().sum()), 0)

    def test_null_mass_is_not_renormalized_away(self):
        fusion = LocalHistoryFusion(1)
        with torch.no_grad():
            for parameter in fusion.parameters():
                parameter.zero_()
            fusion.value.weight.fill_(1)
        current = torch.zeros(1, 1, 2, 2, 2)
        history = torch.ones_like(current)
        output, mass, attention, _ = fusion(current, [history], [history], [history],
                                            [history.bool()])
        torch.testing.assert_close(attention, torch.full_like(attention, .5))
        torch.testing.assert_close(mass, torch.full_like(mass, .5))
        torch.testing.assert_close(output, torch.full_like(output, .5))

    def test_invalid_history_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "history_valid"):
            self.model()(torch.zeros(1, 4, 9, 11, 9), torch.zeros(1, 3))

    def test_coarse_cells_do_not_starve_articulated_history(self):
        model = self.model()
        self.assertGreater(model.v2._feature_radius((2, 2, 2)), model.v2.radius)
        joints = torch.zeros(1, 22, 3)
        joints[0, :, 0] = torch.linspace(-.2, .2, 22)
        variance = torch.full_like(joints, 1e-4)
        with patch.object(model.v2, "_joint_distribution", return_value=(joints, variance)):
            result = model(torch.ones(1, 4, 9, 11, 9), torch.ones(1, 4))
        self.assertGreater(float(result["history_attention"]["scale3"][:, :-1].sum()), 0)

    def test_spatial_moments_stay_float32_under_autocast(self):
        model = self.model()
        feature = torch.rand(1, 8, 2, 2, 2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            mean, variance = model.v2._joint_distribution(feature)
        self.assertEqual(mean.dtype, torch.float32)
        self.assertEqual(variance.dtype, torch.float32)
        self.assertTrue(torch.isfinite(variance).all())
        self.assertTrue((variance >= 1e-4).all())

    def test_one_rejected_history_cannot_bypass_another_accepted_history(self):
        model = self.model().eval()
        raw = torch.ones(1, 4, 9, 11, 9)
        changed = raw.clone()
        changed[:, 0] = torch.rand_like(changed[:, 0]) * 5
        joints = torch.zeros(1, 22, 3)
        joints[0, :, 0] = torch.linspace(-.2, .2, 22)
        variance = torch.full_like(joints, 1e-4)
        calls = 0

        def reject_first_history(*args, **kwargs):
            nonlocal calls
            result = articulated_warp(*args, **kwargs)
            # Every scale processes three histories in chronological order.
            if calls % 3 == 0:
                result["support"] = torch.zeros_like(result["support"])
                result["in_bounds"] = torch.zeros_like(result["in_bounds"])
            calls += 1
            return result

        with torch.no_grad(), patch.object(model.v2, "_joint_distribution",
                                            return_value=(joints, variance)), \
                patch("rt_completion.model_v2.articulated_warp", side_effect=reject_first_history):
            original = model(raw, torch.ones(1, 4))
            other = model(changed, torch.ones(1, 4))
        attention = original["history_attention"]["scale3"]
        self.assertTrue(torch.equal(attention[:, 0], torch.zeros_like(attention[:, 0])))
        self.assertGreater(float(attention[:, 1:-1].sum()), 0)
        for key in ("intensity", "reflection_logits", "log_variance", "body_prior", "joints"):
            self.assertTrue(torch.equal(original[key], other[key]), key)

    def test_reported_joint_variance_includes_between_branch_disagreement(self):
        result = self.model()(torch.rand(1, 4, 9, 11, 9), torch.ones(1, 4))
        coarse_mass = result["history_attention"]["scale3"][:, :-1].sum(1).mean((2, 3, 4)).unsqueeze(-1)
        within = ((1 - coarse_mass) * result["current_only_joint_variance"]
                  + coarse_mass * result["trusted_joint_variance"])
        between = coarse_mass * (1 - coarse_mass) * (
            result["trusted_joints"] - result["current_only_joints"]).square()
        torch.testing.assert_close(result["joint_variance"], within + between)

    def test_sparse_startup_head_is_numerically_conservative(self):
        torch.manual_seed(5)
        model = self.model()
        raw = torch.zeros(1, 4, 9, 11, 9)
        raw[:, :, 4, 5, 4] = 10.
        result = model(raw, torch.ones(1, 4))
        for name, expected in (("intensity", .1), ("reflection_logits", -2.),
                               ("log_variance", 2 * torch.log(torch.tensor(.5)).item())):
            self.assertTrue(torch.isfinite(result[name]).all(), name)
            torch.testing.assert_close(result[name], torch.full_like(result[name], expected),
                                        rtol=0, atol=.1)


if __name__ == "__main__":
    unittest.main()
