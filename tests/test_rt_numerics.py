"""CPU-reproducible AMP overflow recovery and robust gradient clipping."""

import json
import math
import unittest
from unittest import mock

import torch

from rt_completion.numerics import (graceful_load_scaler, make_grad_scaler,
                                    precision_policy, safe_optimizer_step)


class PrecisionPolicyTests(unittest.TestCase):
    def test_cpu_auto_no_amp_and_explicit_bfloat16_no_scaling(self):
        policy = precision_policy({"train": {"amp": True}}, "cpu")
        self.assertEqual(policy, {"enabled": False, "dtype": torch.float32,
                                  "scaling": False, "name": "float32"})
        policy = precision_policy({"train": {"amp": True, "amp_dtype": "bfloat16"}}, "cpu")
        self.assertTrue(policy["enabled"])
        self.assertFalse(make_grad_scaler(policy, "cpu").is_enabled())

    def test_cuda_auto_chooses_native_bfloat16_without_tensor_probe(self):
        with mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)) as capability, \
                mock.patch("torch.version.cuda", "11.8"), \
                mock.patch("torch.cuda.device", side_effect=AssertionError("must not create a context")):
            policy = precision_policy({"train": {"amp": True}}, "cuda:1")
        capability.assert_called_once_with(torch.device("cuda:1"))
        self.assertEqual(policy["dtype"], torch.bfloat16)
        self.assertFalse(policy["scaling"])
        self.assertFalse(make_grad_scaler(policy, "cuda:1").is_enabled())

    def test_cuda_auto_falls_back_to_float16(self):
        with mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)), \
                mock.patch("torch.version.cuda", "11.8"):
            policy = precision_policy({"train": {"amp": True}}, "cuda:0")
        self.assertEqual(policy["dtype"], torch.float16)
        self.assertTrue(policy["scaling"])

    def test_amp_false_avoids_cuda_queries_and_invalid_dtype_rejected(self):
        with mock.patch("torch.cuda.get_device_capability", side_effect=AssertionError("CUDA query")):
            policy = precision_policy({"train": {"amp": False}}, "cuda:0")
        self.assertFalse(policy["enabled"])
        with self.assertRaises(ValueError):
            precision_policy({"train": {"amp": True, "amp_dtype": "half-ish"}}, "cpu")

    def test_old_bfloat16_scaler_state_ignored_and_float16_restored(self):
        legacy = torch.amp.GradScaler("cpu", init_scale=2.0 ** 100)
        disabled = torch.amp.GradScaler("cpu", enabled=False)
        self.assertFalse(graceful_load_scaler(disabled, legacy.state_dict()))
        self.assertEqual(disabled.state_dict(), {})
        enabled = torch.amp.GradScaler("cpu")
        self.assertTrue(graceful_load_scaler(enabled, legacy.state_dict()))
        self.assertEqual(enabled.get_scale(), legacy.get_scale())
        self.assertFalse(graceful_load_scaler(enabled, {}))


class SafeOptimizerStepTests(unittest.TestCase):
    def model(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        return model

    def test_finite_loss_overflow_skips_optimizer_and_backs_off_scaler(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=1e38)
        loss = (model(torch.tensor([[2.0]])) - 0).square().mean()
        self.assertTrue(torch.isfinite(loss))
        scaler.scale(loss).backward()
        self.assertFalse(torch.isfinite(model.weight.grad).all())
        result = safe_optimizer_step(model, optimizer, scaler, 5.)
        self.assertFalse(result["updated"])
        self.assertTrue(result["overflow"])
        self.assertEqual(result["reason"], "amp_overflow")
        self.assertLess(result["scale_after"], result["scale_before"])
        self.assertEqual(result["nonfinite_parameters"], ["weight"])
        self.assertIsNone(result["grad_norm"])
        self.assertEqual(model.weight.item(), 1.)
        self.assertFalse(optimizer.state)
        self.assertIsNone(model.weight.grad)
        json.dumps(result, allow_nan=False)
        # A skipped update must leave the scaler usable on the next batch.
        scaler.scale(model(torch.ones(1, 1)).square().mean()).backward()
        following = safe_optimizer_step(model, optimizer, scaler, 5.)
        self.assertTrue(following["updated"])
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertTrue(optimizer.state)

    def test_finite_large_gradients_have_finite_wide_norm_and_clip(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        model.weight.grad = torch.full_like(model.weight, 1e20)
        self.assertFalse(torch.isfinite(torch.linalg.vector_norm(model.weight.grad)))
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        result = safe_optimizer_step(model, optimizer, None, 5.)
        self.assertTrue(result["updated"])
        self.assertAlmostEqual(result["grad_norm"] / (math.sqrt(2) * 1e20), 1., places=6)
        self.assertAlmostEqual(model.weight.grad.norm().item(), 5., places=5)
        self.assertTrue(torch.isfinite(model.weight).all())

    def test_nonfinite_without_scaler_never_advances_adam_state(self):
        for bad in (float("inf"), float("nan"), -float("inf")):
            with self.subTest(value=bad):
                model = self.model()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
                model(torch.ones(1, 1)).sum().backward()
                safe_optimizer_step(model, optimizer, None, 5.)
                before = model.weight.detach().clone()
                step_before = optimizer.state[model.weight]["step"].clone()
                model.weight.grad = torch.full_like(model.weight, bad)
                result = safe_optimizer_step(model, optimizer, None, 5.)
                self.assertFalse(result["updated"])
                self.assertEqual(result["reason"], "nonfinite_gradient")
                self.assertTrue(torch.equal(model.weight, before))
                self.assertTrue(torch.equal(optimizer.state[model.weight]["step"], step_before))
                self.assertIsNone(model.weight.grad)
                json.dumps(result, allow_nan=False)

    def test_finite_update_matches_unscaled_reference(self):
        for enabled in (False, True):
            with self.subTest(scaling=enabled):
                model = self.model()
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                scaler = torch.amp.GradScaler("cpu", enabled=enabled)
                scaler.scale(model(torch.ones(1, 1)).square().mean()).backward()
                result = safe_optimizer_step(model, optimizer, scaler, 5.)
                self.assertTrue(result["updated"])
                self.assertAlmostEqual(result["grad_norm"], 2.)
                self.assertAlmostEqual(model.weight.item(), .8, places=6)

    def test_empty_gradients_are_not_an_amp_failure(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        result = safe_optimizer_step(model, optimizer, torch.amp.GradScaler("cpu"), 5.)
        self.assertFalse(result["updated"])
        self.assertEqual(result["reason"], "no_gradients")
        self.assertFalse(optimizer.state)

    def test_unrepresentable_double_norm_does_not_mislabel_finite_elements(self):
        model = torch.nn.Linear(2, 1, bias=False).double()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        before = model.weight.detach().clone()
        model.weight.grad = torch.full_like(model.weight, 1.7e308)
        result = safe_optimizer_step(model, optimizer, None, 5.)
        self.assertFalse(result["updated"])
        self.assertEqual(result["reason"], "nonfinite_global_norm")
        self.assertEqual(result["nonfinite_parameters"], [])
        self.assertTrue(torch.equal(model.weight, before))
        json.dumps(result, allow_nan=False)

    def test_invalid_grad_clip_is_rejected(self):
        model = self.model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                safe_optimizer_step(model, optimizer, None, value)


if __name__ == "__main__":
    unittest.main()
