"""Equivalence checks for shared-grid sampling and temporal ROI reuse."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rt_completion import geometry


BOUNDS = [[-1., 1.]] * 3
GRID_SAMPLE = F.grid_sample


def body():
    joints = torch.zeros(1, 22, 3)
    joints[0, :, 0] = torch.linspace(-.7, .7, 22)
    return joints


def separate_values_and_support(volume, grid, **kwargs):
    """Reference the two independent grid_sample calls used before fusion."""
    values = GRID_SAMPLE(volume[:, :-1], grid, **kwargs)
    support = GRID_SAMPLE(volume[:, -1:], grid, **kwargs)
    return torch.cat((values, support), dim=1)


class GeometrySpeedEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        generator = torch.Generator().manual_seed(731)
        volume = torch.rand((1, 3, 9, 9, 9), generator=generator)
        support = torch.rand((1, 1, 9, 9, 9), generator=generator)
        source = body()
        target = source + torch.tensor([.03, .07, -.04])
        return volume, support, source, target

    def evaluate(self, split_sampling):
        volume, support, source, target = [tensor.requires_grad_() for tensor in self.fixture()]
        if split_sampling:
            with patch("rt_completion.geometry.F.grid_sample", side_effect=separate_values_and_support):
                result = geometry.articulated_warp(volume, source, target, BOUNDS, .3,
                                                   source_support=support)
        else:
            with patch("rt_completion.geometry.F.grid_sample", wraps=GRID_SAMPLE) as sampled:
                result = geometry.articulated_warp(volume, source, target, BOUNDS, .3,
                                                   source_support=support)
            self.assertEqual(sampled.call_count, 1)
        (result["values"].square().mean() + .2 * result["support"].square().mean()
         + .1 * result["quality"].mean()).backward()
        return result, [tensor.grad for tensor in (volume, support, source, target)]

    def test_fused_sampling_matches_separate_forward_and_gradients(self):
        optimized, gradients = self.evaluate(False)
        reference, reference_gradients = self.evaluate(True)
        for name in optimized:
            torch.testing.assert_close(optimized[name], reference[name], rtol=0, atol=0)
        for gradient, expected in zip(gradients, reference_gradients):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            torch.testing.assert_close(gradient, expected, rtol=2e-5, atol=2e-7)

    def test_cached_source_body_matches_computed_body(self):
        volume, support, source, target = self.fixture()
        source_body = geometry._body_prior(source, volume.shape[-3:], BOUNDS, .3) >= .25
        reference = geometry.articulated_warp(volume, source, target, BOUNDS, .3,
                                             source_support=support)
        with patch("rt_completion.geometry._body_prior", side_effect=AssertionError("ROI recomputed")):
            optimized = geometry.articulated_warp(volume, source, target, BOUNDS, .3,
                                                 source_support=support, source_body_mask=source_body)
        for name in reference:
            torch.testing.assert_close(optimized[name], reference[name], rtol=0, atol=0)

    def test_source_body_mask_requires_boolean_matching_shape(self):
        volume, support, source, target = self.fixture()
        for invalid in (torch.ones_like(support), torch.ones(1, 9, 9, 9, dtype=torch.bool)):
            with self.assertRaisesRegex(ValueError, "source_body_mask"):
                geometry.articulated_warp(volume, source, target, BOUNDS, .3,
                                          source_body_mask=invalid)

    def test_teacher_reuses_roi_and_matches_recomputed_two_sample_reference(self):
        raw = torch.zeros(1, 4, 9, 9, 9)
        raw[:, :3, 4, 4, 4] = torch.tensor([2., 3., 1.5])
        raw[:, 0, 5, 4, 4] = 2.
        raw[:, 1, 4, 5, 4] = 2.
        joints = body()[:, None].expand(-1, 4, -1, -1).clone()
        joints[:, -1, :, 1] += .07
        history = torch.tensor([[1., 1., 1., 0.]])
        original_warp = geometry.articulated_warp
        original_prior = geometry._body_prior

        def recompute_source_body(*args, **kwargs):
            kwargs.pop("source_body_mask", None)
            return original_warp(*args, **kwargs)

        # Include discarded padded history, degenerate geometry and empty data.
        for raw_case, pose_case, history_case in (
                (raw, joints, history), (raw, joints, history * 0),
                (raw, torch.zeros_like(joints), history), (raw * 0, joints, history)):
            with patch("rt_completion.geometry._body_prior", wraps=original_prior) as prior:
                optimized = geometry.temporal_target_v2(raw_case, pose_case, history_case,
                                                        BOUNDS, .3, {})
            self.assertEqual(prior.call_count, 4)  # Three sources plus one target.
            with patch("rt_completion.geometry.articulated_warp", side_effect=recompute_source_body), \
                    patch("rt_completion.geometry.F.grid_sample", side_effect=separate_values_and_support), \
                    patch("rt_completion.geometry._body_prior", wraps=original_prior) as prior:
                reference = geometry.temporal_target_v2(raw_case, pose_case, history_case,
                                                        BOUNDS, .3, {})
            self.assertEqual(prior.call_count, 7)
            for name in reference:
                torch.testing.assert_close(optimized[name], reference[name], rtol=0, atol=0)
                self.assertFalse(optimized[name].requires_grad)


if __name__ == "__main__":
    unittest.main()
