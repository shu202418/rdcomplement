"""CPU-only tests for conservative-cache and frozen-probe helpers."""

import unittest

import numpy as np

from rt_completion.cache_variants import prune_record
from rt_completion.mesh_validation import summarize_action_sums


class ValidationToolTests(unittest.TestCase):
    def test_prune_record_filters_ranks_scales_and_sorts(self):
        record = {
            "indices": np.array([8, 3, 5, 1], np.int32),
            "values": np.array([1., .5, 2., 4.], np.float32),
            "probability": np.array([1., 1., .8, .25], np.float16),
            "sigma_log": np.array([.2, .05, .1, .5], np.float16),
            "joints": np.ones((22, 3), np.float32),
        }
        result = prune_record(record, max_added_voxels=2, max_sigma_log=.2,
                              intensity_multiplier=.7)
        np.testing.assert_array_equal(result["indices"], [3, 5])
        np.testing.assert_allclose(result["values"], [.35, 1.4], rtol=1e-6)
        np.testing.assert_allclose(result["probability"], [1., .8], rtol=1e-3)
        np.testing.assert_allclose(result["sigma_log"], [.05, .1], rtol=1e-3)
        np.testing.assert_array_equal(result["joints"], record["joints"])
        np.testing.assert_array_equal(record["indices"], [8, 3, 5, 1])

    def test_action_summary_is_micro_average_and_skips_empty_actions(self):
        counts = np.zeros(50, np.int64)
        counts[:2] = [2, 1]
        sums = {
            "mve_mm": np.r_[np.array([20., 40.]), np.zeros(48)],
            "gender_accuracy": np.r_[np.array([1., 1.]), np.zeros(48)],
        }
        overall, actions = summarize_action_sums(sums, counts)
        self.assertAlmostEqual(overall["mve_mm"], 20.)
        self.assertAlmostEqual(overall["gender_accuracy"], 2 / 3)
        self.assertEqual(actions["1"]["frames"], 2)
        self.assertEqual(actions["1"]["mve_mm"], 10.)
        self.assertEqual(actions["2"]["mve_mm"], 40.)
        self.assertNotIn("3", actions)


if __name__ == "__main__":
    unittest.main()
