"""Error distributions count losses separately and retain signed XYZ errors."""
import json
import unittest

import numpy as np

from tools.benchmark_fusion import numeric_differences
from tools.plot_tracking_distribution import series, summarize


class TrackingDistributionTests(unittest.TestCase):
    def test_euclidean_quantiles_and_thresholds_use_only_valid_samples(self):
        vectors = [[3, 4, 0], [np.nan] * 3, [0, 0, 10], [np.nan] * 3, [np.nan] * 3]
        result = summarize(vectors, [5, 6, 7, 8, 9], 30)
        self.assertEqual(result["valid_frames"], 2)
        self.assertEqual(result["missing_frames"], 3)
        self.assertEqual(result["missing_frame_ids"], [6, 8, 9])
        self.assertEqual(result["coverage_percent"], 40)
        self.assertAlmostEqual(result["mean_mm"], 7.5)
        self.assertAlmostEqual(result["rmse_mm"], np.sqrt(62.5))
        self.assertAlmostEqual(result["p95_mm"], 9.75)
        self.assertAlmostEqual(result["p99_mm"], 9.95)
        self.assertEqual(result["over_5mm_valid_percent"], 50)
        self.assertEqual(result["over_10mm_valid_percent"], 0)
        self.assertEqual(result["within_5mm_all_frames_percent"], 20)
        self.assertEqual(result["longest_missing_frames"], 2)
        self.assertAlmostEqual(result["longest_missing_duration_s"], 2 / 30)
        self.assertEqual(result["largest_errors"][0]["frame"], 7)
        np.testing.assert_allclose(result["bias_xyz_mm"], [1.5, 2, 5])
        json.dumps(result, allow_nan=False)

    def test_no_solution_does_not_invent_zero_error(self):
        result = summarize(np.full((3, 3), np.nan), [10, 11, 12], 60)
        self.assertEqual(result["coverage_percent"], 0)
        self.assertEqual(result["longest_missing_frames"], 3)
        self.assertAlmostEqual(result["longest_missing_duration_s"], .05)
        self.assertEqual(result["within_5mm_all_frames_percent"], 0)
        self.assertIsNone(result["mean_mm"])
        self.assertIsNone(result["p99_mm"])
        self.assertEqual(result["bias_xyz_mm"], [None] * 3)
        json.dumps(result, allow_nan=False)

    def test_series_uses_explicit_body_id_and_preserves_error_sign(self):
        records = [
            {"reference": {"7": [1, 2, 3]},
             "objects": [{"droneIndex": 1, "pos": [50, 50, 50]},
                         {"droneIndex": 7, "pos": [1.001, 1.998, 3.003]}],
             "filtered_objects": [{"droneIndex": 7, "pos": [1.004, 2, 2.995]}]},
            {"reference": {"7": [1, 2, 3]}, "objects": [], "filtered_objects": []}]
        filtered, raw = series(records, 7)
        np.testing.assert_allclose(filtered[0], [4, 0, -5], atol=1e-10)
        np.testing.assert_allclose(raw[0], [1, -2, 3], atol=1e-10)
        self.assertTrue(np.isnan(filtered[1]).all())
        self.assertTrue(np.isnan(raw[1]).all())

    def test_empty_misaligned_or_sparse_inputs_are_rejected(self):
        for vectors, frames, fps in (([], [], 30), ([[0, 0, 0]], [0, 1], 30),
                                     ([[0, 0, 0]], [0], 0),
                                     ([[0, 0, 0]], [0], np.nan),
                                     ([[0, 0, 0], [0, 0, 0]], [0, 5], 30)):
            with self.subTest(frames=frames, fps=fps), self.assertRaises(ValueError):
                summarize(vectors, frames, fps)

    def test_numeric_regression_tolerates_roundoff_not_identity_or_missing_fields(self):
        expected = {"droneIndex": 7, "pos": [1.0, 2.0, 3.0], "pose_mode": "rigid"}
        self.assertEqual(numeric_differences(dict(expected, pos=[1.0 + 1e-12, 2.0, 3.0]), expected), [])
        for actual in (dict(expected, droneIndex=8), dict(expected, pose_mode="triangle3"),
                       {"droneIndex": 7, "pos": expected["pos"]},
                       dict(expected, pos=[1.0, 2.0, 3.01])):
            with self.subTest(actual=actual):
                self.assertTrue(numeric_differences(actual, expected))


if __name__ == "__main__":
    unittest.main()
