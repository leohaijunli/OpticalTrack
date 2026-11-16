"""Keep deployment error summaries honest when tracking is unavailable."""
import unittest

import numpy as np

from tools.evaluate_deployment import score


class DeploymentEvaluationTests(unittest.TestCase):
    def truth(self):
        return {0: {"pos": {i: np.array([i / 30, 0, 0]) for i in range(3)},
                    "speed": {0: None, 1: 1.0, 2: 1.0},
                    "rotation": {i: np.eye(3) for i in range(3)}}}

    def test_missing_frame_reduces_coverage_without_filling_error_with_zero(self):
        records = []
        for frame in range(3):
            objects = [] if frame == 1 else [{
                "droneIndex": 0, "pos": [frame / 30, .003, 0],
                "vel": [1, 0, 0], "pose_r": np.eye(3)}]
            records.append({"frame": frame, "filtered_objects": objects})
        result = score(records, self.truth(), [0], 30)
        body = result["bodies"]["0"]
        self.assertAlmostEqual(body["coverage_percent"], 200 / 3)
        self.assertAlmostEqual(result["all_bodies_available_percent"], 200 / 3)
        self.assertEqual(body["position_error_mm"]["count"], 2)
        self.assertAlmostEqual(body["position_error_mm"]["mean"], 3)
        self.assertEqual(body["longest_missing_frames"], 1)
        self.assertAlmostEqual(body["longest_missing_duration_s"], 1 / 30)
        self.assertEqual(body["speed_absolute_error_mps"]["mean"], 0)

    def test_total_loss_has_no_invented_accuracy(self):
        records = [{"frame": frame, "filtered_objects": []} for frame in range(3)]
        result = score(records, self.truth(), [0], 30)
        self.assertEqual(result["all_bodies_available_percent"], 0)
        self.assertIsNone(result["pooled_position_error_mm"]["mean"])
        self.assertEqual(result["bodies"]["0"]["longest_missing_frames"], 3)


if __name__ == "__main__":
    unittest.main()
