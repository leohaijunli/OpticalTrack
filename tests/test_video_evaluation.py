"""Independent checks for capture indexing and accuracy reference conventions."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from tools.accuracy_stats3 import axis_error, truth_tables
from tools.evaluate_multi_video import video_frames


class VideoEvaluationTests(unittest.TestCase):
    def test_sparse_video_selection_uses_requested_pixels(self):
        capture = mock.Mock()
        cursor = [0]

        def seek(prop, frame):
            self.assertEqual(prop, cv.CAP_PROP_POS_FRAMES)
            cursor[0] = frame
            return True

        def read():
            image = np.full((4, 4), cursor[0], dtype=np.uint8)
            cursor[0] += 1
            return True, image

        capture.set.side_effect = seek
        capture.read.side_effect = read
        decoded = list(video_frames([capture], [2, 3, 9]))
        self.assertEqual([int(images[0][0, 0]) for images in decoded], [2, 3, 9])
        self.assertEqual(capture.set.call_count, 2)

    def test_heading_sign_and_velocity_clock_follow_the_reference(self):
        rotation = Rotation.from_euler("z", 45, degrees=True)
        qx, qy, qz, qw = rotation.as_quat()
        template = {"body_id": 7, "base_pair": (0, 1),
                    "locals": np.array([[0, 0, 0], [1, 0, 0], [.5, .4, 0]]),
                    "base_mid_local": np.array([.5, 0, 0])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["body_id", "frame", "x", "y", "z", "qw", "qx", "qy", "qz"])
                for frame, x in ((0, 0), (3, .1)):
                    writer.writerow([7, frame, x, 0, 1, qw, qx, qy, qz])
            truth = truth_tables(path, [template], fps=60)[7]
        self.assertAlmostEqual(axis_error(truth["heading"][0], -np.pi / 4), 0)
        self.assertAlmostEqual(axis_error(truth["heading"][0], np.pi / 4), np.pi / 2)
        np.testing.assert_allclose(truth["velocity"][3], [2, 0, 0], atol=1e-10)
        self.assertAlmostEqual(truth["speed"][3], 2)

    def test_origin_truth_has_no_rotational_velocity_or_acceleration(self):
        template = {"body_id": 0, "base_pair": (0, 1),
                    "locals": np.array([[.2, -.1, .1], [.2, .1, .1], [-.1, 0, .1]]),
                    "base_mid_local": np.array([.2, 0, .1])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["body_id", "frame", "x", "y", "z", "qw", "qx", "qy", "qz"])
                for frame, angle in enumerate((0, 90, 180)):
                    qx, qy, qz, qw = Rotation.from_euler("z", angle, degrees=True).as_quat()
                    writer.writerow([0, frame, 1, 2, 3, qw, qx, qy, qz])
            origin = truth_tables(path, [template], position_reference="body_origin")[0]
            midpoint = truth_tables(path, [template], position_reference="base_midpoint")[0]
        np.testing.assert_allclose(origin["pos"][2], [1, 2, 3])
        np.testing.assert_allclose(origin["velocity"][2], [0, 0, 0])
        np.testing.assert_allclose(origin["acceleration"][2], [0, 0, 0])
        self.assertGreater(midpoint["speed"][2], 8)


if __name__ == "__main__":
    unittest.main()
