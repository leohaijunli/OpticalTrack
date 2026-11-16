"""Subpixel mode preserves measured centroids without changing legacy defaults."""
from unittest import mock
import json
from pathlib import Path
import tempfile
import unittest
import types

import numpy as np

from tests import test_mocap_adapter as mocap_tests
from tests.test_mocap_adapter import helpers, TrackingPipeline, cameras
from multi_body import rigid_pose, _centered_local
from multi_view import _validated_points


class SubpixelTrackingTests(unittest.TestCase):
    def image(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[1, 1] = image[1, 2] = image[2, 1] = 255
        return image

    def test_float_centroid_is_not_integer_plus_a_constant(self):
        image = self.image()
        integer = helpers.detect_visible_dots(image, 140, 3, 200)
        floating = helpers.detect_visible_dots(image, 140, 3, 200, "subpixel")
        self.assertEqual(integer, [[1, 1]])
        np.testing.assert_allclose(floating, [[4 / 3, 4 / 3]], atol=1e-12)
        self.assertTrue(all(type(value) is int for value in integer[0]))
        self.assertTrue(all(type(value) is float for value in floating[0]))

    def test_area_filtering_and_empty_frames_keep_the_same_point_counts(self):
        for image in (self.image(), np.zeros((10, 10), dtype=np.uint8)):
            integer = helpers.detect_visible_dots(image, 140, 3, 200)
            floating = helpers.detect_visible_dots(image, 140, 3, 200, "subpixel")
            self.assertEqual(len(integer), len(floating))
        with self.assertRaises(ValueError):
            helpers.detect_visible_dots(self.image(), centroid_mode="unknown")

    def test_server_rounds_only_drawing_not_analysis_coordinates(self):
        image = np.repeat(self.image()[:, :, None], 3, axis=2)
        camera = types.SimpleNamespace(visible_mode=True)
        with mock.patch.dict("os.environ", {"MOCAP_CENTROID_MODE": "subpixel", "MOCAP_DOT_THRESHOLD": "140"}):
            _, points = helpers.Cameras._decorated._find_dot(camera, image)
        np.testing.assert_allclose(points, [[4 / 3, 4 / 3]], atol=1e-12)

    def test_pipeline_records_nondefault_mode_and_keeps_integer_default(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "calibration.json"
            calibration.write_text(json.dumps({"cameras": [
                {"name": f"CAM{i + 1:02d}", "K": p["K"].tolist(),
                 "R": p["R"].tolist(), "t": p["t"].ravel().tolist()}
                for i, p in enumerate(cameras())]}), encoding="utf-8")
            rig = Path(directory) / "rig.json"
            rig.write_text(json.dumps({"bodies": [mocap_tests.RigidBodyExtensionTests.BODY]}), encoding="utf-8")
            integer = TrackingPipeline(calibration, rig, blur_size=1)
            floating = TrackingPipeline(calibration, rig, blur_size=1, centroid_mode="subpixel")
            self.assertEqual(integer.centroid_mode, "integer")
            self.assertNotIn("centroid_mode", integer.configuration)
            self.assertEqual(floating.configuration["centroid_mode"], "subpixel")
            with mock.patch.object(floating, "process_detections", side_effect=lambda points, t, valid: points):
                result = floating.process([self.image()] * 3, 0)
            np.testing.assert_allclose(result, [[[4 / 3, 4 / 3]]] * 3, atol=1e-12)

    def test_regular_and_ragged_missing_points_have_no_fabricated_zero(self):
        for observations in ([[1, 2], [None, 3], [np.inf, 4], [5, 6]],
                             [[1, 2], [None, 3], [9], [5, 6]]):
            np.testing.assert_array_equal(_validated_points(observations), [[1, 2], [5, 6]])
        self.assertEqual(_validated_points([]).shape, (0, 2))

    def test_local_geometry_cache_detects_in_place_template_edits(self):
        local = np.array([[0, 0, 0], [1, 0, 0], [.2, .8, 0], [.3, .1, .6]], dtype=float)
        world = local + [1, 2, 3]
        _centered_local.cache_clear()
        first = rigid_pose(local, world)
        np.testing.assert_allclose(first[1], [1, 2, 3], atol=1e-12)
        local[:, 0] += .2
        second = rigid_pose(local, world)
        np.testing.assert_allclose(second[1], [.8, 2, 3], atol=1e-12)
        self.assertEqual(_centered_local.cache_info().misses, 2)


if __name__ == "__main__":
    unittest.main()
