"""Optimized fusion must preserve ordering, missing views and live calibration."""
import copy
import unittest

import numpy as np

from tests.test_mocap_adapter import helpers
from multi_view import fuse_points, _geometry


def scene():
    poses = []
    for index in range(6):
        angle = index * np.pi / 3
        center = np.array([3 * np.cos(angle), 3 * np.sin(angle), 3.5])
        forward = np.array([0, 0, 1.2]) - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        R = np.array([right, np.cross(forward, right), forward])
        K = np.array([[700 + index * 30, 0, 640 + index],
                      [0, 730 + index * 40, 400 - index], [0, 0, 1]], float)
        poses.append({"K": K, "R": R, "t": (-R @ center).reshape(3, 1)})
    rng = np.random.default_rng(87)
    world = rng.uniform([-.6, -.5, .8], [.6, .5, 1.6], (10, 3))
    return poses, world


def project(world, poses):
    result = []
    for pose in poses:
        uvw = (pose["K"] @ (pose["R"] @ world.T + pose["t"])).T
        result.append((uvw[:, :2] / uvw[:, 2, None]).tolist())
    return result


class MultiViewTests(unittest.TestCase):
    def assert_reference(self, observations, poses, batch_size=2048):
        before = copy.deepcopy(observations)
        expected = helpers.find_point_correspondance_and_object_points_reference(observations, poses)
        actual = fuse_points(observations, poses, [p["K"] for p in poses],
                             helpers._fundamental_from_projections, batch_size=batch_size)
        self.assertEqual(actual[1].shape, expected[1].shape)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-8, rtol=1e-7)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-8, rtol=1e-7)
        self.assertEqual(observations, before)

    def test_heterogeneous_calibration_missing_first_and_internal_views(self):
        poses, world = scene()
        observations = project(world, poses)
        observations[0] = []
        observations[2] = observations[2][2:]
        observations[4] = [[None, None]] + observations[4][:-2]
        self.assert_reference(observations, poses)

    def test_chunk_boundaries_duplicate_pixels_and_camera_order(self):
        poses, world = scene()
        observations = project(world[:5], poses)
        observations[0].append(observations[0][0].copy())
        order = [4, 0, 5, 1, 3, 2]
        for size in (1, 3, 2048):
            with self.subTest(batch_size=size):
                self.assert_reference([observations[i] for i in order], [poses[i] for i in order], size)

    def test_noisy_points_keep_the_same_candidate_winners(self):
        poses, world = scene()
        rng = np.random.default_rng(63)
        observations = [(np.asarray(pixels) + rng.normal(0, .5, (len(pixels), 2))).tolist()
                        for pixels in project(world, poses)]
        self.assert_reference(observations, poses)

    def test_calibration_value_cache_detects_in_place_edits(self):
        poses, world = scene()
        calls = []

        def fundamental(P1, P2):
            calls.append(1)
            return helpers._fundamental_from_projections(P1, P2)

        def run():
            return fuse_points(project(world[:2], poses), poses, [p["K"] for p in poses], fundamental)

        _geometry.cache_clear()
        original = run()
        self.assertEqual(len(calls), 15)
        repeated = run()
        self.assertEqual(len(calls), 15)
        np.testing.assert_array_equal(original[1], repeated[1])
        poses[2]["K"][0, 0] *= 1.1
        poses[3]["t"][0, 0] += .07
        updated = run()
        self.assertEqual(len(calls), 30)
        expected = helpers.find_point_correspondance_and_object_points_reference(project(world[:2], poses), poses)
        np.testing.assert_allclose(updated[1], expected[1], atol=1e-8, rtol=1e-7)
        np.testing.assert_allclose(updated[0], expected[0], atol=1e-8, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
