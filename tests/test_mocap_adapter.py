"""Regression tests for repository fidelity and input/timing boundaries."""
import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import cv2 as cv
import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters" / "low_cost_mocap"
API = ADAPTER / "low-cost-mocap" / "computer_code" / "api"
sys.path.insert(0, str(ADAPTER))
sys.path.insert(0, str(API))
import helpers
from KalmanFilter import KalmanFilter
from LowPassFilter import LowPassFilter
from multi_body import (BodyTracker, load_body_templates,
                        locate_bodies, rigid_pose)
from video_replay import VideoReplay, discover_clips, _VideoFileReader
from tracking_pipeline import TrackingPipeline


def cameras():
    poses = []
    for index in range(3):
        K = np.array([[700 + 100 * index, 0, 640], [0, 720 + 100 * index, 400], [0, 0, 1]], float)
        poses.append({"R": np.eye(3), "t": np.array([[-index * .7], [0], [0]], float), "K": K})
    return poses


def pixels(point, poses):
    return [((pose["K"] @ (pose["R"] @ point.reshape(3, 1) + pose["t"])).ravel())
            for pose in poses]


def original_algorithms(poses):
    """Load ONLY numeric function ASTs, not pseyepy/the upstream server."""
    wanted = {"triangulate_point", "triangulate_points", "calculate_reprojection_error",
              "calculate_reprojection_errors", "find_point_correspondance_and_object_points",
              "locate_objects", "cartesian_product", "drawlines"}
    tree = ast.parse((API / "helpers.orig.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    stub = types.SimpleNamespace(camera_params=[{"intrinsic_matrix": pose["K"]} for pose in poses],
                                 get_camera_params=lambda i: {"intrinsic_matrix": poses[i]["K"]})
    class CVProxy:
        sfm = types.SimpleNamespace(fundamentalFromProjections=helpers.fundamental_from_projections)
        def __getattr__(self, name):
            return getattr(cv, name)
    namespace = {"np": np, "cv": CVProxy(), "linalg": linalg, "copy": copy,
                 "Cameras": types.SimpleNamespace(instance=lambda: stub)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "helpers.orig.py", "exec"), namespace)
    return namespace


class RepositoryRegressionTests(unittest.TestCase):
    def test_dlt_matches_original_when_all_views_present(self):
        poses = cameras()
        truth = np.array([.2, .1, 4.0])
        observations = [uv[:2] / uv[2] for uv in pixels(truth, poses)]
        original = original_algorithms(poses)["triangulate_point"](observations, poses)
        actual = helpers.triangulate_point(observations, poses)
        np.testing.assert_allclose(actual, original, atol=1e-9)

    def test_missing_view_keeps_original_camera_intrinsics(self):
        poses = cameras()
        truth = np.array([.2, .1, 4.0])
        observations = [uv[:2] / uv[2] for uv in pixels(truth, poses)]
        observations[0] = [None, None]
        with mock.patch.object(helpers.Cameras, "instance", side_effect=AssertionError("hidden source opened")):
            actual = helpers.triangulate_point(observations, poses)
        np.testing.assert_allclose(actual, truth, atol=1e-8)

    def test_epipolar_matching_matches_original_baseline(self):
        poses = cameras()
        # Use identical intrinsics to compare the exact legacy numerical path.
        for pose in poses:
            pose["K"] = poses[0]["K"].copy()
        points = np.array([[.2, .1, 4], [.3, .3, 4.2], [-.2, -.2, 3.8]])
        image_points = [[] for _ in poses]
        for point in points:
            for index, uv in enumerate(pixels(point, poses)):
                image_points[index].append((uv[:2] / uv[2]).tolist())
        frames = [np.zeros((8, 8, 3), np.uint8) for _ in poses]
        original = original_algorithms(poses)["find_point_correspondance_and_object_points"](
            copy.deepcopy(image_points), poses, frames)
        actual = helpers.find_point_correspondance_and_object_points(image_points, poses)
        np.testing.assert_allclose(actual[1], original[1], atol=1e-8)
        np.testing.assert_allclose(actual[0], original[0], atol=1e-8)

    def test_triangle_position_heading_match_upstream(self):
        # A *clearly* isosceles rig (base 0.15 m, apex-edge 0.095 m), so the
        # apex search is unambiguous.  For EVERY point ordering, our extension
        # must return the same base midpoint / heading as upstream running on
        # the same ordering (re-ordering flips the base pair, hence heading,
        # so compare per-permutation, not against one fixed ordering).
        altitude = np.sqrt(.095 ** 2 - .075 ** 2)
        points = np.array([[.0, -.075, 1.0], [.0, .075, 1.0], [-altitude, .0, 1.0]])
        errors = np.array([.1, .2, .3])
        template = {"body_id": 0, "name": "test", "dist1": .095, "dist2": .15,
                    "edge_separation_m": .055}
        for order in itertools.permutations(range(3)):
            reordered, err = points[list(order)], errors[list(order)]
            original = original_algorithms(cameras())["locate_objects"](reordered.copy(), err)[0]
            actual = locate_bodies(reordered, err, [template])[0]
            np.testing.assert_allclose(actual["pos"], original["pos"])
            self.assertAlmostEqual(actual["heading"], original["heading"])
            self.assertAlmostEqual(actual["error"], original["error"])

    def test_empty_camera_lists_do_not_crash_argmin(self):
        errors, points, _ = helpers.find_point_correspondance_and_object_points([[], [], []], cameras())
        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(errors.size, 0)

    def test_markers_cannot_be_consumed_by_two_bodies(self):
        altitude = np.sqrt(.095 ** 2 - .075 ** 2)
        points = np.array([[0, -.075, 1], [0, .075, 1], [-altitude, 0, 1], [altitude, 0, 1]])
        base = {"name": "test", "dist1": .095, "dist2": .15, "edge_separation_m": .055}
        objects = locate_bodies(points, np.zeros(4), [dict(base, body_id=0), dict(base, body_id=1)])
        self.assertEqual(len(objects), 1)

    def test_candidates_are_json_serializable_for_the_server(self):
        altitude = np.sqrt(.095 ** 2 - .075 ** 2)
        points = np.array([[0, -.075, 1], [0, .075, 1], [-altitude, 0, 1]])
        template = {"body_id": 0, "name": "test", "dist1": .095, "dist2": .15,
                    "edge_separation_m": .055}
        objects = locate_bodies(points, np.zeros(3), [template])
        from helpers import _jsonable
        json.dumps(_jsonable(objects[0]))  # the app emit path (no ndarray fallback)

    def test_nearly_equilateral_template_uses_exact_equal_edges(self):
        body = {"body_id": 1, "markers": [{"id": i, "local": point} for i, point in enumerate(
            [[.05, -.14, .205], [.05, .14, .205], [-.19, 0, .205]])]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rig.json"
            path.write_text(json.dumps({"bodies": [body]}), encoding="utf-8")
            template = load_body_templates(path)[0]
        self.assertEqual(template["apex"], 2)
        self.assertEqual(template["base_pair"], (0, 1))
        self.assertLess(template["edge_separation_m"], .003)

    def test_preprocessor_does_not_shift_native_pixels(self):
        params = {"intrinsic_matrix": np.eye(3), "distortion_coef": [0] * 5, "rotation": 0}
        image = np.zeros((800, 1280), np.uint8)
        image[100, 200] = 255
        prepared = helpers.prepare_camera_frame(image, params, True, 1)
        self.assertEqual(prepared.shape, (800, 1280, 3))
        self.assertEqual(prepared[100, 200, 0], 255)

    def test_scalar_lowpass_is_numpy2_compatible(self):
        result = LowPassFilter(20, 60, 1).filter(.25)
        self.assertEqual(result.shape, (1,))
        self.assertTrue(np.isfinite(result).all())

    def test_kalman_first_frame_and_reacquisition_have_no_origin_impulse(self):
        kf = KalmanFilter(1, process_noise=1)
        def measurement(point):
            return [{"droneIndex": 0, "pos": np.array(point, np.float32), "heading": 0.0}]
        first = kf.predict_location(measurement([1, 2, 3]), timestamp=0)[0]
        np.testing.assert_allclose(first["pos"], [1, 2, 3])
        np.testing.assert_allclose(first["vel"], [0, 0, 0], atol=1e-6)
        reacquired = kf.predict_location(measurement([2, 3, 4]), timestamp=1.0)[0]
        np.testing.assert_allclose(reacquired["pos"], [2, 3, 4])
        self.assertEqual(kf.predict_location(measurement([2, 3, 4]), timestamp=1.0), [])


class RigidBodyExtensionTests(unittest.TestCase):
    """Four-marker rigs: declared reference triangle, extras, 6-DOF pose."""

    # reference triangle 0/1/2 = base 0.30 m, apex edges 0.335 m; marker 3 is
    # 60 mm off the triangle plane so it is a genuine rigidity/identity check.
    BODY = {
        "body_id": 0, "name": "quad-A", "frame": "test",
        "apex_marker_id": 2, "base_marker_ids": [0, 1],
        "markers": [
            {"id": 0, "local": [0.15, 0.0, 0.0]},
            {"id": 1, "local": [-0.15, 0.0, 0.0]},
            {"id": 2, "local": [0.0, 0.0, 0.30]},
            {"id": 3, "local": [0.05, -0.06, 0.44]},
        ],
    }

    @staticmethod
    def templates(bodies):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "rig.json"
        path.write_text(json.dumps({"bodies": bodies}), encoding="utf-8")
        return directory, load_body_templates(path)

    def test_four_marker_reference_triangle_is_declared_not_inferred(self):
        directory, templates = self.templates([self.BODY])
        try:
            template = templates[0]
            self.assertEqual(template["apex"], 2)
            self.assertEqual(template["base_pair"], (0, 1))
            self.assertEqual(template["extras"], [3])
            self.assertGreater(template["rigidity_margin_m"], 0.05)
            self.assertEqual(len(template["marker_ids"]), 4)
        finally:
            directory.cleanup()

    def test_four_marker_body_must_declare_its_reference_triangle(self):
        undeclared = copy.deepcopy(self.BODY)
        del undeclared["apex_marker_id"]
        del undeclared["base_marker_ids"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rig.json"
            path.write_text(json.dumps({"bodies": [undeclared]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_body_templates(path)

    def test_extra_marker_is_matched_and_kabsch_recovers_pose(self):
        directory, templates = self.templates([self.BODY])
        try:
            template = templates[0]
            rotation = Rotation.from_euler("zyx", [20.0, 8.0, -5.0], degrees=True)
            translation = np.array([1.0, -0.4, 1.6])
            world = (rotation.as_matrix() @ template["locals"].T).T + translation
            order = [3, 1, 2, 0]          # shuffled detection order
            objects = locate_bodies(world[order], np.zeros(4), templates)
            self.assertEqual(len(objects), 1)
            obj = objects[0]
            self.assertEqual(obj["pose_mode"], "rigid")
            self.assertEqual(obj["markers_used"], 4)
            self.assertEqual(sorted(obj["marker_ids"]), [0, 1, 2, 3])
            self.assertLess(obj["pose_rmse_m"], 1e-9)
            np.testing.assert_allclose(obj["pose_t"], translation, atol=1e-9)
            recovered = Rotation.from_matrix(obj["pose_r"])
            np.testing.assert_allclose(recovered.as_matrix(), rotation.as_matrix(),
                                       atol=1e-9)
        finally:
            directory.cleanup()

    def test_occluded_or_misplaced_extra_falls_back_to_triangle(self):
        directory, templates = self.templates([self.BODY])
        try:
            template = templates[0]
            rotation = np.eye(3)
            translation = np.array([0.0, 0.0, 1.0])
            world = template["locals"] + translation
            missing = locate_bodies(world[:3], np.zeros(3), templates)[0]
            self.assertEqual(missing["pose_mode"], "triangle3")
            self.assertEqual(missing["markers_used"], 3)
            self.assertTrue(missing["pose_ambiguous"])

            misplaced = world.copy()
            misplaced[3] += np.array([0.12, 0.0, 0.0])
            rejected = locate_bodies(misplaced, np.zeros(4), templates)[0]
            self.assertEqual(rejected["pose_mode"], "triangle3")
            accepted = locate_bodies(misplaced, np.zeros(4), templates,
                                     extra_tol=0.20)[0]
            self.assertEqual(accepted["pose_mode"], "rigid")
            self.assertEqual(accepted["markers_used"], 4)
            self.assertGreater(accepted["extra_residuals_m"][0], 0.11)
        finally:
            directory.cleanup()

    def test_temporal_prior_resolves_symmetric_triangle_flip(self):
        directory, templates = self.templates([self.BODY])
        try:
            template = templates[0]
            rotation = Rotation.from_euler("zyx", [25.0, 6.0, 0.0], degrees=True)
            translation = np.array([0.4, -0.2, 1.5])
            world = (rotation.as_matrix() @ template["locals"].T).T + translation
            truth_r = rotation.as_matrix()

            tracker = BodyTracker(templates)
            full = locate_bodies(world, np.zeros(4), templates,
                                 tracker=tracker, timestamp=0.0)[0]
            self.assertEqual(full["pose_mode"], "rigid")
            self.assertEqual(full["pose_disambiguation"], "extra_marker")

            # 33 ms later the extra marker is hidden.  The reference triangle is
            # symmetric, so the mirrored base order fits equally well; the
            # search can choose the half-turn branch (yaw for this mounting).
            hidden = world[[0, 1, 2]][[1, 0, 2]]
            bare = locate_bodies(hidden, np.zeros(3), templates)[0]
            bare_r = np.asarray(bare["pose_r"], dtype=float)
            self.assertAlmostEqual(Rotation.from_matrix(bare_r @ truth_r.T).magnitude(),
                                   np.pi, places=6)

            obj = locate_bodies(hidden, np.zeros(3), templates,
                                tracker=tracker, timestamp=1.0 / 30.0)[0]
            self.assertEqual(obj["pose_mode"], "triangle3")
            self.assertEqual(obj["pose_disambiguation"], "temporal_prior")
            obj_r = np.asarray(obj["pose_r"], dtype=float)
            self.assertLess(Rotation.from_matrix(obj_r @ truth_r.T).magnitude(), 1e-6)
            np.testing.assert_allclose(obj["pose_t"], translation, atol=1e-6)

            # A long capture gap and a replay seek both discard the prior;
            # sparse-frame evaluation must not silently claim temporal support.
            for timestamp in (1.0, 0.0):
                stale = locate_bodies(hidden, np.zeros(3), templates,
                                      tracker=tracker, timestamp=timestamp)[0]
                self.assertEqual(stale["pose_disambiguation"], "unresolved")
                np.testing.assert_allclose(stale["pose_r"], bare_r, atol=1e-8)
        finally:
            directory.cleanup()

    def test_two_four_marker_bodies_never_share_a_marker(self):
        other = copy.deepcopy(self.BODY)
        other["body_id"] = 1
        other["name"] = "quad-B"
        # distinct marker ids and a different extra-marker offset, so the two
        # bodies are recognisable and cannot consume each other's points
        for marker in other["markers"]:
            marker["id"] += 10
        other["markers"][3]["local"] = [0.05, 0.08, 0.40]
        other["apex_marker_id"] = 12
        other["base_marker_ids"] = [10, 11]
        directory, templates = self.templates([self.BODY, other])
        try:
            world = np.vstack([templates[0]["locals"] + [0, 0, 1.0],
                               templates[1]["locals"] + [0, 0, 1.5]])
            objects = locate_bodies(world, np.zeros(8), templates)
            self.assertEqual(len(objects), 2)
            self.assertEqual([obj["droneIndex"] for obj in objects], [0, 1])
            self.assertTrue(all(obj["pose_mode"] == "rigid" for obj in objects))
            self.assertEqual([obj["markers_used"] for obj in objects], [4, 4])
            self.assertTrue(set(objects[0]["marker_ids"]).isdisjoint(
                objects[1]["marker_ids"]))
            # each body consumed its own four points, not the other's
            self.assertAlmostEqual(objects[0]["pose_t"][2], 1.0)
            self.assertAlmostEqual(objects[1]["pose_t"][2], 1.5)
        finally:
            directory.cleanup()

    def test_would_accept_is_read_only_and_timestamps_are_seconds(self):
        tracker = BodyTracker([], gate=0.1, reacquire_gap_s=0.5)
        self.assertTrue(tracker.start_frame(0.0))
        self.assertTrue(tracker.accept({"pos": [0.0, 0.0, 0.0]}, 0))
        tracker.finish_frame({0})

        self.assertTrue(tracker.start_frame(1.0 / 30.0))
        self.assertTrue(tracker.would_accept({"pos": [0.02, 0, 0]}, 0))
        self.assertFalse(tracker.would_accept({"pos": [0.25, 0, 0]}, 0))
        # a rejected probe must leave the committed state untouched
        np.testing.assert_allclose(tracker.state[0]["pos"], [0.0, 0.0, 0.0])
        self.assertEqual(tracker.state[0]["misses"], 0)
        self.assertTrue(tracker.accept({"pos": [0.02, 0, 0]}, 0))
        tracker.finish_frame({0})

        # 0.25 s later the prediction is stale: 4 m/s is not a plausible jump
        self.assertTrue(tracker.start_frame(0.25))
        self.assertFalse(tracker.would_accept({"pos": [1.0, 0, 0]}, 0))
        tracker.finish_frame(set())

        # 0.80 s after the last fix the constant-velocity prediction expires
        self.assertTrue(tracker.start_frame(0.80))
        self.assertIsNone(tracker.predict(0))
        tracker.finish_frame(set())
        self.assertFalse(tracker.start_frame(0.80))
        tracker.finish_frame(set())
        # one loss counted per capture, never twice for the same timestamp
        self.assertEqual(tracker.state[0]["misses"], 2)

    def test_rigid_pose_refuses_a_mirrored_point_set(self):
        local = np.array([[0, 0, 0], [1, 0, 0], [0.1, 0.9, 0], [0.3, 0.2, 0.6]], float)
        mirrored = local * np.array([1.0, 1.0, -1.0])
        rotation, _, rmse = rigid_pose(local, mirrored)
        self.assertGreater(rmse, 0.05)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)

    def test_origin_tracking_does_not_turn_pure_rotation_into_translation(self):
        body = copy.deepcopy(self.BODY)
        for marker in body["markers"]:
            marker["local"] = (np.asarray(marker["local"]) + [.20, .04, .16]).tolist()
        directory, templates = self.templates([body])
        try:
            origin = np.array([.4, -.2, 1.5])
            tracker = BodyTracker(templates, gate=.005)
            kf = KalmanFilter(1, process_noise=1, sampling_frequency=30)
            midpoints = []
            for frame in range(20):
                rotation = Rotation.from_euler("z", frame * 10, degrees=True)
                world = rotation.apply(templates[0]["locals"]) + origin
                points = world[[1, 0, 2]] if frame == 7 else world[[3, 1, 2, 0]]
                found = locate_bodies(points, np.zeros(len(points)), templates, tracker=tracker,
                                      timestamp=frame / 30, position_reference="body_origin")
                self.assertEqual(len(found), 1, f"origin gate rejected pure rotation at frame {frame}")
                obj = found[0]
                midpoints.append(obj["base_midpoint_measured"])
                np.testing.assert_allclose(obj["pos"], origin, atol=1e-9)
                np.testing.assert_allclose(tracker.state[0]["pos"], origin, atol=1e-9)
                filtered = kf.predict_location(found, timestamp=frame / 30)[0]
                np.testing.assert_allclose(filtered["pos"], origin, atol=1e-6)
                np.testing.assert_allclose(filtered["vel"], np.zeros(3), atol=1e-6)
                np.testing.assert_allclose(filtered["acc"], np.zeros(3), atol=1e-6)
                self.assertEqual(kf.kalmans[0].statePre.shape, (9, 1))
            self.assertGreater(np.linalg.norm(midpoints[0] - midpoints[-1]), .3)
            # Stale/cold three-point branches cannot become a fake origin fix.
            self.assertEqual(locate_bodies(world[:3], np.zeros(3), templates, tracker=tracker,
                                          timestamp=2.0, position_reference="body_origin"), [])
            self.assertEqual(locate_bodies(world[:3], np.zeros(3), templates,
                                          position_reference="body_origin"), [])
        finally:
            directory.cleanup()


class SourceBoundaryTests(unittest.TestCase):
    def _server_camera(self, directory):
        calibration = Path(directory) / "calibration.json"
        poses = cameras()
        calibration.write_text(json.dumps({"cameras": [
            {"name": f"CAM{i + 1:02d}", "K": p["K"].tolist(),
             "R": p["R"].tolist(), "t": p["t"].ravel().tolist()}
            for i, p in enumerate(poses)]}), encoding="utf-8")
        source = mock.Mock()
        source.exposure = [100] * len(poses)
        source.read.return_value = ([np.zeros((32, 32, 3), np.uint8) for _ in poses],
                                    np.full(len(poses), 1 / 30))
        with mock.patch.dict(os.environ, {"MOCAP_CALIBRATION": str(calibration),
                                         "MOCAP_VISIBLE_MODE": "1"}), \
                mock.patch.object(helpers, "Camera", return_value=source):
            camera = helpers.Cameras._decorated()
        return camera, source, poses

    def test_empty_server_capture_publishes_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            camera, _, poses = self._server_camera(directory)
            rig = Path(directory) / "rig.json"
            other = copy.deepcopy(RigidBodyExtensionTests.BODY)
            other["body_id"] = 1
            rig.write_text(json.dumps({"bodies": [RigidBodyExtensionTests.BODY, other]}),
                           encoding="utf-8")
            with mock.patch.dict(os.environ, {"MOCAP_BODY_SPECS": str(rig)}):
                camera.set_num_objects(2)
            camera.set_socketio(mock.Mock())
            camera.start_trangulating_points(poses)
            camera.start_locating_objects()
            camera.body_tracker.start_frame(0)
            camera.body_tracker.accept({"pos": [0, 0, 1]}, 0)
            camera.body_tracker.finish_frame({0})
            camera._camera_read()
            event, payload = camera.socketio.emit.call_args.args
            self.assertEqual(event, "object-points")
            self.assertEqual(payload["objects"], [])
            self.assertEqual(payload["filtered_objects"], [])
            self.assertEqual(payload["tracking"]["bodies"][0]["misses"], 1)

    def test_multiple_viewers_share_one_decoded_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            camera, source, _ = self._server_camera(directory)
            with mock.patch.object(helpers.time, "monotonic", return_value=1.0):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    frames = list(pool.map(lambda _: camera.get_jpeg(), range(2)))
            self.assertEqual(source.read.call_count, 1)
            self.assertIs(frames[0], frames[1])
            image = cv.imdecode(np.frombuffer(frames[0], np.uint8), cv.IMREAD_COLOR)
            self.assertEqual(image.shape, (424, 1920, 3))

    def test_server_and_pipeline_filter_the_same_origin_and_serialize_pva(self):
        with tempfile.TemporaryDirectory() as directory:
            camera, _, poses = self._server_camera(directory)
            body = copy.deepcopy(RigidBodyExtensionTests.BODY)
            body["body_id"] = 7  # wire IDs need not be Kalman's contiguous slots
            for marker in body["markers"]:
                marker["local"] = (np.asarray(marker["local"]) + [.2, 0, .1]).tolist()
            rig = Path(directory) / "rig.json"
            rig.write_text(json.dumps({"bodies": [body]}), encoding="utf-8")
            with mock.patch.dict(os.environ, {"MOCAP_BODY_SPECS": str(rig)}):
                camera.set_num_objects(1)
            camera.set_socketio(mock.Mock())
            camera.start_trangulating_points(poses)
            camera.start_locating_objects()
            pipeline = TrackingPipeline(Path(directory) / "calibration.json", rig)
            origin = np.array([.4, -.2, 1.5])
            world = Rotation.from_euler("z", 60, degrees=True).apply(pipeline.templates[0]["locals"]) + origin

            def triangulation(image_points, camera_poses, frames=None):
                return np.zeros(4), world.copy(), frames

            with mock.patch.object(helpers, "find_point_correspondance_and_object_points", side_effect=triangulation), \
                    mock.patch("tracking_pipeline.find_point_correspondance_and_object_points", side_effect=triangulation):
                camera._camera_read()
                offline = pipeline.process_detections([[], [], []], timestamp=1 / 30)
            event, payload = camera.socketio.emit.call_args.args
            self.assertEqual(event, "object-points")
            json.dumps(payload, allow_nan=False)
            actual = payload["filtered_objects"][0]
            self.assertEqual(actual["droneIndex"], 7)
            self.assertEqual(actual["position_reference"], "body_origin")
            self.assertEqual(payload["tracking"]["position_reference"], "body_origin")
            np.testing.assert_allclose(actual["pos"], origin, atol=1e-6)
            for key in ("pos", "vel", "acc"):
                np.testing.assert_allclose(actual[key], offline["filtered_objects"][0][key], atol=1e-7)

    def test_discovery_does_not_mix_datasets_or_hud(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in range(1, 7):
                for tag in ("visible", "t650"):
                    for suffix in ("", "_hud"):
                        (Path(directory) / f"{index:02d}_cam{index:02d}_{tag}_20s{suffix}.mp4").touch()
            self.assertEqual(len(discover_clips(directory, "t650")), 6)
            self.assertTrue(all("t650_20s.mp4" in p for p in discover_clips(directory, "t650")))

    def test_file_timestamps_start_at_zero(self):
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda key: 30 if key == cv.CAP_PROP_FPS else 600
        capture.read.return_value = (True, np.zeros((4, 4, 3), np.uint8))
        with mock.patch("video_replay.cv.VideoCapture", return_value=capture):
            reader = _VideoFileReader("test.mp4", loop=False)
            reader.read()
            self.assertEqual(reader.timestamp(), 0)
            reader.read()
            self.assertAlmostEqual(reader.timestamp(), 1 / 30)

    def test_nonlooping_replay_raises_eof_instead_of_stale_frame(self):
        reader = mock.Mock(kind="file", fps=30, frame_count=1)
        reader.read.side_effect = [np.zeros((4, 4, 3), np.uint8), None]
        reader.timestamp.return_value = 0
        with mock.patch("video_replay.open_reader", return_value=reader):
            source = VideoReplay(source="test.mp4", realtime=False, loop=False)
            source.read()
            with self.assertRaises(StopIteration):
                source.read()

    def test_failed_stream_frame_has_no_pixels_and_nan_timestamp(self):
        reader = mock.Mock(kind="stream", fps=30)
        reader.read.return_value = None
        with mock.patch("video_replay.open_reader", return_value=reader):
            source = VideoReplay(source="rtsp://example.invalid/camera", realtime=False)
            frames, timestamps = source.read()
        self.assertIsNone(frames[0])
        self.assertTrue(np.isnan(timestamps[0]))


if __name__ == "__main__":
    unittest.main()
