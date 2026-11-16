"""Reusable Low-Cost-Mocap analysis, independent of video sources and truth.

Files, PNG sequences, RTSP and device readers all supply the same ``frames``
list.  This module never reads a trajectory or detection-truth CSV.  Numeric
matching/DLT and Kalman filtering are delegated to the vendored repository.
"""
from __future__ import annotations

import json
import hashlib
import os
import sys

import numpy as np

ADAPTER = os.path.dirname(os.path.abspath(__file__))
API = os.path.join(ADAPTER, "low-cost-mocap", "computer_code", "api")
if API not in sys.path:
    sys.path.insert(0, API)

from helpers import (detect_visible_dots, prepare_camera_frame,
                     find_point_correspondance_and_object_points)
from KalmanFilter import KalmanFilter
from multi_body import (BodyTracker, load_body_templates, locate_bodies,
                        POSITION_REFERENCES, POSITION_REFERENCE_LABELS)


def configured_calibration_path():
    """Shared server/UI calibration, resolved relative to the project root."""
    root = os.path.abspath(os.path.join(ADAPTER, "..", ".."))
    path = os.environ.get("MOCAP_CALIBRATION", "out/rig3/cameras_calib.json")
    return path if os.path.isabs(path) else os.path.join(root, path)


def load_calibration(path):
    with open(path, encoding="utf-8") as handle:
        calibration = json.load(handle)
    cameras = calibration["cameras"]
    if len(cameras) < 2 or len({c["name"] for c in cameras}) != len(cameras):
        raise ValueError("calibration needs at least two uniquely named cameras")
    params, poses = [], []
    for camera in cameras:
        K = np.asarray(camera["K"], dtype=float)
        R = np.asarray(camera["R"], dtype=float)
        t = np.asarray(camera["t"], dtype=float).reshape(3, 1)
        if K.shape != (3, 3) or R.shape != (3, 3):
            raise ValueError(f"invalid calibration matrix for {camera['name']}")
        if not all(np.all(np.isfinite(v)) for v in (K, R, t)):
            raise ValueError(f"non-finite calibration for {camera['name']}")
        params.append({"intrinsic_matrix": K,
                       "distortion_coef": np.asarray(camera.get("dist", [0] * 5)),
                       "rotation": int(camera.get("rotation", 0))})
        poses.append({"R": R, "t": t, "K": K})
    return calibration, params, poses


def project(camera, point):
    p = np.asarray(point, dtype=float).reshape(3, 1)
    uvw = (camera["K"] @ (camera["R"] @ p + camera["t"])).ravel()
    if not np.all(np.isfinite(uvw)) or uvw[2] <= 1e-9:
        return None
    return (float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]))


class TrackingPipeline:
    def __init__(self, calibration_path, body_specs_path, threshold=140.0,
                 min_area=3, max_area=200, blur_size=9, tolerance=0.025,
                 fps=30.0, process_noise=1.0, measurement_noise=1.0,
                  reacquire_gap_s=0.5, gate_m=0.5, max_err=6.0, extra_tol=0.05,
                  position_reference="body_origin", max_speed=10.0, centroid_mode="integer"):
        if blur_size < 1 or blur_size % 2 != 1:
            raise ValueError("blur_size must be a positive odd integer")
        if centroid_mode not in ("integer", "subpixel"):
            raise ValueError("centroid_mode must be integer or subpixel")
        self.centroid_mode = centroid_mode
        self.calibration, self.params, self.poses = load_calibration(calibration_path)
        self.camera_names = [camera["name"] for camera in self.calibration["cameras"]]
        self.templates = load_body_templates(body_specs_path)
        self.body_ids = [template["body_id"] for template in self.templates]
        self.slots = {body_id: i for i, body_id in enumerate(self.body_ids)}
        self.threshold = float(threshold)
        self.min_area, self.max_area = min_area, max_area
        self.blur_size, self.tolerance = blur_size, float(tolerance)
        self.fps = float(fps)
        self.max_err = None if max_err is None else float(max_err)
        self.extra_tol = float(extra_tol)
        if position_reference not in POSITION_REFERENCES:
            raise ValueError(f"unknown position_reference: {position_reference}")
        self.position_reference = position_reference
        self.max_speed = float(max_speed)
        self.tracker = (BodyTracker(self.templates, gate=float(gate_m), reacquire_gap_s=reacquire_gap_s)
                        if gate_m is not None and gate_m > 0 else None)
        self.configuration = {
            "threshold": self.threshold, "blur_size": blur_size,
            "min_area": min_area, "max_area": max_area, "tolerance_m": self.tolerance,
            "fps": self.fps, "motion_gate_m": gate_m if self.tracker else None,
            "mean_reprojection_mse_limit_px2": self.max_err,
            "extra_marker_tolerance_m": self.extra_tol,
            "reacquire_gap_s": reacquire_gap_s, "process_noise": process_noise,
            "measurement_noise": measurement_noise}
        self.configuration.update({
            "position_reference": position_reference,
            "position_reference_description": POSITION_REFERENCE_LABELS[position_reference],
            "max_speed_mps": self.max_speed,
            "kalman_state": ["px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "az"],
             "kalman_output": "statePre; velocity additionally Butterworth-filtered"})
        # Keep the historical integer configuration unchanged for old cached
        # regression inputs. Nondefault detection mode is always recorded.
        if centroid_mode != "integer":
            self.configuration["centroid_mode"] = centroid_mode
        self.calibration_path = os.path.abspath(calibration_path)
        self.body_specs_path = os.path.abspath(body_specs_path)
        self.geometry_sha256 = {}
        for key, path in (("calibration", self.calibration_path), ("body_specs", self.body_specs_path)):
            with open(path, "rb") as handle:
                self.geometry_sha256[key] = hashlib.sha256(handle.read()).hexdigest()
        self._frame = 0
        self.kalman = KalmanFilter(len(self.templates), process_noise,
                                   measurement_noise, reacquire_gap_s,
                                   sampling_frequency=self.fps,
                                   max_speed=self.max_speed)

    def process(self, frames, timestamp):
        if len(frames) != len(self.poses):
            raise ValueError("frame/camera count does not match calibration")
        image_points, valid = [], []
        for image, params in zip(frames, self.params):
            valid.append(image is not None)
            if image is None:
                image_points.append([])
                continue
            prepared = prepare_camera_frame(image, params, True, self.blur_size)
            image_points.append(detect_visible_dots(
                prepared, self.threshold, self.min_area, self.max_area, self.centroid_mode))
        return self.process_detections(image_points, timestamp, valid)

    def process_detections(self, image_points, timestamp, valid=None):
        """Also usable for ideal geometry tests, explicitly without pixels."""
        errors, points, _ = find_point_correspondance_and_object_points(
            image_points, self.poses, frames=None)
        objects = locate_bodies(points, errors, self.templates,
                                tracker=self.tracker, tol=self.tolerance,
                                frame=self._frame, dt=1.0 / self.fps,
                                max_err=self.max_err, timestamp=float(timestamp),
                                 extra_tol=self.extra_tol, position_reference=self.position_reference)
        self._frame += 1
        # Repository Kalman indices are contiguous; external body IDs need
        # not be.  Translate only at this boundary, not in the algorithm.
        measurements = [dict(obj, droneIndex=self.slots[obj["droneIndex"]])
                        for obj in objects]
        filtered = self.kalman.predict_location(measurements, timestamp=timestamp)
        for obj in filtered:
            obj["droneIndex"] = self.body_ids[obj["droneIndex"]]
        return {"t": float(timestamp), "position_reference": self.position_reference,
                "detections": image_points,
                "valid_cameras": valid if valid is not None else [True] * len(self.poses),
                "points": points, "reprojection_mse_px2": errors,
                "objects": objects, "filtered_objects": filtered}

    def reset(self):
        self._frame = 0
        if self.tracker is not None:
            self.tracker.reset()
        self.kalman.reset()


def serializable_result(result):
    """Convert a runtime result to a JSON record (no synthetic truth added)."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: convert(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value
    return convert(result)
