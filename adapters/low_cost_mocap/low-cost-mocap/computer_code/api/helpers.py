import numpy as np
from scipy import linalg, optimize, signal
import cv2 as cv
from scipy.spatial.transform import Rotation
import copy
import json
import os
import time
import threading
import numpy as np
import cv2 as cv
from KalmanFilter import KalmanFilter
# Video source: replay the rendered OptiTrack-style clips (no real cameras).
# VideoReplay is a pseyepy-compatible drop-in: exposes .read() returning a
# (frames, timestamps) pair, a .num_cameras attribute and mutable
# .gain / .exposure lists, plus a RES_SMALL tag.  It reads the 6 synchronised
# OV9281 clips and paces them out at source framerate so the pipeline sees a
# live multi-camera stream.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from video_replay import VideoReplay as Camera, default_project_root
from Singleton import Singleton

# Multi-target localisation: one rigid body per drone, told apart by the
# pairwise marker distances in its body spec (see adapters/low_cost_mocap/
# multi_body.py).  Single-body runs keep using the upstream locate_objects().
try:
    from multi_body import (load_body_templates, locate_bodies, BodyTracker,
                            default_rig_path, POSITION_REFERENCES, POSITION_REFERENCE_LABELS)
    _MULTI_OK = True
except Exception as _e:            # pragma: no cover - optional accelerator
    _MULTI_OK = False
    _MULTI_ERR = _e


# --- adapter: pure-OpenCV fallback for the OpenCV-contrib "sfm" module -------
# pip-distributed opencv-python does not ship cv2.sfm (it requires building
# OpenCV from source).  We only need one helper, fundamentalFromProjections,
# used by the live-triangulation epipolar matching.  Closed form:
#     F = skew(e') @ P' @ pinv(P),   e' = P' @ C,  C = nullspace(P)
# (identical to the upstream cv::sfm implementation).  This is used only when
# cv.sfm is unavailable; otherwise the original call is left untouched.
def _skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float64)


def fundamental_from_projections(P1, P2):
    U, S, Vt = np.linalg.svd(P1)
    C = Vt[-1]                       # right null-space vector (centre of P1)
    e2 = P2 @ C                      # epipole in camera 2
    F = _skew(e2) @ P2 @ np.linalg.pinv(P1)
    if abs(F[2, 2]) > 1e-12:         # normalise to OpenCV's F[2,2]=1 convention
        F = F / F[2, 2]
    return F


if hasattr(cv, "sfm"):
    _fundamental_from_projections = lambda P1, P2: cv.sfm.fundamentalFromProjections(P1, P2)
else:
    _fundamental_from_projections = fundamental_from_projections


def prepare_camera_frame(image, params, visible_mode=True, blur_size=9):
    """Native calibrated pixels; do not square-pad/resize the analysis image.

    Padding an 800-row image before using cy=400 shifts every detection by
    240 pixels.  Formatting belongs in the viewer, not the calibrated path.
    """
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    image = np.ascontiguousarray(np.rot90(image, k=params.get("rotation", 0)))
    K = np.asarray(params["intrinsic_matrix"], dtype=float)
    distortion = np.asarray(params.get("distortion_coef", [0] * 5), dtype=float)
    if np.any(distortion):
        image = cv.undistort(image, K, distortion)
    if blur_size > 1:
        image = cv.GaussianBlur(image, (blur_size, blur_size), 0)
    if not visible_mode:
        kernel = np.array([[-2,-1,-1,-1,-2], [-1,1,3,1,-1],
                           [-1,3,4,3,-1], [-1,1,3,1,-1], [-2,-1,-1,-1,-2]])
        image = cv.filter2D(image, -1, kernel)
    return image


def detect_visible_dots(image, threshold=150, min_area=3, max_area=200,
                        centroid_mode="integer"):
    """Visible dots; upstream integer mode or unrounded component centroids.

    Subpixel mode preserves the connected-component centroid's floating-point
    value. It is not a fitted optical spot centre or a truth-derived offset.
    """
    if centroid_mode not in ("integer", "subpixel"):
        raise ValueError("centroid_mode must be integer or subpixel")
    gray = image if image.ndim == 2 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    count, _, stats, centers = cv.connectedComponentsWithStats(mask, 8)
    selected = [i for i in range(1, count)
                if min_area <= stats[i, cv.CC_STAT_AREA] <= max_area]
    if centroid_mode == "subpixel":
        return centers[selected].tolist()
    return [[int(centers[i, 0]), int(centers[i, 1])] for i in selected]


def _intrinsic_for_pose(pose, camera_index):
    # An explicit calibration makes these numeric helpers usable without
    # opening a capture device or a hidden global video replay.
    if "K" in pose:
        return np.asarray(pose["K"], dtype=float)
    return Cameras.instance().get_camera_params(camera_index)["intrinsic_matrix"]


def _projection_for_pose(pose, camera_index):
    return _intrinsic_for_pose(pose, camera_index) @ np.c_[pose["R"], pose["t"]]


def _valid_image_point(point):
    return len(point) == 2 and all(x is not None and np.isfinite(x) for x in point)


@Singleton
class Cameras:
    def __init__(self):
        # Import after this module has loaded: TrackingPipeline reuses the
        # numeric helpers above. The server and offline evaluator read the
        # same current calibration rather than an old camera-params copy.
        from tracking_pipeline import configured_calibration_path, load_calibration
        self.calibration, self.camera_params, self.configured_poses = load_calibration(
            configured_calibration_path())
        self.camera_names = [c["name"] for c in self.calibration["cameras"]]

        # Visible-light mode (default off = upstream IR/high-pass behaviour).
        # Our rendered clips are bright 20 mm markers over a grey floor, so the
        # IR edge-detection chain floods the detector with floor texture.  When
        # enabled we skip the high-pass and detect by absolute brightness.
        self.visible_mode = bool(int(os.environ.get("MOCAP_VISIBLE_MODE", "0")))

        self.cameras = Camera(fps=int(os.environ.get("MOCAP_SOURCE_FPS", "30")),
                              resolution=Camera.RES_SMALL, gain=10, exposure=100)
        self.num_cameras = len(self.cameras.exposure)
        if self.num_cameras != len(self.camera_params):
            self.cameras.release()
            raise ValueError("source count does not match the configured calibration")
        print(self.num_cameras)

        self.is_capturing_points = False

        self.is_triangulating_points = False
        self.camera_poses = None

        self.is_locating_objects = False

        self.to_world_coords_matrix = None

        self.drone_armed = []

        self.num_objects = None

        # multi-body state
        self.body_templates = None
        self.body_tracker = None
        self.position_reference = os.environ.get("MOCAP_POSITION_REFERENCE", "body_origin")
        if self.position_reference not in POSITION_REFERENCES:
            raise ValueError(f"unknown position reference: {self.position_reference}")
        self.body_slots = {}
        self.body_ids = []
        self.body_max_err = 6.0
        self.body_extra_tol = 0.05
        self.frame_index = 0
        self.last_source_status = {}
        self.last_detection_counts = [0] * self.num_cameras
        self._capture_lock = threading.RLock()
        self._last_mosaic = None
        self._last_capture_started = float("-inf")
        self._last_jpeg = None

        self.kalman_filter = None

        self.socketio = None
        self.ser = None

        self.serialLock = None

        global cameras_init
        cameras_init = True

    def set_socketio(self, socketio):
        self.socketio = socketio
    
    def set_ser(self, ser):
        self.ser = ser

    def set_serialLock(self, serialLock):
        self.serialLock = serialLock

    def set_num_objects(self, num_objects):
        if self.num_objects == num_objects:
            return  # another viewer must not reset a running track
        self.num_objects = num_objects
        self.drone_armed = [False for i in range(0, self.num_objects)]

        # n > 1 means the scene holds several rigid bodies: load their marker
        # layouts once, here, so the hot loop stays allocation free.
        self.body_templates = None
        self.body_tracker = None
        use_templates = num_objects > 1 or self.position_reference == "body_origin"
        if use_templates and not _MULTI_OK:
            raise RuntimeError("multi-body adapter could not be imported") from _MULTI_ERR
        if use_templates:
            path = (os.environ.get("MOCAP_BODY_SPECS")
                    or os.environ.get("MOCAP_RIG_JSON")
                    or default_rig_path())
            if not os.path.isabs(path):
                path = os.path.join(default_project_root(), path)
            if os.path.exists(path):
                self.body_templates = load_body_templates(path)
                if len(self.body_templates) != num_objects:
                    raise ValueError("body template count does not match num_objects")
                self.body_ids = [t["body_id"] for t in self.body_templates]
                self.body_slots = {body_id: slot for slot, body_id in enumerate(self.body_ids)}
                gate = float(os.environ.get("MOCAP_BODY_GATE_M", "0.5"))
                self.body_extra_tol = float(os.environ.get("MOCAP_BODY_EXTRA_TOL", "0.05"))
                self.body_max_err = os.environ.get("MOCAP_BODY_MAX_ERR", "6.0") or None
                self.body_max_err = None if self.body_max_err is None else float(self.body_max_err)
                # Temporal association: a timestamped constant-velocity gate
                # per body.  It is optional (gate <= 0 disables it) and only
                # rejects physically impossible jumps; it never uses truth.
                self.body_tracker = (BodyTracker(
                    self.body_templates, gate=gate,
                    reacquire_gap_s=float(os.environ.get("MOCAP_REACQUIRE_GAP_S", "0.5")))
                    if gate > 0 else None)
                print("[multi] bodies:",
                      ", ".join(f"{t['body_id']}:{t['name']}"
                                f"({t['frame']}, {len(t['marker_ids'])} markers)"
                                for t in self.body_templates))
                print("[multi] temporal gate:",
                      "off" if self.body_tracker is None else f"{gate:.2f} m")
            else:
                raise FileNotFoundError(f"[multi] body spec not found: {path}")
    
    def edit_settings(self, exposure, gain):
        self.cameras.exposure = [exposure] * self.num_cameras
        self.cameras.gain = [gain] * self.num_cameras

    def _camera_read(self):
        frames, timestamps = self.cameras.read()
        self.frame_index += 1
        valid = [image is not None for image in frames]
        finite_stamps = np.asarray(timestamps)[np.isfinite(timestamps)]
        self.last_source_status = {"valid_cameras": valid,
                                    "timestamp": (float(np.median(finite_stamps)) if finite_stamps.size else None),
                                    "max_skew_s": (float(np.ptp(finite_stamps)) if finite_stamps.size else None)}

        for i in range(0, self.num_cameras):
            if valid[i]:
                frames[i] = prepare_camera_frame(
                    frames[i], self.get_camera_params(i), self.visible_mode,
                    int(os.environ.get("MOCAP_BLUR_SIZE", "9")))
            else:
                frames[i] = np.zeros((Camera.RES_SMALL[1], Camera.RES_SMALL[0], 3), np.uint8)

        if (self.is_capturing_points):
            image_points = []
            for i in range(0, self.num_cameras):
                if valid[i]:
                    frames[i], single_camera_image_points = self._find_dot(frames[i])
                else:
                    single_camera_image_points = [[None, None]]
                image_points.append(single_camera_image_points)
            self.last_detection_counts = [sum(_valid_image_point(p) for p in points)
                                          for points in image_points]
             
            # Empty captures still advance the tracker and publish loss state.
            if self.is_triangulating_points or any(self.last_detection_counts):
                if self.is_capturing_points and not self.is_triangulating_points:
                    self.socketio.emit("image-points", [x[0] for x in image_points])
                elif self.is_triangulating_points:
                    errors, object_points, frames = find_point_correspondance_and_object_points(image_points, self.camera_poses, frames)

                    # convert to world coordinates
                    for i, object_point in enumerate(object_points):
                        if self.to_world_coords_matrix is None:
                            continue  # calibrated R/t already define the output world
                        new_object_point = np.array([[-1,0,0],[0,-1,0],[0,0,1]]) @ object_point
                        new_object_point = np.concatenate((new_object_point, [1]))
                        new_object_point = np.array(self.to_world_coords_matrix) @ new_object_point
                        new_object_point = new_object_point[:3] / new_object_point[3]
                        new_object_point[1], new_object_point[2] = new_object_point[2], new_object_point[1]
                        object_points[i] = new_object_point

                    objects = []
                    filtered_objects = []
                    tracking = None
                    if self.is_locating_objects:
                        stamp = (float(np.median(finite_stamps)) if finite_stamps.size
                                 else self.frame_index / float(os.environ.get("MOCAP_SOURCE_FPS", "30")))
                        if self.body_templates:
                            objects = locate_bodies(
                                object_points, errors, self.body_templates,
                                tracker=self.body_tracker, timestamp=stamp,
                                tol=float(os.environ.get("MOCAP_BODY_TOL", 0.025)),
                                max_err=self.body_max_err,
                                extra_tol=self.body_extra_tol, position_reference=self.position_reference)
                        else:
                            objects = locate_objects(object_points, errors)
                        measurements = ([dict(obj, droneIndex=self.body_slots[obj["droneIndex"]])
                                         for obj in objects] if self.body_templates else objects)
                        filtered_objects = self.kalman_filter.predict_location(measurements, timestamp=stamp)
                        if self.body_templates:
                            for obj in filtered_objects:
                                obj["droneIndex"] = self.body_ids[obj["droneIndex"]]
                        tracking = self._tracking_payload(objects, stamp, filtered_objects)
                        tracking["source_status"] = self.last_source_status
                        
                        if len(filtered_objects) != 0:
                            for filtered_object in filtered_objects:
                                slot = self.body_slots.get(filtered_object['droneIndex'], filtered_object['droneIndex'])
                                if self.drone_armed[slot]:
                                    filtered_object["heading"] = round(filtered_object["heading"], 4)

                                    serial_data = { 
                                        "pos": [round(x, 4) for x in filtered_object["pos"].tolist()] + [filtered_object["heading"]],
                                        "vel": [round(x, 4) for x in filtered_object["vel"].tolist()]
                                    }
                                    with self.serialLock:
                                        self.ser.write(f"{filtered_object['droneIndex']}{json.dumps(serial_data)}".encode('utf-8'))
                                        time.sleep(0.001)
                            
                    self.socketio.emit("object-points", _jsonable({
                        "object_points": object_points.tolist(), 
                        "errors": errors.tolist(), 
                        "objects": [_jsonable(object) for object in objects], 
                        "filtered_objects": filtered_objects,
                        "tracking": tracking
                    }))
        
        return frames

    def _tracking_payload(self, objects, timestamp, filtered_objects=()):
        """Operator-facing identity/pose state; geometry only, no truth."""
        found = {obj["droneIndex"]: obj for obj in objects}
        filtered = {obj["droneIndex"]: obj for obj in filtered_objects}
        bodies = []
        for tpl in (self.body_templates or []):
            body_id = tpl["body_id"]
            obj = found.get(body_id)
            state = (self.body_tracker.status(body_id) if self.body_tracker is not None
                     else {"state": "locked" if obj is not None else "lost",
                           "misses": 0, "age_s": None})
            entry = {
                "body_id": body_id, "name": tpl["name"], "frame": tpl["frame"],
                "marker_ids": tpl["marker_ids"],
                "marker_count": len(tpl["marker_ids"]),
                "reference_ids": [tpl["marker_ids"][tpl["apex"]],
                                   tpl["marker_ids"][tpl["base_pair"][0]],
                                   tpl["marker_ids"][tpl["base_pair"][1]]],
                "extra_ids": [tpl["marker_ids"][i] for i in tpl["extras"]],
                "non_coplanar": bool(tpl["rigidity_margin_m"] > 0.005),
                "state": state["state"], "misses": state["misses"],
                "age_s": state["age_s"],
                "markers_used": 0, "pose_mode": None,
            }
            if obj is not None:
                for key in ("markers_used", "pose_mode", "pose_rmse_m", "pose_tilt_deg", "pose_disambiguation",
                            "pose_euler_zyx_deg", "pose_ambiguous"):
                    if key in obj:
                        entry[key] = obj[key]
                if "marker_ids" in obj:
                    entry["used_marker_ids"] = obj["marker_ids"]
            estimate = filtered.get(body_id)
            if estimate is not None:
                for key in ("pos", "vel", "acc"):
                    if key in estimate:
                        entry[key] = estimate[key]
            bodies.append(entry)
        return {"timestamp": float(timestamp), "bodies": bodies,
                "position_reference": self.position_reference,
                "position_reference_description": POSITION_REFERENCE_LABELS[self.position_reference],
                "kalman_state": ["px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "az"],
                "motion_gate_m": getattr(self.body_tracker, "gate", None),
                "extra_tolerance_m": self.body_extra_tol,
                "max_reprojection_mse_px2": self.body_max_err}

    def get_frames(self):
        # VideoCapture/FFmpeg must never be entered by two HTTP clients at
        # once. Share each processed capture instead of consuming alternate
        # source frames when a second viewer connects or refreshes.
        with self._capture_lock:
            now = time.monotonic()
            interval = 1.0 / float(os.environ.get("MOCAP_SOURCE_FPS", "30"))
            if self._last_mosaic is not None and now - self._last_capture_started < interval:
                return self._last_mosaic
            self._last_mosaic = self._capture_mosaic()
            self._last_capture_started = now
            self._last_jpeg = None
            return self._last_mosaic

    def get_jpeg(self):
        with self._capture_lock:
            frames = self.get_frames()
            if self._last_jpeg is None:
                ok, encoded = cv.imencode('.jpg', frames)
                if not ok:
                    raise RuntimeError("MJPEG encoding failed")
                self._last_jpeg = encoded.tobytes()
            return self._last_jpeg

    def _capture_mosaic(self):
        frames = self._camera_read()
        # Presentation resizing happens only after native-pixel detection.
        columns = min(3, self.num_cameras)
        rows = (self.num_cameras + columns - 1) // columns
        width, height, header = 640, 400, 24
        mosaic = np.zeros((rows * (height + header), columns * width, 3), np.uint8)
        valid = self.last_source_status.get("valid_cameras", [True] * self.num_cameras)
        for i, frame in enumerate(frames):
            x, y = (i % columns) * width, (i // columns) * (height + header)
            tile = mosaic[y:y + height + header, x:x + width]
            tile[:header] = (31, 40, 54)
            tile[header:] = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
            label = self.camera_names[i]
            if not valid[i]:
                label += " | NO FRAME"
            elif self.is_capturing_points:
                label += f" | dots {self.last_detection_counts[i]}"
            stamp = self.last_source_status.get("timestamp")
            if stamp is not None:
                label += f" | t={stamp:.3f}s"
            cv.putText(tile, label, (10, 17), cv.FONT_HERSHEY_SIMPLEX,
                       0.48, (235, 235, 235), 1, cv.LINE_AA)
        return mosaic

    def _find_dot(self, img):
        # img = cv.GaussianBlur(img,(5,5),0)
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if self.visible_mode:
            # Brightness detector: markers are bright (>= ~150) over a grey
            # floor (~62).  Threshold high so floor texture never passes, then
            # keep only blobs of a plausible marker size (a 20 mm marker
            # subtends a few px at ~8 m; the MJPEG mosaic also feathers edges).
            thresh = float(os.environ.get("MOCAP_DOT_THRESHOLD", 150))
            min_area = float(os.environ.get("MOCAP_DOT_MIN_AREA", 3))
            max_area = float(os.environ.get("MOCAP_DOT_MAX_AREA", 200))
            image_points = detect_visible_dots(
                img, thresh, min_area, max_area,
                centroid_mode=os.environ.get("MOCAP_CENTROID_MODE", "integer"))
            for cx, cy in image_points:
                # Only drawing rounds the pixel location; the analysis points
                # stay unrounded all the way to multiview fusion.
                drawing_point = (int(round(cx)), int(round(cy)))
                cv.putText(img, f'({cx}, {cy})', (drawing_point[0], drawing_point[1] - 15),
                           cv.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
                cv.circle(img, drawing_point, 1, (100,255,100), -1)
        else:
            # Upstream IR detector: threshold the high-pass response.
            thresh = float(os.environ.get("MOCAP_DOT_THRESHOLD", 255 * 0.2))
            grey = cv.threshold(grey, thresh, 255, cv.THRESH_BINARY)[1]
            contours, _ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            img = cv.drawContours(img, contours, -1, (0,255,0), 1)
            image_points = []
            min_area = float(os.environ.get("MOCAP_DOT_MIN_AREA", 0))
            for contour in contours:
                moments = cv.moments(contour)
                if moments["m00"] != 0:
                    if moments["m00"] < min_area:
                        continue
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                    cv.putText(img, f'({center_x}, {center_y})', (center_x,center_y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
                    cv.circle(img, (center_x,center_y), 1, (100,255,100), -1)
                    image_points.append([center_x, center_y])

        if len(image_points) == 0:
            image_points = [[None, None]]

        return img, image_points

    def start_capturing_points(self):
        self.is_capturing_points = True

    def stop_capturing_points(self):
        self.is_capturing_points = False

    def start_trangulating_points(self, camera_poses):
        if len(camera_poses) != self.num_cameras:
            raise ValueError("camera pose count does not match the input streams")
        self.is_capturing_points = True
        self.is_triangulating_points = True
        self.camera_poses = camera_poses
        self.kalman_filter = KalmanFilter(
            self.num_objects, process_noise=float(os.environ.get("MOCAP_KALMAN_Q", "1.0")),
            measurement_noise=float(os.environ.get("MOCAP_KALMAN_R", "1.0")),
            reacquire_gap_s=float(os.environ.get("MOCAP_REACQUIRE_GAP_S", "0.5")),
            sampling_frequency=float(os.environ.get("MOCAP_SOURCE_FPS", "30")),
            max_speed=float(os.environ.get("MOCAP_MAX_SPEED_MPS", "10.0")))
        if self.body_tracker is not None:
            self.body_tracker.reset()

    def stop_trangulating_points(self):
        self.is_capturing_points = False
        self.is_triangulating_points = False
        self.camera_poses = None

    def start_locating_objects(self):
        self.is_locating_objects = True

    def stop_locating_objects(self):
        self.is_locating_objects = False
    
    def get_camera_params(self, camera_num):
        return {
            "intrinsic_matrix": np.array(self.camera_params[camera_num]["intrinsic_matrix"]),
            "distortion_coef": np.array(self.camera_params[camera_num]["distortion_coef"]),
            "rotation": self.camera_params[camera_num]["rotation"]
        }
    
    def set_camera_params(self, camera_num, intrinsic_matrix=None, distortion_coef=None):
        if intrinsic_matrix is not None:
            self.camera_params[camera_num]["intrinsic_matrix"] = intrinsic_matrix
        
        if distortion_coef is not None:
            self.camera_params[camera_num]["distortion_coef"] = distortion_coef


def _jsonable(value):
    """Convert nested runtime results to JSON-safe lists and scalars."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def calculate_reprojection_errors(image_points, object_points, camera_poses):
    errors = np.array([])
    for image_points_i, object_point in zip(image_points, object_points):
        error = calculate_reprojection_error(image_points_i, object_point, camera_poses)
        if error is None:
            continue
        errors = np.concatenate([errors, [error]])

    return errors


def calculate_reprojection_error(image_points, object_point, camera_poses):
    valid = [(i, p, pose) for i, (p, pose) in enumerate(zip(image_points, camera_poses))
             if _valid_image_point(p)]
    if len(valid) <= 1:
        return None
    errors = []
    for i, point, camera_pose in valid:
        projected_img_points, _ = cv.projectPoints(
            np.expand_dims(object_point, axis=0).astype(np.float32), 
            np.array(camera_pose["R"], dtype=np.float64), 
            np.array(camera_pose["t"], dtype=np.float64), 
            _intrinsic_for_pose(camera_pose, i),
            np.array([])
        )
        projected_img_point = projected_img_points[:,0,:][0]
        errors.extend((np.asarray(point) - projected_img_point).flatten() ** 2)
    # Upstream returns mean squared pixel residual, NOT RMS pixels.
    return float(np.mean(errors))


def bundle_adjustment(image_points, camera_poses, socketio):
    cameras = Cameras.instance()

    def params_to_camera_poses(params):
        focal_distances = []
        num_cameras = int((params.size-1)/7)+1
        camera_poses = [{
            "R": np.eye(3),
            "t": np.array([0,0,0], dtype=np.float32)
        }]
        focal_distances.append(params[0])
        for i in range(0, num_cameras-1):
            focal_distances.append(params[i*7+1])
            camera_poses.append({
                "R": Rotation.as_matrix(Rotation.from_rotvec(params[i*7 + 2 : i*7 + 3 + 2])),
                "t": params[i*7 + 3 + 2 : i*7 + 6 + 2]
            })

        return camera_poses, focal_distances

    def residual_function(params):
        camera_poses, focal_distances = params_to_camera_poses(params)
        for i in range(0, len(camera_poses)):
            intrinsic = cameras.get_camera_params(i)["intrinsic_matrix"]
            intrinsic[0, 0] = focal_distances[i]
            intrinsic[1, 1] = focal_distances[i]
            # cameras.set_camera_params(i, intrinsic)
        object_points = triangulate_points(image_points, camera_poses)
        errors = calculate_reprojection_errors(image_points, object_points, camera_poses)
        errors = errors.astype(np.float32)
        socketio.emit("camera-pose", {"camera_poses": camera_pose_to_serializable(camera_poses)})
        
        return errors

    focal_distance = cameras.get_camera_params(0)["intrinsic_matrix"][0,0]
    init_params = np.array([focal_distance])
    for i, camera_pose in enumerate(camera_poses[1:]):
        rot_vec = Rotation.as_rotvec(Rotation.from_matrix(camera_pose["R"])).flatten()
        focal_distance = cameras.get_camera_params(i)["intrinsic_matrix"][0,0]
        init_params = np.concatenate([init_params, [focal_distance]])
        init_params = np.concatenate([init_params, rot_vec])
        init_params = np.concatenate([init_params, camera_pose["t"].flatten()])

    res = optimize.least_squares(
        residual_function, init_params, verbose=2, loss="cauchy", ftol=1E-2
    )
    return params_to_camera_poses(res.x)[0]
    

def triangulate_point(image_points, camera_poses):
    valid = [(i, p, pose) for i, (p, pose) in enumerate(zip(image_points, camera_poses))
             if _valid_image_point(p)]
    if len(valid) <= 1:
        return [None, None, None]
    # Retain original camera indices when views are absent.  Reindexing K
    # after deleting missing views selects the wrong real-camera intrinsics.
    Ps = [_projection_for_pose(pose, i) for i, _, pose in valid]
    image_points = np.asarray([p for _, p, _ in valid], dtype=float)

    # https://temugeb.github.io/computer_vision/2021/02/06/direct-linear-transorms.html
    def DLT(Ps, image_points):
        A = []

        for P, image_point in zip(Ps, image_points):
            A.append(image_point[1]*P[2,:] - P[1,:])
            A.append(P[0,:] - image_point[0]*P[2,:])
            
        A = np.array(A).reshape((len(Ps)*2,4))
        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices = False)
        if abs(Vh[3,3]) < 1e-12:
            return [None, None, None]
        object_point = Vh[3,0:3]/Vh[3,3]

        return object_point

    object_point = DLT(Ps, image_points)

    return object_point


def triangulate_points(image_points, camera_poses):
    object_points = []
    for image_points_i in image_points:
        object_point = triangulate_point(image_points_i, camera_poses)
        object_points.append(object_point)
    
    return np.array(object_points)


def find_point_correspondance_and_object_points(image_points, camera_poses, frames=None):
    from multi_view import fuse_points
    intrinsics = [_intrinsic_for_pose(pose, i) for i, pose in enumerate(camera_poses)]
    return fuse_points(image_points, camera_poses, intrinsics,
                       _fundamental_from_projections, frames=frames, drawlines=drawlines)


def find_point_correspondance_and_object_points_reference(image_points, camera_poses, frames=None):
    """Pre-optimization implementation retained as a regression/benchmark oracle."""
    # Keep upstream's 5 px epipolar candidates and min-reprojection DLT choice.
    # Missing/invalid observations are bookkeeping, not a new tracking gate.
    image_points = [[list(p) for p in points if _valid_image_point(p)]
                    for points in image_points]
    if len(image_points) != len(camera_poses):
        raise ValueError("one observation list is required per calibrated camera")

    # [object_points, possible image_point groups, image_point from camera]
    correspondances = [[[i]] for i in image_points[0]]

    Ps = [] # projection matricies
    for i, camera_pose in enumerate(camera_poses):
        Ps.append(_projection_for_pose(camera_pose, i))

    root_image_points = [{"camera": 0, "point": point} for point in image_points[0]]

    for i in range(1, len(camera_poses)):
        epipolar_lines = []
        for root_image_point in root_image_points:
            F = _fundamental_from_projections(Ps[root_image_point["camera"]], Ps[i])
            line = cv.computeCorrespondEpilines(np.array([root_image_point["point"]], dtype=np.float32), 1, F)
            epipolar_lines.append(line[0,0].tolist())
            if frames is not None:
                frames[i] = drawlines(frames[i], line[0])

        not_closest_match_image_points = np.array(image_points[i])
        points = np.array(image_points[i])

        for j, [a, b, c] in enumerate(epipolar_lines):
            distances_to_line = np.array([])
            if len(points) != 0:
                distances_to_line = np.abs(a*points[:,0] + b*points[:,1] + c) / np.sqrt(a**2 + b**2)

            possible_matches = points[distances_to_line < 5].copy()

            # Commenting out this code produces more points, but more garbage points too
            # delete closest match from future consideration
            # if len(points) != 0:
            #     points = np.delete(points, np.argmin(distances_to_line), axis=0)

            # sort possible matches from smallest to largest
            distances_to_line = distances_to_line[distances_to_line < 5]
            possible_matches_sorter = distances_to_line.argsort()
            possible_matches = possible_matches[possible_matches_sorter]
    
            if len(possible_matches) == 0:
                for possible_group in correspondances[j]:
                    possible_group.append([None, None])
            else:
                not_closest_match_image_points = [row for row in not_closest_match_image_points.tolist() if row != possible_matches.tolist()[0]]
                not_closest_match_image_points = np.array(not_closest_match_image_points)
                
                new_correspondances_j = []
                for possible_match in possible_matches:
                    temp = copy.deepcopy(correspondances[j])
                    for possible_group in temp:
                        possible_group.append(possible_match.tolist())
                    new_correspondances_j += temp
                correspondances[j] = new_correspondances_j

        for not_closest_match_image_point in not_closest_match_image_points:
            root_image_points.append({"camera": i, "point": not_closest_match_image_point})
            temp = [[[None, None]] * i]
            temp[0].append(not_closest_match_image_point.tolist())
            correspondances.append(temp)

    object_points = []
    errors = []
    for groups in correspondances:
        candidates = []
        for group in groups:
            point = triangulate_point(group, camera_poses)
            if any(x is None for x in point) or not np.all(np.isfinite(point)):
                continue
            error = calculate_reprojection_error(group, point, camera_poses)
            if error is not None and np.isfinite(error):
                candidates.append((error, point))
        if candidates:
            error, point = min(candidates, key=lambda candidate: candidate[0])
            object_points.append(point)
            errors.append(error)
    return np.asarray(errors), np.asarray(object_points).reshape(-1, 3), frames


def locate_objects(object_points, errors):
    # Marker-rig geometry, in metres.  Upstream defaults are tuned for the
    # original drone rig (two markers 9.5 cm apart, third 15 cm from them).
    # Our 20 mm-marker rig is a small isosceles triangle:
    #   base (marker0-marker1) = 47 mm
    #   apex (marker2) to either base marker = 69.6 mm
    # So the reference point (apex) sees two markers at 69.6 mm that are
    # themselves 47 mm apart.  All three knobs are overridable.
    dist1 = float(os.environ.get("MOCAP_RIG_DIST1", 0.095))
    dist2 = float(os.environ.get("MOCAP_RIG_DIST2", 0.15))
    tol   = float(os.environ.get("MOCAP_RIG_TOL", 0.025))
    # Autoscale: the reconstructed points come out in an arbitrary metric from
    # the online calibration.  Because the marker rig's true size is known,
    # we can recover the real-world scale and hand the matcher metre units.
    # (Upstream does this manually via the "Set Scale Using Points" button.)
    if object_points.ndim == 2 and object_points.shape[0] >= 3 and \
       int(os.environ.get("MOCAP_AUTOSCALE", "0")):
        ds = []
        for a in range(object_points.shape[0]):
            for b in range(a + 1, object_points.shape[0]):
                ds.append(np.linalg.norm(object_points[a] - object_points[b]))
        ds = np.sort(np.array(ds))[:3]
        # expected triangle edges (smallest-to-largest)
        exp = np.sort(np.array([dist2, dist1, dist1]))
        if ds[0] > 1e-9:
            scale = float(np.median(exp / ds))
            object_points = object_points * scale

    distance_matrix = np.zeros((object_points.shape[0], object_points.shape[0]))
    already_matched_points = []
    objects = []

    for i in range(0, object_points.shape[0]):
        for j in range(0, object_points.shape[0]):
            distance_matrix[i,j] = np.sqrt(np.sum((object_points[i] - object_points[j])**2))

    for i in range(0, object_points.shape[0]):
        if i in already_matched_points:
            continue
        
        distance_deltas = np.abs(distance_matrix[i] - dist1)
        num_matches = distance_deltas < tol
        matches_index = np.where(distance_deltas < tol)[0]
        if np.sum(num_matches) >= 2:
            for possible_pair in cartesian_product(matches_index, matches_index):
                pair_distance = np.sqrt(np.sum((object_points[possible_pair[0]] - object_points[possible_pair[1]])**2))
                
                # if the pair isnt the correct distance apart
                if np.abs(pair_distance - dist2) > tol:
                    continue

                best_match_1_i = possible_pair[0]
                best_match_2_i = possible_pair[1]

                already_matched_points.append(i)
                already_matched_points.append(best_match_1_i)
                already_matched_points.append(best_match_2_i)

                location = (object_points[best_match_1_i]+object_points[best_match_2_i])/2
                error = np.mean([errors[i], errors[best_match_1_i], errors[best_match_2_i]])

                heading_vec = object_points[best_match_1_i] - object_points[best_match_2_i]
                heading_vec /= linalg.norm(heading_vec)
                heading = np.arctan2(heading_vec[1], heading_vec[0])

                heading = heading - np.pi if heading > np.pi/2 else heading
                heading = heading + np.pi if heading < -np.pi/2 else heading

                # determine drone index based on which side third light is on
                drone_index = 0 if (object_points[i] - location)[1] > 0 else 1

                objects.append({
                    "pos": location,
                    "heading": -heading,
                    "error": error,
                    "droneIndex": drone_index
                })

                break
    
    return objects


def numpy_fillna(data):
    data = np.array(data, dtype=object)
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.full((mask.shape[0], mask.shape[1], 2), [None, None])
    out[mask] = np.concatenate(data)
    return out
        

def drawlines(img1,lines):
    r,c,_ = img1.shape
    for r in lines:
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
    return img1


def make_square(img):
    x, y, _ = img.shape
    size = max(x, y)
    new_img = np.zeros((size, size, 3), dtype=np.uint8)
    ax,ay = (size - img.shape[1])//2,(size - img.shape[0])//2
    new_img[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img

    # Pad the new_img array with edge pixel values
    # Apply feathering effect
    feather_pixels = 8
    for i in range(feather_pixels):
        alpha = (i + 1) / feather_pixels
        new_img[ay - i - 1, :] = img[0, :] * (1 - alpha)  # Top edge
        new_img[ay + img.shape[0] + i, :] = img[-1, :] * (1 - alpha)  # Bottom edge


    return new_img


def camera_pose_to_serializable(camera_poses):
    for i in range(0, len(camera_poses)):
        camera_poses[i] = {k: v.tolist() for (k, v) in camera_poses[i].items()}

    return camera_poses

def cartesian_product(x, y):
    return np.array([[x0, y0] for x0 in x for y0 in y])

def add_white_border(image, border_size):
    height, width = image.shape[:2]
    bordered_image = cv.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv.BORDER_CONSTANT, value=[255, 255, 255])
    return bordered_image
