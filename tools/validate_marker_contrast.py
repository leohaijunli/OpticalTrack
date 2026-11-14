"""Audit visible-marker contrast and false blobs using the actual detector.

Ground truth is used only for post-hoc one-to-one scoring.  Image pixels are
never masked or altered before the detector.  Motor locations are diagnostic
regions, not tracking priors.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adapters/low_cost_mocap"))
from tracking_pipeline import load_calibration, project
from helpers import prepare_camera_frame, detect_visible_dots


def load_truth(rig_dir, trajectory):
    markers, body_poses = {}, {}
    with open(rig_dir / "markers_3d.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            markers.setdefault(int(row["frame"]), []).append(
                np.array([float(row[axis]) for axis in "xyz"]))
    with open(trajectory, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            body_poses.setdefault(int(row["frame"]), []).append((
                np.array([float(row[axis]) for axis in "xyz"]),
                Rotation.from_quat([float(row[key]) for key in ("qx", "qy", "qz", "qw")]).as_matrix()))
    return markers, body_poses


def audit(image, marker_points, motors, camera, params, threshold, blur_size):
    prepared = prepare_camera_frame(image, params, True, blur_size)
    gray = cv.cvtColor(prepared, cv.COLOR_BGR2GRAY)
    detections = detect_visible_dots(prepared, threshold, 3, 200)
    height, width = gray.shape
    projected = [project(camera, point) for point in marker_points]
    projected = [point for point in projected if point is not None and 0 <= point[0] < width and 0 <= point[1] < height]
    motor_uv = [project(camera, point) for point in motors]
    motor_uv = [point for point in motor_uv if point is not None]
    detected = np.asarray(detections, float).reshape(-1, 2)
    matched, error = set(), []
    if len(projected) and len(detected):
        distances = np.linalg.norm(np.asarray(projected)[:, None, :] - detected[None, :, :], axis=2)
        ti, di = linear_sum_assignment(distances)
        for a, b in zip(ti, di):
            if distances[a, b] <= 4:
                matched.add(int(b))
                error.append(float(distances[a, b]))
    false = [point for index, point in enumerate(detections) if index not in matched]
    motor_false = [point for point in false if motor_uv and
                   min(np.linalg.norm(np.asarray(point) - uv) for uv in motor_uv) <= 6]
    def patch_peak(uv, radius=3):
        x, y = np.rint(uv).astype(int)
        if not (0 <= x < width and 0 <= y < height):
            return None
        return int(gray[max(0, y-radius):min(height, y+radius+1),
                        max(0, x-radius):min(width, x+radius+1)].max())
    # Exclude motor measurement ROIs that overlap a projected LED, otherwise
    # the LED itself would be incorrectly reported as a bright motor surface.
    motor_peaks = [patch_peak(uv) for uv in motor_uv
                   if not projected or min(np.linalg.norm(np.asarray(uv)-point) for point in projected) > 10]
    motor_peaks = [value for value in motor_peaks if value is not None]
    led_peaks = [patch_peak(uv) for uv in projected]
    return {"detections": len(detections), "projected_markers": len(projected),
            "matched_markers": len(matched), "false_blobs": len(false),
            "motor_false_blobs": len(motor_false), "false_blob_pixels": false,
            "motor_patch_max_after_blur": max(motor_peaks, default=None),
            "led_patch_peaks_after_blur": led_peaks, "matched_errors_px": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-dir", type=Path, default=ROOT / "out/rig3")
    parser.add_argument("--trajectory", type=Path, default=ROOT / "out/scenario3.csv")
    parser.add_argument("--pattern", default="still_{cam}_t650")
    parser.add_argument("--baseline-pattern", default="")
    parser.add_argument("--frames", default="0,85,214,546,599")
    parser.add_argument("--cams", default="", help="comma-separated camera names (default: all)")
    parser.add_argument("--threshold", type=float, default=140)
    parser.add_argument("--blur-size", type=int, default=9)
    parser.add_argument("--out", type=Path, default=ROOT / "out/rig3/led_contrast_pilot.json")
    args = parser.parse_args()
    frames = list(range(600)) if args.frames == "all" else [int(v) for v in args.frames.split(",")]
    calibration, params, poses = load_calibration(args.rig_dir / "cameras_calib.json")
    truth, body_poses = load_truth(args.rig_dir, args.trajectory)
    results = {"threshold": args.threshold, "blur_size": args.blur_size, "frames": frames, "datasets": {}}
    for label, pattern in (("baseline", args.baseline_pattern), ("corrected", args.pattern)):
        if not pattern:
            continue
        rows = []
        for frame in frames:
            motors = [position + rotation @ np.array([x, y, .062])
                      for position, rotation in body_poses[frame]
                      for x in (-.23, .23) for y in (-.23, .23)]
            for c, params_ci, pose in zip(calibration["cameras"], params, poses):
                if args.cams and c["name"] not in args.cams.split(","):
                    continue
                path = ROOT / "out/frames" / pattern.format(cam=c["name"]) / f"f{frame:04d}.png"
                image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(path)
                row = audit(image, truth[frame], motors, pose, params_ci, args.threshold, args.blur_size)
                row.update({"frame": frame, "camera": c["name"]})
                rows.append(row)
        summary = {key: sum(row[key] for row in rows) for key in
                   ("detections", "projected_markers", "matched_markers", "false_blobs", "motor_false_blobs")}
        summary["motor_patch_max_after_blur"] = max(
            (row["motor_patch_max_after_blur"] for row in rows if row["motor_patch_max_after_blur"] is not None), default=None)
        results["datasets"][label] = {"pattern": pattern, "summary": summary, "rows": rows}
        print(label, json.dumps(summary), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
