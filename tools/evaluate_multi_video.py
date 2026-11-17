"""Measure the repository-based pipeline on actual clean delivery videos.

Truth is used AFTER tracking, only to score outputs.  Every absent solution
is recorded; curves break at gaps. The selected position reference is used by
both the estimator and the independent trajectory truth (default: body origin).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
from tracking_pipeline import TrackingPipeline, serializable_result
from multi_body import rigid_pose, POSITION_REFERENCES, POSITION_REFERENCE_LABELS


def rotation_error_deg(estimated, truth):
    """Geodesic angle between two rotation matrices, in degrees."""
    value = (np.trace(np.asarray(estimated) @ np.asarray(truth).T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def reference_points(markers_path, templates):
    markers = {}
    with open(markers_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["frame"]), int(row.get("body_id", 0)), int(row["marker"]))
            markers[key] = np.array([float(row[axis]) for axis in "xyz"])
    out = {}
    for template in templates:
        body = template["body_id"]
        # Marker IDs in these rig files are local indices; use the declared
        # exact isosceles base, not whichever pair happens to be matched.
        i, j = template["base_pair"]
        frames = {key[0] for key in markers if key[1] == body}
        for frame in frames:
            out[(frame, body)] = 0.5 * (markers[(frame, body, i)] + markers[(frame, body, j)])
    return out


def reference_poses(markers_path, templates):
    """Truth 6-DOF pose per frame from the rendered marker positions.

    Kabsch on every marker the rig declares, so this is the same rigid-body
    definition the estimator reports, not an extra fit.
    """
    per = {}
    with open(markers_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["frame"]), int(row.get("body_id", 0)))
            per.setdefault(key, {})[int(row["marker"])] = np.array(
                [float(row[axis]) for axis in "xyz"])
    out = {}
    for template in templates:
        body = template["body_id"]
        for (frame, other), markers in per.items():
            if other != body:
                continue
            index = [i for i in range(len(template["locals"])) if i in markers]
            if len(index) < 3:
                continue
            rotation, translation, rmse = rigid_pose(
                template["locals"][index], np.array([markers[i] for i in index]))
            out[(frame, body)] = (rotation, translation, rmse)
    return out


def trajectory_poses(path):
    """Independent body-origin/quaternion truth, used only for scoring."""
    poses = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rotation = Rotation.from_quat([float(row[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
            translation = np.array([float(row[k]) for k in "xyz"])
            poses[(int(row["frame"]), int(row["body_id"]))] = (rotation, translation, 0.0)
    return poses


def error_summary(values):
    data = np.asarray(values, dtype=float)
    if not data.size:
        return {"count": 0, "mean_mm": None, "p90_mm": None, "max_mm": None}
    return {"count": int(data.size), "mean_mm": float(data.mean()),
            "median_mm": float(np.median(data)), "p90_mm": float(np.percentile(data, 90)),
            "rmse_mm": float(np.sqrt(np.mean(data ** 2))),
            "max_mm": float(data.max())}


def value_summary(values, unit=1.0):
    data = np.asarray(values, dtype=float)
    if not data.size:
        return {"count": 0}
    return {"count": int(data.size),
            "mean": float(data.mean() * unit),
            "median": float(np.median(data) * unit),
            "p90": float(np.percentile(data, 90) * unit),
            "max": float(data.max() * unit),
             "std": float(data.std() * unit)}


def video_frames(captures, frame_ids):
    """Decode the requested frame numbers, seeking only across gaps."""
    next_frame = None
    for frame in frame_ids:
        if frame != next_frame:
            for capture in captures:
                if not capture.set(cv.CAP_PROP_POS_FRAMES, frame):
                    raise OSError(f"cannot seek a camera video to frame {frame}")
        decoded = [capture.read() for capture in captures]
        if not all(ok and image is not None for ok, image in decoded):
            raise OSError(f"video decode failed at frame {frame}; not filling with stale pixels")
        yield [image for _, image in decoded]
        next_frame = frame + 1


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def video_provenance(paths, manifest_path, geometry_sha256):
    """Bind scores to actual clip bytes and, when supplied, render provenance."""
    hashes = {os.path.basename(path): sha256(path) for path in paths}
    provenance = {"video_sha256": hashes, "render_source_sha256": None}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        declared = {clip["clean"]["file"]: clip["clean"]["sha256"] for clip in manifest["clips"]}
        if hashes != declared:
            raise ValueError("video bytes do not match their encoding manifest")
        if manifest["source_sha256"]["rig"] != geometry_sha256["body_specs"]:
            raise ValueError("videos were rendered from a different rig")
        provenance["render_source_sha256"] = manifest["source_sha256"]
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-dir", default=os.path.join(ROOT, "out", "rig3"))
    parser.add_argument("--video-dir", default=os.path.join(ROOT, "out", "video"))
    parser.add_argument("--tag", default="t650")
    parser.add_argument("--trajectory", default=os.path.join(ROOT, "out", "scenario3.csv"))
    parser.add_argument("--position-reference", choices=POSITION_REFERENCES, default="body_origin")
    parser.add_argument("--max-speed", type=float, default=10.0,
                        help="velocity measurement magnitude limit (m/s), for the selected reference")
    parser.add_argument("--pattern", default="{camera}_{tag}",
                        help="frame folder pattern for --input images")
    parser.add_argument("--input", choices=("video", "images"), default="video")
    parser.add_argument("--frames", default="0-599")
    parser.add_argument("--threshold", type=float, default=140)
    parser.add_argument("--blur-size", type=int, default=9)
    parser.add_argument("--centroid-mode", choices=("integer", "subpixel"), default="integer")
    parser.add_argument("--tol", type=float, default=0.025)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--gate-m", type=float, default=0.5,
                        help="max per-body single-frame jump (m) before a blip is dropped")
    parser.add_argument("--extra-tol", type=float, default=0.05,
                        help="rigid prediction budget for non-reference markers (m)")
    parser.add_argument("--rigidity-tol", type=float, default=0.002,
                        help="reject a fix whose reference-triangle shape residual exceeds this (m); "
                             "0 or negative disables the gate")
    parser.add_argument("--reacquire-inflation", type=float, default=5.0,
                        help="Kalman covariance reopened per missed frame, in units of the "
                             "measurement noise; 0 disables it")
    parser.add_argument("--max-err", type=float, default=6.0,
                        help="drop a matched triangle whose mean reprojection MSE (px^2) exceeds this")
    parser.add_argument("--out-prefix", default=os.path.join(ROOT, "out", "rig3", "tracking_from_video"))
    args = parser.parse_args()
    try:
        if "-" in args.frames:
            start, end = map(int, args.frames.split("-"))
            frame_ids = list(range(start, end + 1))
        else:
            frame_ids = [int(value) for value in args.frames.split(",")]
        if (not frame_ids or frame_ids[0] < 0 or
                any(b <= a for a, b in zip(frame_ids, frame_ids[1:]))):
            raise ValueError("frame numbers must be nonnegative and strictly increasing")
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("fps must be positive")
    if not np.isfinite(args.rigidity_tol):
        parser.error("rigidity tolerance must be finite")
    if args.rigidity_tol <= 0:
        args.rigidity_tol = None  # explicit opt-out of the rigid-shape gate
    if not np.isfinite(args.reacquire_inflation) or args.reacquire_inflation < 0:
        parser.error("reacquire inflation must be non-negative")
    start, end = frame_ids[0], frame_ids[-1]
    pipeline = TrackingPipeline(os.path.join(args.rig_dir, "cameras_calib.json"),
                                os.path.join(args.rig_dir, "rig_blender.json"),
                                threshold=args.threshold, blur_size=args.blur_size,
                                tolerance=args.tol, fps=args.fps,
                                gate_m=args.gate_m, max_err=args.max_err,
                                extra_tol=args.extra_tol,
                                rigidity_tol=args.rigidity_tol,
                                reacquire_inflation=args.reacquire_inflation,
                                position_reference=args.position_reference,
                                max_speed=args.max_speed, centroid_mode=args.centroid_mode)
    truth_poses = trajectory_poses(args.trajectory)
    offsets = {t["body_id"]: (t["base_mid_local"] if args.position_reference == "base_midpoint"
                              else np.zeros(3)) for t in pipeline.templates}
    reference = {key: translation + rotation @ offsets[key[1]]
                 for key, (rotation, translation, _) in truth_poses.items() if key[1] in offsets}
    video_paths = [os.path.join(args.video_dir, f"{camera[-2:]}_{camera.lower()}_{args.tag}_20s.mp4")
                   for camera in pipeline.camera_names]
    captures = []
    records, rows = [], []
    provenance = {}
    try:
        if args.input == "video":
            provenance = video_provenance(video_paths, os.path.join(args.video_dir, f"{args.tag}_manifest.json"),
                                          pipeline.geometry_sha256)
            captures = [cv.VideoCapture(path) for path in video_paths]
            if not all(capture.isOpened() for capture in captures):
                raise OSError("one or more clean camera videos could not be opened")
            for capture in captures:
                if abs(capture.get(cv.CAP_PROP_FPS) - args.fps) > 1e-3:
                    raise ValueError("video frame rate does not match the evaluation clock")
                if int(capture.get(cv.CAP_PROP_FRAME_COUNT)) <= end:
                    raise ValueError("a camera video is shorter than the requested range")
            decoded_frames = video_frames(captures, frame_ids)
        stats = {body: {"raw": [], "kf": [], "identity_n": 0, "identity_correct": 0,
                        "pose_present": 0, "pose_modes": {},
                        "pose_origin": [], "pose_rotation": [], "pose_tilt": [],
                        "pose_fit": []}
                 for body in pipeline.body_ids}
        for frame in frame_ids:
            if captures:
                images = next(decoded_frames)
            else:
                images = [cv.imread(os.path.join(ROOT, "out", "frames",
                                                 args.pattern.format(camera=camera,
                                                                     tag=args.tag),
                                                 f"f{frame:04d}.png"), cv.IMREAD_GRAYSCALE)
                          for camera in pipeline.camera_names]
                if any(image is None for image in images):
                    raise OSError(f"missing rendered image at frame {frame}")
            result = pipeline.process(images, frame / args.fps)
            result["frame"] = frame
            refs = {body: reference[(frame, body)] for body in pipeline.body_ids}
            raw = {obj["droneIndex"]: obj for obj in result["objects"]}
            filtered = {obj["droneIndex"]: obj for obj in result["filtered_objects"]}
            result["reference"] = {str(body): point for body, point in refs.items()}
            result["errors"] = {}
            for body in pipeline.body_ids:
                row = {"frame": frame, "t": frame / args.fps, "body_id": body,
                       "raw_available": int(body in raw), "kf_available": int(body in filtered)}
                for axis, value in zip("xyz", refs[body]):
                    row[f"truth_{axis}"] = float(value)
                metrics = {}
                for key, values, prefix in (("raw", raw, "raw"), ("kf", filtered, "kf")):
                    if body not in values:
                        metrics[key] = None
                        continue
                    point = np.asarray(values[body]["pos"])
                    if values[body].get("position_reference") != args.position_reference:
                        raise ValueError("estimate and scoring reference differ")
                    error = float(np.linalg.norm(point - refs[body]) * 1000)
                    stats[body][key].append(error)
                    metrics[key] = error
                    row[f"{prefix}_error_mm"] = error
                    for axis, value in zip("xyz", point):
                        row[f"{prefix}_{axis}"] = float(value)
                    if key == "raw":
                        for axis, value in zip("xyz", values[body]["base_midpoint_measured"]):
                            row[f"base_midpoint_measured_{axis}"] = float(value)
                    nearest = min(refs, key=lambda other: np.linalg.norm(point - refs[other]))
                    row[f"{prefix}_nearest_body"] = nearest
                    if key == "raw":
                        stats[body]["identity_n"] += 1
                        stats[body]["identity_correct"] += int(nearest == body)
                        # Rigid-body pose is carried by the raw measurement;
                        # the Kalman state only passes it through unchanged.
                        pose = values[body].get("pose_t")
                        truth_pose = truth_poses.get((frame, body))
                        if pose is not None and truth_pose is not None:
                            rotation_t, translation_t, _ = truth_pose
                            stats[body]["pose_present"] += 1
                            mode = values[body].get("pose_mode", "triangle3")
                            stats[body]["pose_modes"][mode] = \
                                stats[body]["pose_modes"].get(mode, 0) + 1
                            origin = float(np.linalg.norm(
                                np.asarray(pose) - np.asarray(translation_t)) * 1000)
                            rotation = rotation_error_deg(values[body]["pose_r"], rotation_t)
                            truth_tilt = math.degrees(math.acos(np.clip(rotation_t[2, 2], -1.0, 1.0)))
                            stats[body]["pose_origin"].append(origin)
                            stats[body]["pose_rotation"].append(rotation)
                            stats[body]["pose_tilt"].append(
                                abs(float(values[body].get("pose_tilt_deg", truth_tilt)) - truth_tilt))
                            stats[body]["pose_fit"].append(
                                float(values[body].get("pose_rmse_m", 0.0)) * 1000)
                            row["pose_mode"] = mode
                            row["pose_disambiguation"] = values[body].get("pose_disambiguation")
                            row["pose_markers_used"] = values[body].get("markers_used", 0)
                            row["pose_origin_err_mm"] = origin
                            row["pose_rot_err_deg"] = rotation
                            row["pose_fit_rmse_mm"] = stats[body]["pose_fit"][-1]
                result["errors"][str(body)] = metrics
                rows.append(row)
            records.append(serializable_result(result))
            if frame % 100 == 0 or frame == end:
                print(f"frame {frame}: dots={list(map(len, result['detections']))}, "
                      f"triangulated={len(result['points'])}, bodies={list(raw)}", flush=True)
    finally:
        for capture in captures:
            capture.release()
    count = len(frame_ids)
    summary = {"input": args.input, "source_files": video_paths if captures else [],
                "frame_range": [start, end], "frame_count": count, "fps": args.fps,
                "selected_frames": frame_ids,
                "camera_names": pipeline.camera_names,
                "configuration": pipeline.configuration,
                "geometry_sha256": pipeline.geometry_sha256,
                "provenance": provenance,
                "truth_source": {"trajectory": os.path.abspath(args.trajectory),
                                 "sha256": sha256(args.trajectory)},
                 "algorithm_sha256": {path.replace(os.sep, "/"): sha256(os.path.join(ROOT, path)) for path in (
                    "adapters/low_cost_mocap/multi_body.py", "adapters/low_cost_mocap/multi_view.py",
                    "adapters/low_cost_mocap/tracking_pipeline.py",
                    "adapters/low_cost_mocap/low-cost-mocap/computer_code/api/helpers.py",
                    "adapters/low_cost_mocap/low-cost-mocap/computer_code/api/KalmanFilter.py",
                    "adapters/low_cost_mocap/low-cost-mocap/computer_code/api/LowPassFilter.py")},
               "algorithm": ("Low-Cost-Mocap ordered 5px epipolar search + cached/batched DLT + multi-template "
                             "apex/base search + extra-marker Kabsch pose + repository Kalman statePre"),
                "detection": {"threshold": args.threshold, "blur_size": args.blur_size,
                              "min_area": 3, "max_area": 200,
                              "centroid_mode": args.centroid_mode,
                              "integer_centroids": args.centroid_mode == "integer"},
               "match_tolerance_m": args.tol,
               "extra_marker_tolerance_m": args.extra_tol,
                "position_reference_key": args.position_reference,
                "position_reference": POSITION_REFERENCE_LABELS[args.position_reference] + " in calibrated ENU world",
                "pose_reference": "body-frame origin and quaternion from the independent scenario trajectory",
                "kalman": {"process_noise": 1.0, "measurement_noise": 1.0, "reacquire_gap_s": 0.5,
                           "max_speed_mps": args.max_speed, "output": "statePre", "state_dim": 9},
               "bodies": []}
    for template in pipeline.templates:
        body = template["body_id"]
        stat = stats[body]
        body_summary = {"body_id": body, "name": template["name"],
                        "raw_coverage_percent": 100 * len(stat["raw"]) / count,
                        "kalman_coverage_percent": 100 * len(stat["kf"]) / count,
                        "raw": error_summary(stat["raw"]), "kalman": error_summary(stat["kf"]),
                        "nearest_truth_id_correct": stat["identity_correct"],
                        "nearest_truth_id_count": stat["identity_n"],
                        "nearest_truth_id_percent": (100 * stat["identity_correct"] / stat["identity_n"]
                                                     if stat["identity_n"] else None),
                        "ambiguous_apex_at_tolerance": template["edge_separation_m"] <= 2 * args.tol,
                        "pose_accuracy": {
                            "coverage_percent": 100 * stat["pose_present"] / count,
                            "mode_counts": dict(sorted(stat["pose_modes"].items())),
                            "origin_err_mm": value_summary(stat["pose_origin"]),
                            "rotation_err_deg": value_summary(stat["pose_rotation"]),
                            "tilt_err_deg": value_summary(stat["pose_tilt"]),
                            "fit_rmse_mm": value_summary(stat["pose_fit"]),
                        }}
        summary["bodies"].append(body_summary)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    fields = ["frame", "t", "body_id", "raw_available", "kf_available"]
    fields += [f"{prefix}_{axis}" for prefix in ("truth", "raw", "kf") for axis in "xyz"]
    fields += [f"base_midpoint_measured_{axis}" for axis in "xyz"]
    fields += ["raw_error_mm", "kf_error_mm", "raw_nearest_body", "kf_nearest_body",
                "pose_mode", "pose_disambiguation", "pose_markers_used", "pose_origin_err_mm",
               "pose_rot_err_deg", "pose_fit_rmse_mm"]
    with open(args.out_prefix + ".csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out_prefix + ".json", "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, allow_nan=False,
                  separators=(",", ":"))
    with open(args.out_prefix + "_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    for body in summary["bodies"]:
        raw, kf = body["raw"], body["kalman"]
        pose = body["pose_accuracy"]
        print(f"{body['name']}: coverage={body['raw_coverage_percent']:.1f}%, "
              f"raw mean={raw['mean_mm']}, p90={raw['p90_mm']}, max={raw['max_mm']} mm; "
              f"KF mean={kf['mean_mm']} mm", flush=True)
        if pose["mode_counts"]:
            origin, rotation = pose["origin_err_mm"], pose["rotation_err_deg"]
            print(f"   pose {pose['coverage_percent']:.1f}% {pose['mode_counts']} "
                  f"origin {origin['mean']:.2f} mm, rotation {rotation['mean']:.3f} deg "
                  f"(max {rotation['max']:.2f})", flush=True)
    print("wrote", args.out_prefix + ".csv", "and .json")


if __name__ == "__main__":
    main()
