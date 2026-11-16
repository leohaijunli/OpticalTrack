"""Single-factor deployment sensitivity using the actual video detection cache.

Complete 30-fps sequences retain their original capture timestamps. Perturbations
exercise the real matching, rigid-body and Kalman code, without truth labels in
the estimator. Truth is used only afterwards for scoring. Centroid jitter is a
point-level error model, not a simulation of image noise or changed exposure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import cProfile
from datetime import datetime, timezone
import json
import multiprocessing
import os
import pstats
import sys

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
from tracking_pipeline import TrackingPipeline
from tools.accuracy_stats3 import rotation_error_deg, truth_tables
from tools.evaluate_multi_video import sha256


def metric(values):
    data = np.asarray(values, dtype=float)
    if not data.size:
        return {"count": 0, "mean": None, "p95": None, "max": None, "rmse": None}
    return {"count": len(values), "mean": float(data.mean()),
            "p95": float(np.percentile(data, 95)), "max": float(data.max()),
            "rmse": float(np.sqrt(np.mean(data ** 2)))}


def score(records, truth, body_ids, fps):
    values = {body: {"position": [], "speed": [], "rotation": [],
                     "missing": 0, "longest_gap": 0, "nearest_other": 0,
                     "temporal_prior": 0} for body in body_ids}
    all_present = 0
    for record in records:
        frame = record["frame"]
        objects = {obj["droneIndex"]: obj for obj in record["filtered_objects"]}
        all_present += int(all(body in objects for body in body_ids))
        for body in body_ids:
            stat = values[body]
            if body not in objects:
                stat["missing"] += 1
                stat["longest_gap"] = max(stat["longest_gap"], stat["missing"])
                continue
            stat["missing"] = 0
            obj = objects[body]
            point = np.asarray(obj["pos"])
            stat["position"].append(float(np.linalg.norm(point - truth[body]["pos"][frame]) * 1000))
            nearest = min(body_ids, key=lambda other: np.linalg.norm(point - truth[other]["pos"][frame]))
            stat["nearest_other"] += int(nearest != body)
            speed = truth[body]["speed"][frame]
            if speed is not None:
                stat["speed"].append(abs(float(np.linalg.norm(obj["vel"])) - speed))
            if "pose_r" in obj:
                stat["rotation"].append(rotation_error_deg(obj["pose_r"], truth[body]["rotation"][frame]))
            stat["temporal_prior"] += int(obj.get("pose_disambiguation") == "temporal_prior")
    bodies = {str(body): {
        "coverage_percent": 100 * len(stat["position"]) / len(records),
        "position_error_mm": metric(stat["position"]),
        "speed_absolute_error_mps": metric(stat["speed"]),
        "rotation_error_deg": metric(stat["rotation"]),
        "longest_missing_frames": stat["longest_gap"],
        "longest_missing_duration_s": stat["longest_gap"] / fps,
        "nearest_other_body_count": stat["nearest_other"],
        "temporal_prior_count": stat["temporal_prior"]} for body, stat in values.items()}
    return {"bodies": bodies, "all_bodies_available_percent": 100 * all_present / len(records),
            "pooled_position_error_mm": metric([v for stat in values.values() for v in stat["position"]]),
            "pooled_rotation_error_deg": metric([v for stat in values.values() for v in stat["rotation"]])}


def evaluate_case(job):
    cv.setNumThreads(1)
    with open(job["cache"], encoding="utf-8") as handle:
        cache = json.load(handle)
    configuration = cache["summary"]["configuration"]
    fps = cache["summary"]["fps"]
    pipeline = TrackingPipeline(os.path.join(job["rig_dir"], "cameras_calib.json"),
                                 os.path.join(job["rig_dir"], "rig_blender.json"), fps=fps)
    if (configuration != pipeline.configuration or
            cache["summary"]["geometry_sha256"] != pipeline.geometry_sha256):
        raise ValueError("cache is not from the current baseline configuration/geometry")
    case = job["case"]
    angle = case.get("calibration_rotation_deg", 0.0)
    if angle:
        # A calibration error: rotate each camera's extrinsics around its local
        # optical Y axis, alternating signs; centre position remains fixed.
        for index, pose in enumerate(pipeline.poses):
            delta = Rotation.from_euler("y", angle * (-1) ** index, degrees=True).as_matrix()
            pose["R"] = delta @ pose["R"]
            pose["t"] = delta @ pose["t"]
    source = cache["records"]
    if [record["frame"] for record in source] != list(range(len(source))):
        raise ValueError("sensitivity study needs the complete contiguous sequence from frame zero")
    rng = np.random.default_rng(job["seed"])
    records = []
    for index, record in enumerate(source):
        points = [[list(point) for point in camera] for camera in record["detections"]]
        valid = list(record["valid_cameras"])
        if "drop_camera" in case:
            points[case["drop_camera"]] = []
            valid[case["drop_camera"]] = False
        if "lag_cameras" in case and index >= case["lag_frames"]:
            for camera in case["lag_cameras"]:
                points[camera] = source[index - case["lag_frames"]]["detections"][camera]
        if "jitter_sigma_px" in case:
            points = [(np.asarray(camera) + rng.normal(0, case["jitter_sigma_px"], (len(camera), 2))).tolist()
                      if camera else [] for camera in points]
        result = pipeline.process_detections(points, record["t"], valid)
        result["frame"] = record["frame"]
        records.append(result)
    # Ground truth is introduced only after all tracking outputs have been made.
    truth = truth_tables(job["trajectory"], pipeline.templates, fps=fps, position_reference="body_origin")
    return dict(case, scores=score(records, truth, pipeline.body_ids, fps))


def profile_fusion(cache, pipeline, frame_count):
    profiler = cProfile.Profile()
    for record in cache["records"][:frame_count]:
        profiler.runcall(pipeline.process_detections, record["detections"], record["t"],
                         record["valid_cameras"])
    stats = pstats.Stats(profiler)
    entries = []
    for (filename, line, name), (_, calls, self_s, cumulative_s, _) in stats.stats.items():
        path = os.path.relpath(filename, ROOT).replace(os.sep, "/") if os.path.isabs(filename) else filename
        entries.append({"function": f"{path}:{line}:{name}", "calls": calls,
                        "self_s": self_s, "cumulative_s": cumulative_s})
    selected = sorted(entries, key=lambda entry: entry["cumulative_s"], reverse=True)[:25]
    print("Fusion profile (instrumented, not a throughput benchmark):", flush=True)
    for entry in selected[:8]:
        print(f"  {entry['cumulative_s']:.3f}s {entry['function']}", flush=True)
    return {"frame_count": min(frame_count, len(cache["records"])), "total_instrumented_s": stats.total_tt,
            "note": "cProfile adds overhead; cumulative entries overlap; use edge_equivalence.json for uninstrumented service time",
            "top_cumulative": selected,
            "top_self": sorted(entries, key=lambda entry: entry["self_s"], reverse=True)[:25]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-dir", default=os.path.join(ROOT, "out", "rig3"))
    parser.add_argument("--cache", default=os.path.join(ROOT, "out", "rig3", "tracking_from_video.json"))
    parser.add_argument("--trajectory", default=os.path.join(ROOT, "out", "scenario3.csv"))
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--profile-frames", type=int, default=60)
    parser.add_argument("--out", default=os.path.join(ROOT, "out", "rig3", "deployment_sensitivity.json"))
    args = parser.parse_args()
    if args.workers < 1 or args.profile_frames < 1:
        parser.error("workers and profile-frames must be positive")
    with open(args.cache, encoding="utf-8") as handle:
        cache = json.load(handle)
    pipeline = TrackingPipeline(os.path.join(args.rig_dir, "cameras_calib.json"),
                                 os.path.join(args.rig_dir, "rig_blender.json"), fps=cache["summary"]["fps"])
    profile = profile_fusion(cache, pipeline, args.profile_frames)
    truth = truth_tables(args.trajectory, pipeline.templates, fps=cache["summary"]["fps"],
                         position_reference="body_origin")
    cases = [{"name": f"drop_{cam}", "drop_camera": i} for i, cam in enumerate(pipeline.camera_names)]
    cases.extend([{"name": "centroid_jitter_0.5px", "jitter_sigma_px": 0.5},
                  {"name": "centroid_jitter_1.0px", "jitter_sigma_px": 1.0},
                  {"name": "CAM04_06_lag_one_frame", "lag_cameras": [3, 4, 5], "lag_frames": 1},
                  {"name": "calibration_rotation_0.05deg", "calibration_rotation_deg": 0.05}])
    jobs = [dict(case=case, seed=args.seed, cache=args.cache, rig_dir=args.rig_dir,
                 trajectory=args.trajectory) for case in cases]
    results = [{"name": "baseline", "scores": score(cache["records"], truth, pipeline.body_ids,
                                                     cache["summary"]["fps"])}]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(evaluate_case, jobs):
            results.append(result)
            stats = result["scores"]
            print(f"{result['name']}: all-three={stats['all_bodies_available_percent']:.2f}%, "
                  f"position mean={stats['pooled_position_error_mm']['mean']:.3f} mm", flush=True)
    motion = {str(body): {"max_speed_mps": max(v for v in table["speed"].values() if v is not None)}
              for body, table in truth.items()}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "single-factor perturbations of actual decoded video centroids; not measured hardware degradation",
              "fps": cache["summary"]["fps"], "frame_count": len(cache["records"]), "seed": args.seed,
              "centroid_noise_model": "independent Gaussian per point/axis/frame, after detection; one seeded realization",
              "lag_model": "CAM04..06 use previous video frame after frame 0; timestamp intentionally mislabeled as current",
              "calibration_model": "alternating +/-0.05 degrees about each camera local Y axis; centre fixed",
              "metric_note": "errors are conditional on available output; coverage and longest gaps reported separately; nearest body is not a formal ID-switch metric",
              "configuration": pipeline.configuration, "geometry_sha256": pipeline.geometry_sha256,
              "cache_sha256": sha256(args.cache), "trajectory_sha256": sha256(args.trajectory),
              "script_sha256": sha256(__file__), "motion": motion, "profile": profile, "cases": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
