"""Measure the actual centre pipeline and compare every output to video evidence.

Input is the original decoded-video centroid cache. Capture timestamps, all
camera/point orders and all tracking gates remain unchanged. No image decoding,
networking or UI is included in these centre-only service times.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import sys
import time

import cv2 as cv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
import tracking_pipeline
from tracking_pipeline import TrackingPipeline, serializable_result
from tools.edge_equivalence_check import json_bytes, timing_summary
from tools.evaluate_multi_video import sha256


def numeric_differences(actual, expected, path="", atol=1e-8, rtol=1e-7):
    """Discrete fields are exact; only floating-point leaves admit roundoff."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return [path + ": keys differ"]
        return [error for key in expected for error in
                numeric_differences(actual[key], expected[key], path + "/" + key, atol, rtol)]
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [path + ": lengths differ"]
        return [error for i, (a, e) in enumerate(zip(actual, expected)) for error in
                numeric_differences(a, e, path + f"/{i}", atol, rtol)]
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not np.isfinite(actual):
            return [path + ": invalid numeric output"]
        return [] if abs(actual - expected) <= atol + rtol * abs(expected) else [path + ": numeric difference"]
    return [] if type(actual) is type(expected) and actual == expected else [path + ": discrete difference"]


def update_deltas(deltas, actual, expected):
    for collection in ("objects", "filtered_objects"):
        old = {obj["droneIndex"]: obj for obj in expected[collection]}
        for obj in actual[collection]:
            ref = old.get(obj["droneIndex"])
            if ref is None:
                continue
            for key, unit in (("pos", 1000), ("pose_t", 1000), ("vel", 1), ("acc", 1)):
                if key in obj and key in ref:
                    delta = float(np.linalg.norm(np.asarray(obj[key]) - np.asarray(ref[key])) * unit)
                    name = f"{collection}.{key}.max_delta_" + ("mm" if unit == 1000 else "SI")
                    deltas[name] = max(deltas.get(name, 0), delta)
            if "pose_r" in obj and "pose_r" in ref:
                delta = float(np.max(np.abs(np.asarray(obj["pose_r"]) - np.asarray(ref["pose_r"]))))
                name = f"{collection}.pose_r.max_component_delta"
                deltas[name] = max(deltas.get(name, 0), delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=os.path.join(ROOT, "out", "rig3", "tracking_from_video.json"))
    parser.add_argument("--rig-dir", default=os.path.join(ROOT, "out", "rig3"))
    parser.add_argument("--out", default=os.path.join(ROOT, "out", "rig3", "fusion_benchmark.json"))
    parser.add_argument("--baseline-report")
    parser.add_argument("--backend", choices=("current", "reference"), default="current")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--target-fps", type=float, default=60)
    parser.add_argument("--cpu-label", default=platform.processor())
    args = parser.parse_args()
    if args.repeats < 1 or args.opencv_threads < 1 or args.target_fps <= 0:
        parser.error("repeats, opencv-threads and target-fps must be positive")
    cv.setNumThreads(args.opencv_threads)
    if args.backend == "reference":
        from helpers import find_point_correspondance_and_object_points_reference
        tracking_pipeline.find_point_correspondance_and_object_points = find_point_correspondance_and_object_points_reference
    with open(args.cache, encoding="utf-8") as handle:
        cache = json.load(handle)
    records = cache["records"]
    # Capture provenance before measurements, including when another process
    # subsequently edits a source file during an independent baseline run.
    paths = ["adapters/low_cost_mocap/tracking_pipeline.py", "adapters/low_cost_mocap/multi_body.py",
             "adapters/low_cost_mocap/low-cost-mocap/computer_code/api/helpers.py",
             "adapters/low_cost_mocap/low-cost-mocap/computer_code/api/KalmanFilter.py",
             "adapters/low_cost_mocap/low-cost-mocap/computer_code/api/LowPassFilter.py",
             "adapters/low_cost_mocap/multi_view.py"]
    hashes = {path: sha256(os.path.join(ROOT, path)) for path in paths if os.path.isfile(os.path.join(ROOT, path))}
    script_hash = sha256(__file__)

    def make_pipeline():
        pipeline = TrackingPipeline(os.path.join(args.rig_dir, "cameras_calib.json"),
                                     os.path.join(args.rig_dir, "rig_blender.json"), fps=cache["summary"]["fps"],
                                     centroid_mode=cache["summary"]["configuration"].get("centroid_mode", "integer"))
        if pipeline.configuration != cache["summary"]["configuration"]:
            raise ValueError("reference and runtime tracking parameters must agree")
        if pipeline.geometry_sha256 != cache["summary"]["geometry_sha256"]:
            raise ValueError("reference and runtime geometry must agree")
        return pipeline

    warmup = make_pipeline()
    for record in records[:10]:
        warmup.process_detections(record["detections"], record["t"], record["valid_cameras"])
    timings, passes, deltas, failures = [], [], {}, []
    exact = equivalent = 0
    for repeat in range(args.repeats):
        pipeline = make_pipeline()
        sample_times, output_hash = [], hashlib.sha256()
        for index, record in enumerate(records):
            start = time.perf_counter()
            output = pipeline.process_detections(record["detections"], record["t"], record["valid_cameras"])
            elapsed = (time.perf_counter() - start) * 1000
            sample_times.append(elapsed)
            actual = serializable_result(output)
            expected = {key: record[key] for key in actual}
            encoded = json_bytes(actual)
            exact += int(encoded == json_bytes(expected))
            errors = numeric_differences(actual, expected)
            equivalent += int(not errors)
            if errors and len(failures) < 10:
                failures.append({"repeat": repeat, "frame": record["frame"], "differences": errors[:12]})
            update_deltas(deltas, actual, expected)
            output_hash.update(encoded + b"\n")
            if index % 100 == 0 or index == len(records) - 1:
                print(f"pass {repeat + 1}, frame {record['frame']}: {elapsed:.3f} ms, equivalent={not errors}", flush=True)
        passes.append(dict(timing_summary(sample_times, args.target_fps), output_sha256=output_hash.hexdigest()))
        timings.extend(sample_times)
    summary = timing_summary(timings, args.target_fps)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "backend": args.backend,
              "scope": "centre-only service time from actual video points; excludes decode, network and UI",
              "fps_capture": cache["summary"]["fps"], "target_fps_budget_only": args.target_fps,
              "warmup_frames": min(10, len(records)), "frames_per_pass": len(records),
              "repeats": args.repeats, "timing": summary, "passes": passes,
              "comparison": {"frames": len(timings), "exact_frames": exact, "equivalent_frames": equivalent,
                             "numeric_atol": 1e-8, "numeric_rtol": 1e-7,
                             "note": "identities, ordering, availability and metadata exact; float leaves tolerate only roundoff",
                             "max_deltas": deltas, "first_failures": failures},
              "configuration": pipeline.configuration, "geometry_sha256": pipeline.geometry_sha256,
              "cache_sha256": sha256(args.cache), "algorithm_sha256": hashes, "script_sha256": script_hash,
              "environment": {"cpu": args.cpu_label, "platform": platform.platform(),
                              "python": platform.python_version(), "opencv": cv.__version__,
                              "numpy": np.__version__, "opencv_threads": cv.getNumThreads()},
              "per_frame_ms": timings}
    if args.baseline_report:
        with open(args.baseline_report, encoding="utf-8") as handle:
            baseline = json.load(handle)
        if (baseline["cache_sha256"] != report["cache_sha256"] or
                baseline["configuration"] != report["configuration"] or
                baseline["environment"] != report["environment"]):
            raise ValueError("baseline timing needs identical input/configuration/environment")
        report["baseline_comparison"] = {
            "sha256": sha256(args.baseline_report),
            "mean_speedup": baseline["timing"]["mean_ms"] / summary["mean_ms"],
            "p95_speedup": baseline["timing"]["p95_ms"] / summary["p95_ms"],
            "p99_speedup": baseline["timing"]["p99_ms"] / summary["p99_ms"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"mean={summary['mean_ms']:.3f} ms, P95={summary['p95_ms']:.3f} ms, "
          f"P99={summary['p99_ms']:.3f} ms; equivalent {equivalent}/{len(timings)}", flush=True)
    print("wrote", args.out, flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
