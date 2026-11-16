"""Verify the edge/centre split on independently decoded, correctly indexed video.

Six local worker processes each own one decoder and return JSON point packets.
The normal TrackingPipeline.process path is compared with process_detections,
including every position, velocity, acceleration, pose and identity field.
This is a software partition/serialization test, not a six-device network test.

    python tools/edge_equivalence_check.py --frames 0-599 --target-fps 60
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
import platform
import sys
import time

import cv2 as cv
import numpy as np
import scipy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
from tracking_pipeline import TrackingPipeline, load_calibration, serializable_result
from helpers import prepare_camera_frame, detect_visible_dots
from tools.evaluate_multi_video import sha256, video_frames, video_provenance


def parse_frames(text):
    if "-" in text:
        start, end = map(int, text.split("-"))
        frames = list(range(start, end + 1))
    else:
        frames = [int(value) for value in text.split(",")]
    if (not frames or frames[0] < 0 or
            any(b <= a for a, b in zip(frames, frames[1:]))):
        raise ValueError("frames must be nonnegative and strictly increasing")
    return frames


def json_bytes(value):
    return json.dumps(serializable_result(value), sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def timing_summary(samples, target_fps):
    values = np.asarray(samples, dtype=float)
    return {"count": int(values.size), "mean_ms": float(values.mean()),
            "p50_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(values.max()),
            "inverse_mean_hz": float(1000 / values.mean()),
            "over_target_budget_percent": float(100 * np.mean(values > 1000 / target_fps))}


def validate_capture(capture, path, frames, fps):
    if not capture.isOpened():
        raise OSError(f"cannot open {path}")
    if abs(capture.get(cv.CAP_PROP_FPS) - fps) > 1e-3:
        raise ValueError(f"source FPS mismatch: {path}; do not relabel 30 fps video as 60 fps")
    if int(capture.get(cv.CAP_PROP_FRAME_COUNT)) <= frames[-1]:
        raise ValueError(f"source is shorter than requested frame range: {path}")


def edge_video_packets(job):
    """One local process, one calibration slot, one video and its exact pixels."""
    # Limit nested OpenCV parallelism in the six simultaneous local workers.
    # These worker timings are explicitly NOT Raspberry Pi measurements.
    cv.setNumThreads(1)
    _, params, _ = load_calibration(job["calibration"])
    param = params[job["camera_index"]]
    capture = cv.VideoCapture(job["path"])
    packets, pixel_hashes, detect_ms = [], [], []
    config = job["detector"]
    try:
        validate_capture(capture, job["path"], job["frames"], job["fps"])
        shape = [int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
                 int(capture.get(cv.CAP_PROP_FRAME_WIDTH))]
        for frame_id, images in zip(job["frames"], video_frames([capture], job["frames"])):
            image = images[0]
            pixel_hashes.append(hashlib.sha256(image).hexdigest())
            start = time.perf_counter()
            prepared = prepare_camera_frame(image, param, True, config["blur_size"])
            points = detect_visible_dots(prepared, config["threshold"],
                                         config["min_area"], config["max_area"],
                                         config.get("centroid_mode", "integer"))
            detect_ms.append((time.perf_counter() - start) * 1000)
            packets.append(json_bytes({
                "schema": 1, "session_id": job["session_id"],
                "camera": job["camera"], "frame_id": frame_id,
                "exposure_midpoint_ns": round(frame_id * 1e9 / job["fps"]),
                "clock_domain": "simulation", "coordinate_space": "undistorted_pixels",
                "calibration_sha256": job["calibration_sha256"],
                "detector_sha256": job["detector_sha256"],
                "valid": True, "points": points}))
    finally:
        capture.release()
    return {"camera": job["camera"], "pid": os.getpid(), "shape": shape,
            "packets": packets, "pixel_hashes": pixel_hashes, "detect_ms": detect_ms}


def assemble_frame(packets, camera_names, frame_id, fps, expected):
    """Offline strict assembly; packet arrival order cannot select camera order.

    No timeout or loss policy is implied by this verifier. A live receiver must
    add bounded queues, deadlines, restart epochs and explicit missing-view state.
    """
    indexed = {}
    for raw in packets:
        packet = json.loads(raw)
        camera = packet.get("camera")
        if camera not in camera_names or camera in indexed:
            raise ValueError("unknown or duplicate camera packet")
        required = dict(expected, schema=1, frame_id=frame_id, valid=True,
                        exposure_midpoint_ns=round(frame_id * 1e9 / fps),
                        clock_domain="simulation", coordinate_space="undistorted_pixels")
        if any(packet.get(key) != value for key, value in required.items()):
            raise ValueError(f"packet epoch, frame, time or configuration mismatch: {camera}")
        points = packet.get("points")
        array = np.asarray(points, dtype=float)
        if points != [] and (array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all()):
            raise ValueError(f"invalid point coordinates: {camera}")
        indexed[camera] = points
    if set(indexed) != set(camera_names):
        raise ValueError("one packet per calibrated camera is required")
    return [indexed[camera] for camera in camera_names]


def bandwidth_summary(payload_bytes_by_camera, shapes, frame_count, fps, h264_mbps):
    # One JSON packet per camera per frame, IPv4 UDP, Ethernet without VLAN:
    # IP 20 + UDP 8 + Ethernet 14 + FCS 4 + preamble/SFD 8 + IFG 12 = 66 B.
    # This is a calculated wire budget, not a network capture.
    max_packet = max(max(sizes) for sizes in payload_bytes_by_camera)
    if max_packet > 1472:
        raise ValueError("point packet exceeds a 1500-byte IPv4/UDP MTU budget")
    payload = np.asarray([sum(sizes) / frame_count for sizes in payload_bytes_by_camera])
    payload_mbps = payload * 8 * fps / 1e6
    wire_mbps = (payload + 66) * 8 * fps / 1e6
    raw = np.asarray([h * w * 8 * fps / 1e6 for h, w in shapes])
    monitor_total = h264_mbps * len(shapes)
    return {"fps": fps, "max_point_packet_bytes": max_packet,
            "point_payload_kbps_per_camera": (payload_mbps * 1000).tolist(),
            "point_payload_total_mbps": float(payload_mbps.sum()),
            "point_wire_total_mbps": float(wire_mbps.sum()),
            "point_transport_assumption": "one UDP/IPv4 packet per camera frame; 66 B wire overhead; no VLAN",
            "raw_gray8_mbps_per_camera": raw.tolist(), "raw_gray8_total_mbps": float(raw.sum()),
            "h264_assumed_mbps_per_camera": h264_mbps,
            "h264_assumed_total_mbps": monitor_total,
            "combined_mbps_excluding_video_transport_overhead": float(monitor_total + wire_mbps.sum()),
            "nominal_1gbe_remaining_mbps": float(1000 - monitor_total - wire_mbps.sum())}


class TimedPipeline(TrackingPipeline):
    def process_detections(self, *args, **kwargs):
        start = time.perf_counter()
        result = super().process_detections(*args, **kwargs)
        self.last_fusion_ms = (time.perf_counter() - start) * 1000
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-dir", default=os.path.join(ROOT, "out", "rig3"))
    parser.add_argument("--video-dir", default=os.path.join(ROOT, "out", "video"))
    parser.add_argument("--tag", default="t650")
    parser.add_argument("--frames", default="0-599")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=60.0,
                        help="service-time/bandwidth target; source timestamps are never changed")
    parser.add_argument("--threshold", type=float, default=140.0)
    parser.add_argument("--blur-size", type=int, default=9)
    parser.add_argument("--centroid-mode", choices=("integer", "subpixel"), default="integer")
    parser.add_argument("--max-err", type=float, default=6.0)
    parser.add_argument("--h264-mbps", type=float, default=6.0,
                        help="assumed monitor bitrate per camera, independent of point FPS")
    parser.add_argument("--reference", help="optional full-sequence tracking JSON to compare")
    parser.add_argument("--cpu-label", default=platform.processor())
    parser.add_argument("--out", default=os.path.join(ROOT, "out", "rig3", "edge_equivalence.json"))
    args = parser.parse_args()
    try:
        frame_ids = parse_frames(args.frames)
        if any(not np.isfinite(value) or value <= 0 for value in
               (args.source_fps, args.target_fps, args.h264_mbps)):
            raise ValueError("frame rates and assumed video bitrate must be positive")
    except ValueError as error:
        parser.error(str(error))

    calibration_path = os.path.join(args.rig_dir, "cameras_calib.json")
    body_path = os.path.join(args.rig_dir, "rig_blender.json")
    pipeline = TimedPipeline(calibration_path, body_path, threshold=args.threshold,
                             blur_size=args.blur_size, max_err=args.max_err, fps=args.source_fps,
                             centroid_mode=args.centroid_mode)
    edge_pipeline = TimedPipeline(calibration_path, body_path, threshold=args.threshold,
                                  blur_size=args.blur_size, max_err=args.max_err, fps=args.source_fps,
                                  centroid_mode=args.centroid_mode)
    paths = [os.path.join(args.video_dir, f"{cam[-2:]}_{cam.lower()}_{args.tag}_20s.mp4")
             for cam in pipeline.camera_names]
    provenance = video_provenance(paths, os.path.join(args.video_dir, f"{args.tag}_manifest.json"),
                                  pipeline.geometry_sha256)
    detector = {key: pipeline.configuration[key] for key in
                ("threshold", "blur_size", "min_area", "max_area")}
    if args.centroid_mode != "integer":
        detector["centroid_mode"] = args.centroid_mode
    detector["helper_sha256"] = sha256(os.path.join(ROOT, "adapters", "low_cost_mocap",
                                                  "low-cost-mocap", "computer_code", "api", "helpers.py"))
    expected = {"session_id": "sim-" + hashlib.sha256(json_bytes(provenance)).hexdigest()[:16],
                "calibration_sha256": pipeline.geometry_sha256["calibration"],
                "detector_sha256": hashlib.sha256(json_bytes(detector)).hexdigest()}
    jobs = [dict(expected, path=path, camera=cam, camera_index=i,
                 calibration=calibration_path, detector=detector, fps=args.source_fps, frames=frame_ids)
            for i, (cam, path) in enumerate(zip(pipeline.camera_names, paths))]
    print(f"Independent local edge processes: {len(jobs)}, actual frames {frame_ids[0]}..{frame_ids[-1]}", flush=True)
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        workers = list(pool.map(edge_video_packets, jobs))
    print("Edge decoding complete; measuring the central pipeline without worker contention.", flush=True)

    reference = None
    if args.reference:
        with open(args.reference, encoding="utf-8") as handle:
            cached = json.load(handle)
        if (cached["summary"]["selected_frames"] != frame_ids or
                cached["summary"]["configuration"] != pipeline.configuration or
                cached["summary"]["geometry_sha256"] != pipeline.geometry_sha256):
            raise ValueError("reference sequence/configuration differs; temporal state is not comparable")
        reference = {record["frame"]: record for record in cached["records"]}

    captures = []
    timers = {key: [] for key in ("decode_six", "detect_six", "central_compute", "central_fusion",
                                 "edge_fusion", "json_assembly", "central_decode_and_compute")}
    pixel_equal = detection_equal = fusion_equal = reference_equal = 0
    mismatch = None
    hashes = {key: hashlib.sha256() for key in ("central", "edge")}
    detections = {cam: [] for cam in pipeline.camera_names}
    coverage = Counter()
    all_bodies = 0
    try:
        for path in paths:
            capture = cv.VideoCapture(path)
            captures.append(capture)
            validate_capture(capture, path, frame_ids, args.source_fps)
        decoded = video_frames(captures, frame_ids)
        for index, frame_id in enumerate(frame_ids):
            start = time.perf_counter()
            images = next(decoded)
            decode_ms = (time.perf_counter() - start) * 1000
            pixels_match = all(hashlib.sha256(image).hexdigest() == worker["pixel_hashes"][index]
                               for image, worker in zip(images, workers))
            pixel_equal += int(pixels_match)
            start = time.perf_counter()
            central = pipeline.process(images, frame_id / args.source_fps)
            compute_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            # Reverse delivery order deliberately; calibration order is restored by ID.
            points = assemble_frame([w["packets"][index] for w in reversed(workers)],
                                    pipeline.camera_names, frame_id, args.source_fps, expected)
            assembly_ms = (time.perf_counter() - start) * 1000
            distributed = edge_pipeline.process_detections(points, frame_id / args.source_fps)
            central_bytes, edge_bytes = json_bytes(central), json_bytes(distributed)
            hashes["central"].update(central_bytes + b"\n")
            hashes["edge"].update(edge_bytes + b"\n")
            detection_equal += int(central["detections"] == points)
            fusion_equal += int(central_bytes == edge_bytes)
            if reference is not None:
                ref_bytes = json_bytes({key: reference[frame_id][key] for key in central})
                reference_equal += int(ref_bytes == central_bytes)
            if mismatch is None and (not pixels_match or central_bytes != edge_bytes or
                                     (reference is not None and ref_bytes != central_bytes)):
                mismatch = {"frame": frame_id, "pixels_match": pixels_match,
                            "fusion_fields": [key for key in central
                                              if json_bytes(central[key]) != json_bytes(distributed[key])],
                            "reference_matches": None if reference is None else ref_bytes == central_bytes}
            timers["decode_six"].append(decode_ms)
            timers["central_compute"].append(compute_ms)
            timers["central_fusion"].append(pipeline.last_fusion_ms)
            timers["detect_six"].append(compute_ms - pipeline.last_fusion_ms)
            timers["edge_fusion"].append(edge_pipeline.last_fusion_ms)
            timers["json_assembly"].append(assembly_ms)
            timers["central_decode_and_compute"].append(decode_ms + compute_ms)
            for cam, found in zip(pipeline.camera_names, central["detections"]):
                detections[cam].append(len(found))
            found_bodies = [obj["droneIndex"] for obj in central["filtered_objects"]]
            coverage.update(found_bodies)
            all_bodies += int(len(found_bodies) == len(pipeline.body_ids))
            if index % 100 == 0 or index == len(frame_ids) - 1:
                print(f"frame {frame_id}: exact={central_bytes == edge_bytes}, fusion={edge_pipeline.last_fusion_ms:.2f} ms", flush=True)
    finally:
        for capture in captures:
            capture.release()

    sizes = [[len(raw) for raw in worker["packets"]] for worker in workers]
    shapes = [worker["shape"] for worker in workers]
    summary = {
        "schema": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "six local processes + JSON round-trip + normal TrackingPipeline; no network or ARM hardware",
        "supersedes": "the old six-frame result mislabeled sequential frames as sparse frames",
        "frames_checked": frame_ids, "frames_equal": fusion_equal,
        "source_fps": args.source_fps, "target_fps_budget_only": args.target_fps,
        "pixel_equal_frames": pixel_equal, "detection_equal_frames": detection_equal,
        "fusion_identical": fusion_equal == len(frame_ids),
        "first_mismatch": mismatch,
        "reference_equal_frames": None if reference is None else reference_equal,
        "reference_sha256": None if reference is None else sha256(args.reference),
        "output_sha256": {key: value.hexdigest() for key, value in hashes.items()},
        "configuration": pipeline.configuration, "geometry_sha256": pipeline.geometry_sha256,
        "provenance": provenance, "detector": detector, "packet_contract": expected,
        "script_sha256": sha256(__file__),
        "environment": {"cpu": args.cpu_label, "platform": platform.platform(),
                        "python": platform.python_version(), "opencv": cv.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "central_opencv_threads": cv.getNumThreads(), "edge_opencv_threads": 1,
                        "worker_pids": [worker["pid"] for worker in workers]},
        "timing_note": "all frames; no pacing; central runs after workers finish; excludes UI, live capture, network and encoding",
        "timing": {key: timing_summary(values, args.target_fps) for key, values in timers.items()},
        "edge_worker_timing_contention_not_hardware_benchmark": {
            worker["camera"]: timing_summary(worker["detect_ms"], args.target_fps) for worker in workers},
        "bandwidth_source_fps": bandwidth_summary(sizes, shapes, len(frame_ids), args.source_fps, args.h264_mbps),
        "bandwidth_target_fps_projection": bandwidth_summary(sizes, shapes, len(frame_ids), args.target_fps, args.h264_mbps),
        "detections_per_camera": {cam: {"min": min(n), "mean": float(np.mean(n)), "max": max(n)}
                                  for cam, n in detections.items()},
        "body_coverage_percent": {str(body): 100 * coverage[body] / len(frame_ids) for body in pipeline.body_ids},
        "all_bodies_available_percent": 100 * all_bodies / len(frame_ids)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Exact fusion: {fusion_equal}/{len(frame_ids)}; wrote {args.out}", flush=True)
    if mismatch is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
