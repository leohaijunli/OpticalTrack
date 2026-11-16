r"""Run the same repository pipeline on clips, frame directories or live feeds.

Example (from the project root):
  .venv\Scripts\python.exe adapters/low_cost_mocap/run_tracking.py --frames 600

For real feeds, supply --sources and their own --calibration / --body-specs.
OpenCV's host-arrival timestamps do NOT establish hardware exposure sync.
No ground-truth files are read by this entry point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from tracking_pipeline import TrackingPipeline, serializable_result
from multi_body import POSITION_REFERENCES
from video_replay import VideoReplay, discover_clips, default_project_root


def main():
    root = Path(default_project_root())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", help="comma-separated sources in calibration camera order")
    parser.add_argument("--calibration", default=str(root / "out/rig3/cameras_calib.json"))
    parser.add_argument("--body-specs", default=str(root / "out/rig3/rig_blender.json"))
    parser.add_argument("--video-tag", default="t650")
    parser.add_argument("--frames", type=int, default=600, help="0 = until source ends / Ctrl+C")
    parser.add_argument("--output", default="-", help="NDJSON file, or - for stdout")
    parser.add_argument("--unpaced", action="store_true", help="offline replay, not realtime pacing")
    parser.add_argument("--max-skew-s", type=float, default=.010)
    parser.add_argument("--gate-m", type=float, default=0.5,
                        help="per-body motion gate (m); 0 disables temporal association")
    parser.add_argument("--extra-tol", type=float, default=0.05,
                        help="rigid-prediction budget for non-reference markers (m)")
    parser.add_argument("--max-err", type=float, default=6.0,
                        help="reference-triangle mean reprojection MSE limit (px^2)")
    parser.add_argument("--position-reference", choices=POSITION_REFERENCES, default="body_origin")
    parser.add_argument("--max-speed", type=float, default=10.0, help="velocity measurement magnitude limit (m/s)")
    parser.add_argument("--centroid-mode", choices=("integer", "subpixel"), default="integer")
    args = parser.parse_args()
    pipeline = TrackingPipeline(args.calibration, args.body_specs,
                                gate_m=args.gate_m, extra_tol=args.extra_tol,
                                max_err=args.max_err, position_reference=args.position_reference,
                                max_speed=args.max_speed, centroid_mode=args.centroid_mode)
    sources = args.sources or ",".join(discover_clips(tag=args.video_tag))
    source = VideoReplay(source=sources, realtime=False if args.unpaced else None, loop=False)
    if len(source) != len(pipeline.camera_names):
        source.release()
        raise ValueError("source count/order must match the calibrated cameras")
    output = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        index = 0
        while args.frames == 0 or index < args.frames:
            try:
                images, timestamps = source.read()
            except StopIteration:
                break
            valid = np.isfinite(timestamps)
            skew = float(np.ptp(timestamps[valid])) if np.any(valid) else None
            timestamp = float(np.median(timestamps[valid])) if np.any(valid) else index / pipeline.fps
            accepted = skew is not None and skew <= args.max_skew_s
            result = pipeline.process(images if accepted else [None] * len(images), timestamp)
            result.update({"batch": index, "capture_time_kind": (
                "frame_index" if all(kind in ("file", "dir") for kind in source.kinds) else "host_arrival"),
                "observed_skew_s": skew, "software_time_gate_passed": accepted})
            output.write(json.dumps(serializable_result(result), allow_nan=False) + "\n")
            output.flush()
            index += 1
    finally:
        source.release()
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    main()
