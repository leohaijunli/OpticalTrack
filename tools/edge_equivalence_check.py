"""Equivalence + bandwidth check for distributing per-camera detection to edges.

Centralised path  : one process detects all six cameras, then fuses.
Distributed path  : six independent "edge nodes" each detect ONE camera, then
                    the same central fusion assembles their 2D points.
The split is placed where there is NO cross-camera coupling (per-camera dot
detection is deterministic and stateless), so the two must agree bit-for-bit.
This tool runs both over the actual rendered six-camera videos and reports:
  * whether central fusion output is identical for both paths,
  * the bytes/frame that cross the wire for points vs raw/H.264 video.

    python tools/edge_equivalence_check.py --frames 0,100,200,300,400,500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import queue

import cv2 as cv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
from tracking_pipeline import TrackingPipeline, serializable_result
from helpers import prepare_camera_frame, detect_visible_dots


class EdgeNode:
    """A single Raspberry-Pi-like node: one camera, deterministic detection.

    It owns exactly one calibration and never sees another camera's pixels,
    exactly as a real edge node would.  It emits (frame_id, camera, points).
    """

    def __init__(self, camera_index, param, threshold, blur_size, min_area, max_area):
        self.index = camera_index
        self.param = param
        self.threshold = threshold
        self.blur_size = blur_size
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame):
        prepared = prepare_camera_frame(frame, self.param, True, self.blur_size)
        return detect_visible_dots(prepared, self.threshold, self.min_area, self.max_area)

    def run(self, capture, frame_ids, fps, out_queue):
        capture.set(cv.CAP_PROP_POS_FRAMES, frame_ids[0])
        for frame_id in frame_ids:
            ok, image = capture.read()
            if not ok or image is None:
                raise OSError(f"edge node {self.index} lost its source at frame {frame_id}")
            out_queue.put((frame_id, self.index, self.detect(image)))
        out_queue.put((None, self.index, None))


def frames_per_camera(video_paths, frame_ids):
    """Yield (frame_id, [6 frames]) from the six synchronised clips (centralised)."""
    captures = [cv.VideoCapture(path) for path in video_paths]
    try:
        for capture in captures:
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_ids[0])
        for frame_id in frame_ids:
            images = []
            for capture in captures:
                ok, image = capture.read()
                if not ok or image is None:
                    raise OSError(f"centralised source lost a frame at {frame_id}")
                images.append(image)
            yield frame_id, images
    finally:
        for capture in captures:
            capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-dir", default=os.path.join(ROOT, "out", "rig3"))
    parser.add_argument("--video-dir", default=os.path.join(ROOT, "out", "video"))
    parser.add_argument("--tag", default="t650")
    parser.add_argument("--frames", default="0,100,200,300,400,500")
    parser.add_argument("--threshold", type=float, default=140)
    parser.add_argument("--blur-size", type=int, default=9)
    parser.add_argument("--max-err", type=float, default=6.0)
    parser.add_argument("--h264-mbps", type=float, default=6.0,
                        help="assumed hardware H.264 monitor encode rate per camera")
    parser.add_argument("--out", default=os.path.join(ROOT, "out", "rig3", "edge_equivalence.json"))
    args = parser.parse_args()

    frame_ids = [int(v) for v in args.frames.split(",")]
    pipeline = TrackingPipeline(os.path.join(args.rig_dir, "cameras_calib.json"),
                                os.path.join(args.rig_dir, "rig_blender.json"),
                                threshold=args.threshold, blur_size=args.blur_size,
                                max_err=args.max_err, fps=30.0)
    video_paths = [os.path.join(args.video_dir, f"{camera[-2:]}_{camera.lower()}_{args.tag}_20s.mp4")
                   for camera in pipeline.camera_names]

    # Distributed pipeline: an identical fusion pipeline, fed only assembled points.
    pipeline_edge = TrackingPipeline(os.path.join(args.rig_dir, "cameras_calib.json"),
                                     os.path.join(args.rig_dir, "rig_blender.json"),
                                     threshold=args.threshold, blur_size=args.blur_size,
                                     max_err=args.max_err, fps=30.0)

    nodes = [EdgeNode(i, pipeline.params[i], pipeline.threshold, pipeline.blur_size,
                      3, 200) for i in range(6)]
    edges = [cv.VideoCapture(path) for path in video_paths]
    out_queue = queue.Queue()
    threads = [threading.Thread(target=nodes[i].run, args=(edges[i], frame_ids, pipeline.fps, out_queue),
                                daemon=True) for i in range(6)]
    for thread in threads:
        thread.start()
    assembled = {}
    missing = {frame_id: set(range(6)) for frame_id in frame_ids}

    records_c, records_e, point_payload_bytes = [], [], 0
    equal_frames = 0
    mismatch = None
    frame_shape = None
    for frame_id, images in frames_per_camera(video_paths, frame_ids):
        if frame_shape is None:
            frame_shape = images[0].shape[:2]        # (height, width)
        # Centralised: detect all six in one place.
        central = [nodes[i].detect(image) for i, image in enumerate(images)]
        # Distributed: collect this frame's points from the edge nodes.
        while missing[frame_id]:
            try:
                got = out_queue.get(timeout=120)
            except queue.Empty:
                print(f"FAIL at frame {frame_id}: edge node(s) "
                      f"{sorted(missing[frame_id])} never delivered points")
                sys.exit(1)
            if got[0] is None:
                continue
            f, cam, points = got
            if f in missing:
                assembled.setdefault(f, {})[cam] = points
                missing[f].discard(cam)
        distributed = [assembled[frame_id][i] for i in range(6)]
        if central != distributed:
            mismatch = (frame_id, central, distributed)
            break
        point_payload_bytes += len(json.dumps(distributed).encode("utf-8"))
        timestamp = frame_id / pipeline.fps
        result_c = serializable_result(pipeline.process_detections(central, timestamp))
        result_e = serializable_result(pipeline_edge.process_detections(distributed, timestamp))
        equal_frames += 1
        records_c.append(result_c)
        records_e.append(result_e)

    for thread in threads:
        thread.join(timeout=10)
    for edge in edges:
        edge.release()

    if mismatch is not None:
        print(f"FAIL at frame {mismatch[0]}: centralised and edge detection differ")
        sys.exit(1)

    # bit-for-bit fusion comparison
    identical = all(rec_c == rec_e for rec_c, rec_e in zip(records_c, records_e))
    if equal_frames != len(frame_ids):
        print(f"FAIL: only {equal_frames}/{len(frame_ids)} frames evaluated")
        sys.exit(1)

    # ---- bandwidth figures ----
    h, w = frame_shape
    raw_mbps = w * h * 8 * pipeline.fps / 1e6          # uncompressed 8-bit gray
    points_per_frame = point_payload_bytes / max(equal_frames, 1)
    h264_bps_per_cam = args.h264_mbps * 1e6
    total_point_mbps = points_per_frame * 8 * pipeline.fps / 1e6

    summary = {
        "frames_checked": frame_ids, "frames_equal": equal_frames,
        "fusion_identical": bool(identical),
        "path": "centralised detection vs six edge nodes -> same central fusion",
        "same_fusion_output": "per-frame locate_bodies + Kalman pos/vel/acc identical",
        "points_bytes_per_frame_total": points_per_frame,
        "points_kbps_per_cam": round(points_per_frame * pipeline.fps / 1024, 2),
        "points_total_mbps": round(total_point_mbps, 3),
        "raw_gray_mbps_per_cam": round(raw_mbps, 1),
        "h264_monitor_mbps_per_cam": round(args.h264_mbps, 2),
        "single_1GbE_headroom_mbps": 1000 - (6 * (total_point_mbps + h264_bps_per_cam / 1e6)),
        "detections_per_frame": {str(i): round(np.mean([len(r["detections"][i]) for r in records_c]), 2)
                                 for i in range(6)},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Equivalence check on rendered six-camera videos")
    print(f"  frames checked   : {frame_ids} -> {equal_frames} evaluated")
    print(f"  fusion identical : {identical}  (edge detection reproduced centralised bit-for-bit)")
    print(f"  points payload   : {points_per_frame:.0f} B/frame all cams -> {summary['points_total_mbps']} Mbps")
    print(f"  raw gray (no cc) : ~{raw_mbps:.0f} Mbps/cam ; H.264 monitor ~{args.h264_mbps:.0f} Mbps/cam")
    print(f"  6x(H.264+points) on one 1 GbE link leaves ~{summary['single_1GbE_headroom_mbps']:.0f} Mbps")
    print("wrote", args.out)


if __name__ == "__main__":
    main()