"""Six camera views, per-body error curves and truth/Kalman XY trajectories.

Uses the actual-pixel evaluation cache; never substitutes truth for tracking.
All graphs draw into bounded ROIs, so lines cannot cover labels/other panels.
Missing observations break curves instead of being silently interpolated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import cv2 as cv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
from tracking_pipeline import load_calibration, project
from evaluate_multi_video import sha256, video_frames

W, H = 2600, 900
PW, PH, GAP, MARGIN = 512, 320, 8, 16
GRID_W = 3 * PW + 2 * GAP
RX, RW = MARGIN + GRID_W + MARGIN, 1000
BG, PANEL, DIM, FG = (17, 20, 27), (27, 32, 42), (125, 137, 153), (226, 233, 241)
COLORS = [(115, 228, 135), (77, 191, 255), (251, 184, 101),
          (205, 140, 250), (250, 130, 120), (125, 223, 238)]
FONT = cv.FONT_HERSHEY_SIMPLEX


def text(image, value, x, y, scale=0.5, color=FG, thickness=1):
    cv.putText(image, str(value), (int(x), int(y)), FONT, scale, color, thickness, cv.LINE_AA)


def nice_upper(value):
    value = max(float(value) * 1.12, 10.0)
    unit = 10 ** math.floor(math.log10(value))
    return math.ceil(value / unit) * unit


def trace(image, points, color, width=1, dashed=False):
    """Draw only adjacent valid samples, retaining gaps and panel clipping."""
    segment = []
    def flush():
        if len(segment) > 1:
            cv.polylines(image, [np.asarray(segment, np.int32)], False, color, width, cv.LINE_AA)
        segment.clear()
    for index, point in enumerate(points):
        if point is None or (dashed and index % 10 >= 6):
            flush()
        else:
            segment.append(point)
    flush()


def draw_error(canvas, records, current, body, color, x, y, width=1000, height=164):
    tile = canvas[y:y + height, x:x + width]
    tile[:] = PANEL
    metrics = records[current]["errors"][str(body["body_id"])]
    raw = "LOST" if metrics["raw"] is None else f"{metrics['raw']:.1f}"
    kf = "LOST" if metrics["kf"] is None else f"{metrics['kf']:.1f}"
    text(tile, body["name"] + " | position error (mm)", 12, 23, 0.59, color, 1)
    text(tile, f"raw {raw}   Kalman {kf}", 560, 23, 0.52)
    plot_x, plot_y, plot_w, plot_h = 63, 39, width - 83, 96
    plot = tile[plot_y:plot_y + plot_h, plot_x:plot_x + plot_w]
    plot[:] = (20, 25, 33)
    values = [record["errors"][str(body["body_id"])][key]
              for record in records[:current + 1] for key in ("raw", "kf")]
    ymax = nice_upper(max((v for v in values if v is not None), default=0))
    for fraction in (0, 0.5, 1):
        py = int((1 - fraction) * (plot_h - 1))
        cv.line(plot, (0, py), (plot_w - 1, py), (48, 56, 67), 1)
        label = f"{fraction * ymax:g}"
        text(tile, label, 5, plot_y + py + 4, 0.40, DIM)
    first, last = records[0]["t"], records[-1]["t"]
    duration = max(last - first, 1 / 30)
    for fraction in np.linspace(0, 1, 5):
        px = int(fraction * (plot_w - 1))
        cv.line(plot, (px, 0), (px, plot_h - 1), (39, 47, 58), 1)
        text(tile, f"{first + fraction * duration:.1f}s", plot_x + px - 14,
             height - 10, 0.37, DIM)
    for key, series_color, thickness in (("raw", (149, 156, 169), 1), ("kf", color, 2)):
        points = []
        for record in records[:current + 1]:
            value = record["errors"][str(body["body_id"])][key]
            if value is None:
                points.append(None)
            else:
                points.append((int((record["t"] - first) / duration * (plot_w - 1)),
                               int((1 - value / ymax) * (plot_h - 1))))
        trace(plot, points, series_color, thickness)
        if points and points[-1] is not None:
            cv.circle(plot, points[-1], 3, series_color, -1, cv.LINE_AA)


def draw_trajectory(canvas, records, current, bodies):
    x, y, width, height = RX, 592, RW, 288
    tile = canvas[y:y + height, x:x + width]
    tile[:] = PANEL
    text(tile, "XY trajectory | truth dashed / Kalman solid | ENU metres", 12, 24, 0.52)
    all_points = []
    for record in records:
        all_points.extend(np.asarray(point)[:2] for point in record["reference"].values())
        all_points.extend(np.asarray(obj["pos"])[:2] for obj in record["filtered_objects"])
    array = np.asarray(all_points)
    lower, upper = array.min(axis=0), array.max(axis=0)
    span = np.maximum(upper - lower, [4.8, 3.0]) * 1.15
    middle = (upper + lower) * 0.5
    lower, upper = middle - span * 0.5, middle + span * 0.5
    px, py, pw, ph = 65, 42, width - 84, height - 76
    plot = tile[py:py + ph, px:px + pw]
    plot[:] = (20, 25, 33)
    def screen(point):
        v = (np.asarray(point)[:2] - lower) / (upper - lower)
        return (int(v[0] * (pw - 1)), int((1 - v[1]) * (ph - 1)))
    for fraction in np.linspace(0, 1, 5):
        vx, vy = int(fraction * (pw - 1)), int((1 - fraction) * (ph - 1))
        cv.line(plot, (vx, 0), (vx, ph - 1), (43, 52, 63), 1)
        cv.line(plot, (0, vy), (pw - 1, vy), (43, 52, 63), 1)
        text(tile, f"{lower[0] + fraction * span[0]:.1f}", px + vx - 12, height - 11, .39, DIM)
        text(tile, f"{lower[1] + fraction * span[1]:.1f}", 10, py + vy + 4, .39, DIM)
    for body_index, body in enumerate(bodies):
        bid, color = body["body_id"], COLORS[body_index % len(COLORS)]
        truth = [screen(record["reference"][str(bid)]) for record in records[:current + 1]]
        estimate = []
        for record in records[:current + 1]:
            obj = next((obj for obj in record["filtered_objects"] if obj["droneIndex"] == bid), None)
            estimate.append(screen(obj["pos"]) if obj is not None else None)
        trace(plot, truth, tuple(int(c * .65) for c in color), 1, dashed=True)
        trace(plot, estimate, color, 2)
        if estimate and estimate[-1] is not None:
            cv.circle(plot, estimate[-1], 4, color, -1, cv.LINE_AA)
        cv.circle(plot, truth[-1], 5, color, 1, cv.LINE_AA)


def draw_readouts(canvas, record, summary):
    tile = canvas[716:880, MARGIN:MARGIN + GRID_W]
    tile[:] = PANEL
    reference = ("BODY ORIGIN (0, 0, 0) | geometric center reference"
                 if summary.get("position_reference_key") == "body_origin" else "declared base-marker midpoint")
    text(tile, "POSITION REFERENCE: " + reference, 12, 22, .52)
    text(tile, "body", 12, 50, .47, DIM)
    text(tile, "truth XYZ (m)", 146, 50, .47, DIM)
    text(tile, "Kalman XYZ (m)", 525, 50, .47, DIM)
    text(tile, "raw / KF error (mm)", 919, 50, .47, DIM)
    for index, body in enumerate(summary["bodies"]):
        bid, y = body["body_id"], 78 + index * 27
        color = COLORS[index % len(COLORS)]
        ref = record["reference"][str(bid)]
        kf = next((obj for obj in record["filtered_objects"] if obj["droneIndex"] == bid), None)
        metrics = record["errors"][str(bid)]
        text(tile, body["name"], 12, y, .52, color)
        text(tile, "  ".join(f"{v:+.3f}" for v in ref), 146, y, .5)
        text(tile, "LOST" if kf is None else "  ".join(f"{v:+.3f}" for v in kf["pos"]), 525, y, .5)
        text(tile, " / ".join("LOST" if metrics[key] is None else f"{metrics[key]:.1f}"
                              for key in ("raw", "kf")), 919, y, .52, color)
        text(tile, f"coverage {body['raw_coverage_percent']:.1f}%", 1190, y, .44, DIM)
    text(tile, "4th marker: rigid identity / 6-DOF.  When hidden: recent pose resolves base order.  Gaps = no solution.",
         12, 153, .43, DIM)


def draw_frame(images, record, records, current, summary, poses):
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = BG
    text(canvas, f"LOW-COST-MOCAP | TAROT IRONMAN 650 x3 | frame {record['frame']:04d}  t={record['t']:5.2f}s",
         MARGIN, 31, .70)
    text(canvas, "raw dots: red  |  truth reference: white  |  fused body: colour", RX, 31, .49, DIM)
    names = summary["camera_names"]
    body_colors = {body["body_id"]: COLORS[index % len(COLORS)]
                   for index, body in enumerate(summary["bodies"])}
    for ci, image in enumerate(images):
        small = cv.resize(image, (PW, PH), interpolation=cv.INTER_AREA)
        panel = cv.cvtColor(small, cv.COLOR_GRAY2BGR) if small.ndim == 2 else small.copy()
        scale_x, scale_y = PW / image.shape[1], PH / image.shape[0]
        def screen(uv):
            return (int(round(uv[0] * scale_x)), int(round(uv[1] * scale_y)))
        def visible(uv):
            return uv is not None and 0 <= uv[0] < image.shape[1] and 0 <= uv[1] < image.shape[0]
        for uv in record["detections"][ci]:
            cv.drawMarker(panel, screen(uv), (70, 70, 250), cv.MARKER_CROSS, 7, 1, cv.LINE_AA)
        for body in summary["bodies"]:
            uv = project(poses[ci], record["reference"][str(body["body_id"])])
            if visible(uv):
                cv.circle(panel, screen(uv), 5, (250, 250, 250), 1, cv.LINE_AA)
        for obj in record["objects"]:
            uv = project(poses[ci], obj["pos"])
            if visible(uv):
                p, color = screen(uv), body_colors[obj["droneIndex"]]
                cv.drawMarker(panel, p, color, cv.MARKER_DIAMOND, 11, 2, cv.LINE_AA)
                text(panel, f"T{obj['droneIndex']}", p[0] + 7, p[1] + 2, .35, color)
        cv.rectangle(panel, (0, 0), (PW - 1, 24), (31, 40, 54), -1)
        text(panel, names[ci], 8, 17, .48)
        text(panel, f"dots {len(record['detections'][ci])} | bodies {len(record['objects'])}", 205, 17, .4, DIM)
        x, y = MARGIN + ci % 3 * (PW + GAP), 52 + ci // 3 * (PH + GAP)
        canvas[y:y + PH, x:x + PW] = panel
    for index, body in enumerate(summary["bodies"]):
        draw_error(canvas, records, current, body, COLORS[index % len(COLORS)], RX, 52 + index * 176)
    draw_trajectory(canvas, records, current, summary["bodies"])
    draw_readouts(canvas, record, summary)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=os.path.join(ROOT, "out", "rig3", "tracking_from_video.json"))
    parser.add_argument("--rig-dir", default=os.path.join(ROOT, "out", "rig3"))
    parser.add_argument("--out", default=os.path.join(ROOT, "out", "video", "fused_tracking_error_multibody.mp4"))
    parser.add_argument("--preview", default=os.path.join(ROOT, "out", "viz", "fused_tracking_error_multibody_f0300.png"))
    args = parser.parse_args()
    with open(args.cache, encoding="utf-8") as handle:
        data = json.load(handle)
    summary, records = data["summary"], data["records"]
    calibration, _, poses = load_calibration(os.path.join(args.rig_dir, "cameras_calib.json"))
    expected_calibration = summary.get("geometry_sha256", {}).get("calibration")
    if expected_calibration and sha256(os.path.join(args.rig_dir, "cameras_calib.json")) != expected_calibration:
        raise ValueError("evaluation cache and plot calibration differ")
    summary["camera_names"] = [camera["name"] for camera in calibration["cameras"]]
    if len(poses) != 6 or len(summary["bodies"]) != 3:
        raise ValueError("this layout expects six cameras and three bodies")
    source_paths = summary["source_files"]
    if len(source_paths) != 6:
        raise ValueError("a clean-video evaluation cache is required")
    frame_ids = [record["frame"] for record in records]
    if not frame_ids or frame_ids != list(range(frame_ids[0], frame_ids[-1] + 1)):
        raise ValueError("a consecutive-frame cache is required for a constant-fps curve video")
    source_hashes = summary.get("provenance", {}).get("video_sha256", {})
    for path in source_paths:
        if source_hashes and sha256(path) != source_hashes.get(os.path.basename(path)):
            raise ValueError(f"source video changed since evaluation: {path}")
    captures = [cv.VideoCapture(path) for path in source_paths]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    partial = os.path.splitext(args.out)[0] + ".partial.mp4"
    encoder = None
    try:
        for capture in captures:
            if not capture.isOpened():
                raise OSError("cannot open a source clip")
        decoded_frames = video_frames(captures, frame_ids)
        command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-video_size", f"{W}x{H}", "-framerate", str(summary["fps"]),
                   "-i", "pipe:0", "-an", "-frames:v", str(len(records)),
                   "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-threads", "4",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", partial]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        preview_frame = min(records, key=lambda r: abs(r["frame"] - 300))["frame"]
        for index, record in enumerate(records):
            canvas = draw_frame(next(decoded_frames), record, records, index, summary, poses)
            if record["frame"] == preview_frame:
                os.makedirs(os.path.dirname(os.path.abspath(args.preview)), exist_ok=True)
                if not cv.imwrite(args.preview, canvas):
                    raise OSError("failed to write video preview")
            encoder.stdin.write(canvas.tobytes())
            if index % 100 == 0 or index + 1 == len(records):
                print(f"video frame {index + 1}/{len(records)}", flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", partial],
                               check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        if int(stream["nb_frames"]) != len(records) or stream["width"] != W or stream["height"] != H:
            raise RuntimeError("encoded video does not match the requested frames/layout")
        os.replace(partial, args.out)
        print("wrote", args.out, "|", W, "x", H, "|", len(records), "frames", flush=True)
    finally:
        for capture in captures:
            capture.release()
        if encoder is not None and encoder.poll() is None:
            if encoder.stdin is not None and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.terminate()
            encoder.wait()


if __name__ == "__main__":
    main()
