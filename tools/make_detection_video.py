"""Render a "detection video" with TRUE, RAW-DETECTED and FUSED (triangulated)
marker positions overlaid on the rendered footage.

For every frame we run the full Low-Cost-Mocap pipeline:
  1. per-camera blob detection (visible-light brightness mode),
  2. cross-camera epipolar matching + DLT triangulation -> fused 3-D points,
  3. reproject every fused 3-D point back into each camera.

Per panel the overlay is:
  * green filled circle  = TRUE marker (ground truth 2-D projection)
  * red   circle + cross = RAW detected blob in that camera
  * cyan  filled diamond = FUSED 3-D point reprojected into this camera
The title bar shows the number of fused 3-D points for that frame.

Detection / fusion use the exact same code path and settings as the adapter
(helpers.Cameras + find_point_correspondance_and_object_points), so the video
is what the Low-Cost-Mocap pipeline would produce live.

Output: a 6-panel mosaic video (write PNG frames then ffmpeg-encode).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import cv2 as cv


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FRAMES = os.path.join(ROOT, "out", "frames")
DET = os.path.join(ROOT, "out", "rig", "detections_2d.csv")
RIG = os.path.join(ROOT, "out", "rig", "rig_blender.json")
CAMS = ["CAM01", "CAM02", "CAM03", "CAM04", "CAM05", "CAM06"]
COLORS = [(255, 122, 89), (255, 194, 71), (126, 231, 135),
          (89, 211, 255), (167, 139, 250), (244, 114, 182)]

# reuse the patched Low-Cost-Mocap helpers for detection + triangulation
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap",
                                "low-cost-mocap", "computer_code", "api"))
from helpers import Cameras, find_point_correspondance_and_object_points  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=140.0)
    ap.add_argument("--min-area", type=float, default=3.0)
    ap.add_argument("--max-area", type=float, default=200.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--frames", default="0-599")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "video", "detections_6cam_fused.mp4"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--fused-keep", type=int, default=4,
                    help="keep only the N lowest-reprojection-error fused points per frame")
    args = ap.parse_args()

    a, b = args.frames.split("-")
    f0, f1 = int(a), int(b)
    count = f1 - f0 + 1

    # ---- calibration: cameras in order CAM01..CAM06 -----------------------
    rig = json.load(open(RIG))
    cal = json.load(open(os.path.join(ROOT, "out", "rig", "cameras_calib.json")))
    calib = []
    for cam, cc in zip(rig["cameras"], cal["cameras"]):
        R = np.array(cam["extrinsics_cv"]["R"], dtype=float)
        t = np.array(cam["extrinsics_cv"]["t"], dtype=float).reshape(3, 1)
        K = np.array(cc["K"], dtype=float)
        calib.append({"R": R, "t": t, "K": K})
    poses = [{"R": c["R"], "t": c["t"]} for c in calib]

    # ---- true 2-D positions: frame -> cam_name -> [[u,v],...] -------------
    true = {}
    with open(DET, newline="") as fh:
        for r in csv.DictReader(fh):
            fr = int(r["frame"])
            if fr < f0 or fr > f1:
                continue
            true.setdefault(fr, {}).setdefault(r["cam_name"], []).append(
                [float(r["u_px"]), float(r["v_px"])])

    # ---- detection config (keep in step with the adapter) ------------------
    Cameras.instance()  # instantiate (reads camera-params.json)
    _Cam = Cameras.instance()

    def detect_bgr(img_bgr):
        grey = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(grey, args.threshold, 255, cv.THRESH_BINARY)
        n, _, stats, cent = cv.connectedComponentsWithStats(bw, 8)
        pts = []
        for i in range(1, n):
            a = stats[i, cv.CC_STAT_AREA]
            if a < args.min_area or a > args.max_area:
                continue
            pts.append([float(cent[i, 0]), float(cent[i, 1])])
        return pts

    pw, ph = int(1280 * args.scale), int(800 * args.scale)
    mosaic_w, mosaic_h = pw * 6, ph

    outdir = args.outdir
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        vw = None
    else:
        vw = cv.VideoWriter(args.out, cv.VideoWriter_fourcc(*"MJPG"), 30.0,
                            (mosaic_w, mosaic_h))

    title_h = 34
    for fr in range(f0, f1 + 1):
        # load the 6 full-res panels
        full = []
        for cam in CAMS:
            p = os.path.join(FRAMES, f"{cam}_vis", f"f{fr:04d}.png")
            g = cv.imread(p, cv.IMREAD_GRAYSCALE)
            full.append(None if g is None else g)

        # ---- full pipeline: detect -> match -> triangulate -> reproject ----
        fused_reproj = [[] for _ in range(6)]  # per-camera list of pixel pts
        fused_count = 0
        if all(f is not None for f in full):
            image_points = []
            for g in full:
                rgb = cv.cvtColor(g, cv.COLOR_GRAY2BGR)
                image_points.append(detect_bgr(rgb) or [[None, None]])
            try:
                errs, obj, _ = find_point_correspondance_and_object_points(
                    [list(ip) for ip in image_points], poses,
                    [cv.cvtColor(g, cv.COLOR_GRAY2BGR) for g in full])
                # keep only the N best fused points (lowest reprojection error);
                # the raw matcher emits many spurious combinations from noise.
                fused_pts = []
                for (X, e) in zip(obj, errs):
                    if X is None or not np.all(np.isfinite(X)):
                        continue
                    fused_pts.append((float(e), X))
                fused_pts.sort(key=lambda t: t[0])
                n_keep = max(1, int(args.fused_keep)) if args.fused_keep else len(fused_pts)
                fused_pts = fused_pts[:n_keep]
                for (e, X) in fused_pts:
                    Xh = np.array([X[0], X[1], X[2], 1.0])
                    for ci in range(6):
                        P = calib[ci]["K"] @ np.hstack([calib[ci]["R"], calib[ci]["t"]])
                        uvw = P @ Xh
                        if abs(uvw[2]) < 1e-9:
                            continue
                        u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
                        if 0 <= u < 1280 and 0 <= v < 800:
                            fused_reproj[ci].append([float(u), float(v)])
                fused_count = len(fused_pts)
            except Exception as e:
                print("  f%d triangulation err: %s" % (fr, e), flush=True)

        # ---- assemble canvas ----
        canvas = np.zeros((mosaic_h + title_h, mosaic_w, 3), np.uint8)
        cv.putText(canvas, "LOW-COST-MOCAP  detection+fused  |  frame %04d  t=%5.2fs  fused:%d"
                   % (fr, fr / 30.0, fused_count), (12, title_h - 9),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
        cv.rectangle(canvas, (0, 0), (mosaic_w, title_h), (40, 48, 60), -1)
        cv.line(canvas, (0, title_h), (mosaic_w, title_h), (90, 110, 140), 1)

        for ci, cam in enumerate(CAMS):
            x0 = ci * pw
            if full[ci] is None:
                continue
            small = cv.resize(full[ci], (pw, ph))
            panel = cv.cvtColor(small, cv.COLOR_GRAY2BGR)

            det = detect_bgr(cv.cvtColor(small, cv.COLOR_GRAY2BGR))

            cv.rectangle(panel, (0, 0), (pw, 20), COLORS[ci], -1)
            cv.putText(panel, cam, (6, 14), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (8, 12, 20), 1, cv.LINE_AA)
            cv.putText(panel, "true:%d raw:%d fused:%d"
                       % (len(true.get(fr, {}).get(cam, [])), len(det), len(fused_reproj[ci])),
                       (pw - 8, 14), cv.FONT_HERSHEY_SIMPLEX, 0.42,
                       (220, 220, 235), 1, cv.LINE_AA)

            # raw detections (red)
            for (dx, dy) in det:
                cv.circle(panel, (int(dx), int(dy)), 8, (0, 0, 255), 2)
                cv.line(panel, (int(dx) - 8, int(dy)), (int(dx) + 8, int(dy)), (0, 0, 255), 2)
                cv.line(panel, (int(dx), int(dy) - 8), (int(dx), int(dy) + 8), (0, 0, 255), 2)

            # true markers (green)
            for (tu, tv) in true.get(fr, {}).get(cam, []):
                cx, cy = int(tu * args.scale), int(tv * args.scale)
                cv.circle(panel, (cx, cy), 5, (0, 255, 0), -1)
                cv.circle(panel, (cx, cy), 10, (0, 255, 0), 1)

            # fused reprojected 3-D points (cyan diamond), on top
            for (u, v) in fused_reproj[ci]:
                cx, cy = int(u * args.scale), int(v * args.scale)
                r = 7
                cv.fillConvexPoly(panel, np.array([[cx, cy - r], [cx + r, cy],
                                                   [cx, cy + r], [cx - r, cy]]),
                                  (0, 255, 255))
                cv.polylines(panel, [np.array([[cx, cy - r], [cx + r, cy],
                                               [cx, cy + r], [cx - r, cy]])],
                             True, (0, 0, 0), 1)

            canvas[title_h:, x0:x0 + pw] = panel

        if outdir:
            cv.imwrite(os.path.join(outdir, f"d{fr:04d}.png"), canvas)
        else:
            vw.write(canvas)
        if (fr - f0) % 60 == 0:
            print("  frame %04d / %04d  fused=%d" % (fr, f1, fused_count), flush=True)

    if vw is not None:
        vw.release()
    print("wrote", args.out, "(%dx%d, %d frames)" % (mosaic_w, mosaic_h, count))


if __name__ == "__main__":
    main()