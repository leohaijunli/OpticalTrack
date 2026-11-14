"""Multi-target detection video: truth / raw detections / fused bodies.

Per frame the full Low-Cost-Mocap chain runs live:

    blob detection (visible mode) -> epipolar matching + DLT triangulation
      -> locate_bodies(): the upstream triangle matcher, once per body
         signature -> body origins reprojected into every camera

Overlay per panel:
  * green circle          = ground-truth marker projection (labelled body/marker)
  * red cross             = raw blob detected in this camera
  * body-coloured diamond = fused rigid-body origin (base-pair midpoint),
                            with a heading tick from the base-pair direction

Output: 6-panel mosaic PNG frames + ffmpeg encode (VideoWriter cannot handle
the 3072 px width reliably, so everything goes through PNGs).
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import cv2 as cv

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FRAMES = os.path.join(ROOT, "out", "frames")
CAMS = ["CAM01", "CAM02", "CAM03", "CAM04", "CAM05", "CAM06"]
BODY_COLORS = [(80, 220, 100), (60, 170, 255), (255, 170, 60),
               (200, 120, 255), (255, 90, 90), (120, 230, 230)]

sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap",
                                "low-cost-mocap", "computer_code", "api"))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
os.environ.setdefault("MOCAP_VISIBLE_MODE", "1")
os.environ.setdefault("MOCAP_AUTOSCALE", "0")

from helpers import Cameras, find_point_correspondance_and_object_points  # noqa: E402
from multi_body import load_body_templates, locate_bodies  # noqa: E402


def project(calib_ci, X):
    K, R, t = calib_ci["K"], calib_ci["R"], calib_ci["t"]
    uvw = K @ (R @ np.asarray(X, float).reshape(3, 1) + t)
    uvw = uvw.ravel()
    if abs(uvw[2]) < 1e-9:
        return None
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig-dir", default=os.path.join(ROOT, "out", "rig3"))
    ap.add_argument("--frames-dir", default="{cam}_t650")
    ap.add_argument("--threshold", type=float, default=140.0)
    ap.add_argument("--min-area", type=float, default=3.0)
    ap.add_argument("--max-area", type=float, default=200.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--frames", default="0-599")
    ap.add_argument("--tol", type=float, default=0.025)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "video",
                                                  "detections_6cam_multibody.mp4"))
    ap.add_argument("--outdir", default=None, help="keep the PNG frames")
    args = ap.parse_args()

    f0, f1 = (int(x) for x in args.frames.split("-"))

    rig = json.load(open(os.path.join(args.rig_dir, "rig_blender.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(args.rig_dir, "cameras_calib.json"), encoding="utf-8"))
    calib = []
    for cam, cc in zip(rig["cameras"], cal["cameras"]):
        calib.append({
            "R": np.array(cam["extrinsics_cv"]["R"], float),
            "t": np.array(cam["extrinsics_cv"]["t"], float).reshape(3, 1),
            "K": np.array(cc["K"], float),
        })
    poses = [{"R": c["R"], "t": c["t"]} for c in calib]

    templates = load_body_templates(os.path.join(args.rig_dir, "rig_blender.json"))

    # ground truth 2-D, per body: frame -> cam -> body_id -> [[u,v], ...]
    true = {}
    with open(os.path.join(args.rig_dir, "detections_2d.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            fr = int(r["frame"])
            if f0 <= fr <= f1:
                true.setdefault(fr, {}).setdefault(r["cam_name"], {}) \
                    .setdefault(int(r["body_id"]), []).append(
                        [float(r["u_px"]), float(r["v_px"]),
                         int(r["marker"])])

    Cameras.instance()

    def detect(img_gray):
        _, bw = cv.threshold(img_gray, args.threshold, 255, cv.THRESH_BINARY)
        n, _, stats, cent = cv.connectedComponentsWithStats(bw, 8)
        return [[float(cent[i, 0]), float(cent[i, 1])] for i in range(1, n)
                if args.min_area <= stats[i, cv.CC_STAT_AREA] <= args.max_area]

    pw, ph = int(1280 * args.scale), int(800 * args.scale)
    mosaic_w, mosaic_h = pw * 6, ph
    outdir = args.outdir or os.path.join(ROOT, "out", "video", "multibody_det_png")
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))

    title_h = 36
    for fr in range(f0, f1 + 1):
        full = [cv.imread(os.path.join(FRAMES, args.frames_dir.format(cam=c),
                                       f"f{fr:04d}.png"), cv.IMREAD_GRAYSCALE)
                for c in CAMS]

        bodies = []
        body_reproj = []          # body_reproj[i] = [(u,v) per camera]
        if all(f is not None for f in full):
            image_points = [detect(g) or [[None, None]] for g in full]
            try:
                errs, obj, _ = find_point_correspondance_and_object_points(
                    [list(ip) for ip in image_points], poses,
                    [np.zeros((8, 8, 3), np.uint8) for _ in full])
                pts, evs = [], []
                for X, e in zip(obj, errs):
                    try:
                        if X is None:
                            continue
                        Xa = np.asarray(X, float).reshape(-1)
                        if Xa.size != 3 or not np.all(np.isfinite(Xa)):
                            continue
                        try:
                            ev = float(np.asarray(e).reshape(-1)[0])
                        except Exception:
                            ev = 0.0
                        pts.append(Xa)
                        evs.append(ev)
                    except Exception:
                        continue
                bodies = locate_bodies(np.array(pts) if pts else np.zeros((0, 3)),
                                       np.array(evs) if evs else np.zeros(0),
                                       templates, tol=args.tol)
                for b in bodies:
                    per = []
                    for ci in range(6):
                        uv = project(calib[ci], np.asarray(b["pos"], float))
                        if uv and 0 <= uv[0] < 1280 and 0 <= uv[1] < 800:
                            per.append(uv)
                        else:
                            per.append(None)
                    body_reproj.append(per)
            except Exception as e:
                print(f"  f{fr} triangulation error: {e}", flush=True)

        canvas = np.zeros((mosaic_h + title_h, mosaic_w, 3), np.uint8)
        cv.rectangle(canvas, (0, 0), (mosaic_w, title_h), (40, 48, 60), -1)
        nb = len(bodies)
        cv.putText(canvas,
                   "multi-target mocap  |  frame %04d  t=%5.2fs  bodies:%d  "
                   "ids:%s" % (fr, fr / 30.0, nb,
                               ",".join(str(b["droneIndex"]) for b in bodies)),
                   (12, title_h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 1, cv.LINE_AA)
        lx = 640
        for b in bodies:
            col = BODY_COLORS[b["droneIndex"] % len(BODY_COLORS)]
            cv.putText(canvas, f"T{b['droneIndex']}", (lx, title_h - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv.LINE_AA)
            lx += 46

        for ci, cam in enumerate(CAMS):
            if full[ci] is None:
                continue
            small = cv.resize(full[ci], (pw, ph))
            panel = cv.cvtColor(small, cv.COLOR_GRAY2BGR)
            sx = sy = args.scale

            cv.rectangle(panel, (0, 0), (pw, 20), (70, 90, 120), -1)
            cv.putText(panel, cam, (6, 14), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (240, 240, 240), 1, cv.LINE_AA)
            cv.putText(panel, "raw:%d bodies:%d" % (len(detect(full[ci])), len(bodies)),
                       (pw - 8, 14), cv.FONT_HERSHEY_SIMPLEX, 0.42,
                       (220, 220, 235), 1, cv.LINE_AA)

            # ground truth per body (green, labelled)
            for bid, marks in true.get(fr, {}).get(cam, {}).items():
                col = BODY_COLORS[bid % len(BODY_COLORS)]
                for u, v, mi in marks:
                    p = (int(u * sx), int(v * sy))
                    cv.circle(panel, p, 5, (60, 200, 60), 1, cv.LINE_AA)
                    cv.putText(panel, f"{bid}.{mi}", (p[0] + 6, p[1] - 4),
                               cv.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv.LINE_AA)

            # raw detections (red)
            for u, v in detect(full[ci]):
                p = (int(u * sx), int(v * sy))
                cv.drawMarker(panel, p, (60, 60, 255), cv.MARKER_CROSS, 9, 1)

            # fused body origins (diamond + heading tick), one per body
            for b, per in zip(bodies, body_reproj):
                uv = per[ci]
                if uv is None:
                    continue
                col = BODY_COLORS[b["droneIndex"] % len(BODY_COLORS)]
                p = (int(uv[0] * sx), int(uv[1] * sy))
                cv.drawMarker(panel, p, col, cv.MARKER_DIAMOND, 11, 2)
                cv.putText(panel, f"T{b['droneIndex']}", (p[0] + 7, p[1] + 4),
                           cv.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv.LINE_AA)
                a = float(b["heading"])
                cv.line(panel, p,
                        (int(p[0] + 14 * np.cos(a) * sx),
                         int(p[1] - 14 * np.sin(a) * sy)),
                        col, 1, cv.LINE_AA)

            canvas[title_h:title_h + ph, ci * pw:(ci + 1) * pw] = panel

        cv.imwrite(os.path.join(outdir, f"f{fr:04d}.png"), canvas)
        if fr % 100 == 0:
            print(f"  frame {fr} bodies={nb}", flush=True)

    out = args.out
    subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-start_number", "0",
                    "-i", os.path.join(outdir, "f%04d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                   check=True, capture_output=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
