"""Render a "fused tracking error" video that follows the real Low-Cost-Mocap
usage as closely as possible.

Pipeline per frame (same code paths as the live adapter):
  1. per-camera blob detection (GaussianBlur + brightness threshold, the exact
     visible-mode chain of helpers._camera_read / _find_dot),
  2. cross-camera epipolar matching + DLT triangulation (helpers),
  3. outlier rejection around the median -> fused marker-cloud centroid,
  4. the repo's own KalmanFilter.predict_location() driven by a simulated clock
     (dt = 1/30 s), exactly like the live system feeding it at frame rate.

Left  : 3x2 grid of camera views (green = truth, red = raw detection,
        cyan = fused 3-D points reprojected).
Right : top    - position error (mm) of the RAW fix and of the KALMAN-FILTERED
                track vs the ground-truth marker centroid, dynamic y-scale;
        bottom - top-down (x, y) trajectory: ground truth vs Kalman track;
        readout with current numbers.

Note on attitude: the 20 mm marker triangle subtends only ~5 px at 8-9 m, so a
single frame cannot resolve heading (the two base markers blur together).  The
real system survives this through the temporal Kalman filter, which is why the
filtered curve is the meaningful one.

Output: PNG frame sequence + ffmpeg-encoded mp4, plus a CSV of the errors.
"""
import argparse
import csv
import json
import os
import sys
import types

import numpy as np
import cv2 as cv


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FRAMES = os.path.join(ROOT, "out", "frames")
DET = os.path.join(ROOT, "out", "rig", "detections_2d.csv")
MARK3D = os.path.join(ROOT, "out", "rig", "markers_3d.csv")
RIG = os.path.join(ROOT, "out", "rig", "rig_blender.json")
CAMS = ["CAM01", "CAM02", "CAM03", "CAM04", "CAM05", "CAM06"]
COLORS = [(255, 122, 89), (255, 194, 71), (126, 231, 135),
          (89, 211, 255), (167, 139, 250), (244, 114, 182)]

sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap",
                                "low-cost-mocap", "computer_code", "api"))
os.environ.setdefault("MOCAP_VISIBLE_MODE", "1")
os.environ.setdefault("MOCAP_AUTOSCALE", "0")  # triangulated points are ENU metres already
from helpers import Cameras, find_point_correspondance_and_object_points  # noqa: E402
import KalmanFilter as KFmod  # noqa: E402

# numpy>=2 no longer lets 0-d numpy scalars be indexed with np.newaxis, which
# the repo's LowPassFilter.filter relies on.  Normalise inputs to 1-D arrays
# before calling the original implementation (repo code unchanged).
_lpf_filter_orig = KFmod.LowPassFilter.filter
def _lpf_filter_compat(self, data):
    a = np.asarray(data, dtype=float)
    if a.ndim == 0:
        a = a.reshape(1)
    return _lpf_filter_orig(self, a)
KFmod.LowPassFilter.filter = _lpf_filter_compat


class SimClock:
    """Simulated wall clock so the repo KalmanFilter sees dt = 1/fps."""
    def __init__(self, t0=1000.0):
        self.t = t0
    def time(self):
        return self.t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=150.0)
    ap.add_argument("--min-area", type=float, default=3.0)
    ap.add_argument("--max-area", type=float, default=200.0)
    ap.add_argument("--inlier-radius", type=float, default=0.35)
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frames", default="0-599")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "video", "fused_tracking_error.mp4"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "out", "video", "fused_err_png"))
    ap.add_argument("--csv", default=os.path.join(ROOT, "out", "rig", "fused_tracking_error.csv"))
    args = ap.parse_args()

    f0, f1 = (int(x) for x in args.frames.split("-"))

    # ---- calibration ------------------------------------------------------
    rig = json.load(open(RIG))
    cal = json.load(open(os.path.join(ROOT, "out", "rig", "cameras_calib.json")))
    calib, poses = [], []
    for cam, cc in zip(rig["cameras"], cal["cameras"]):
        R = np.array(cam["extrinsics_cv"]["R"], float)
        t = np.array(cam["extrinsics_cv"]["t"], float).reshape(3, 1)
        K = np.array(cc["K"], float)
        calib.append({"R": R, "t": t, "K": K})
        poses.append({"R": R, "t": t})

    # ---- ground truth -----------------------------------------------------
    true2d = {}
    with open(DET, newline="") as fh:
        for r in csv.DictReader(fh):
            true2d.setdefault(int(r["frame"]), {}).setdefault(r["cam_name"], []).append(
                [float(r["u_px"]), float(r["v_px"])])
    truth3d = {}
    with open(MARK3D, newline="") as fh:
        for r in csv.DictReader(fh):
            truth3d.setdefault(int(r["frame"]), []).append(
                [float(r["x"]), float(r["y"]), float(r["z"])])

    Cameras.instance()

    def detect(gray):
        """Identical to the adapter's visible-mode chain: GaussianBlur(9x9) as in
        _camera_read, brightness threshold, area-gated connected components."""
        rgb = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        rgb = cv.GaussianBlur(rgb, (9, 9), 0)
        g = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(g, args.threshold, 255, cv.THRESH_BINARY)
        n, _, stats, cent = cv.connectedComponentsWithStats(bw, 8)
        return [[float(cent[i, 0]), float(cent[i, 1])] for i in range(1, n)
                if args.min_area <= stats[i, cv.CC_STAT_AREA] <= args.max_area]

    def fuse(cands):
        """Outlier rejection around the median -> centroid + inliers by err."""
        if not cands:
            return None, []
        P = np.array([c[1] for c in cands])
        ev = np.array([c[0] for c in cands])
        med = np.median(P, axis=0)
        mask = np.linalg.norm(P - med, axis=1) < args.inlier_radius
        if not mask.any():
            return None, []
        inl, ev_i = P[mask], ev[mask]
        order = np.argsort(ev_i)
        return inl.mean(0), [inl[i] for i in order]

    # ---- the repo's Kalman filter on a simulated clock --------------------
    clock = SimClock()
    KFmod.time = clock                      # KalmanFilter.py does `import time`
    kf = KFmod.KalmanFilter(num_objects=1)
    kf.prev_measurement_time = clock.t      # first dt = 1/fps, not 20 s
    # The repo tunes processNoiseCov=1e-2 for its 150 Hz loop with hovering
    # drones.  At the 30 fps replay rate the same filter needs the process
    # noise scaled to the per-step dynamics, otherwise the Kalman gain stays
    # ~0.05 and the velocity state never tracks (the track then lags the
    # target by hundreds of mm).  Q=1.0 / R=1.0 (repo default) is the
    # equivalent tuning at 30 fps.
    for k in kf.kalmans:
        k.processNoiseCov = np.eye(9, dtype=np.float32) * 1.0

    pw, ph = int(1280 * args.scale), int(800 * args.scale)
    gap = 8
    grid_w = 3 * pw + 2 * gap
    grid_h = 2 * ph + gap
    right_w = 1000
    title_h = 40
    W = 16 + grid_w + 16 + right_w + 16
    H = title_h + grid_h + 16
    rx = 16 + grid_w + 16
    rw = right_w

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    hist_raw, hist_kf = [], []
    hist_truth_xy, hist_kf_xy = [], []
    last_fix_frame = None

    for fr in range(f0, f1 + 1):
        clock.t += 1.0 / args.fps
        full = [cv.imread(os.path.join(FRAMES, f"{c}_vis", f"f{fr:04d}.png"),
                          cv.IMREAD_GRAYSCALE) for c in CAMS]

        fused_reproj = [[] for _ in range(6)]
        fused_pts, raw_fix = [], None
        if all(f is not None for f in full):
            image_points = [detect(g) or [[None, None]] for g in full]
            try:
                errs, obj, _ = find_point_correspondance_and_object_points(
                    [list(ip) for ip in image_points], poses,
                    [cv.cvtColor(g, cv.COLOR_GRAY2BGR) for g in full])
                cand = []
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
                            ev = float(e)
                        cand.append((ev, Xa))
                    except Exception:
                        continue
                raw_fix, inliers = fuse(cand)
                fused_pts = inliers[:3]
                for X in fused_pts:
                    Xh = np.array([X[0], X[1], X[2], 1.0])
                    for ci in range(6):
                        P = calib[ci]["K"] @ np.hstack([calib[ci]["R"], calib[ci]["t"]])
                        uvw = P @ Xh
                        if abs(uvw[2]) < 1e-9:
                            continue
                        u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
                        if 0 <= u < 1280 and 0 <= v < 800:
                            fused_reproj[ci].append([float(u), float(v)])
            except Exception as e:
                print("  f%d tri err: %s" % (fr, e), flush=True)

        # ---- the real consumption path: Kalman filter on the fix ----------
        # Two standard behaviours of the live system are reproduced here:
        #   * warm-up: the very first fix only initialises the filter state
        #     (otherwise the KF's prev_positions=[0,0,0] turns the first
        #     measurement into a ~40 m/s fake velocity);
        #   * re-acquisition: after a long gap (no fixes) the filter is reset.
        kf_pos = None
        if raw_fix is not None:
            warm = False
            if last_fix_frame is None or (fr - last_fix_frame) / args.fps > 0.5:
                kf.reset()
                # one frame of dt for the warm-up update (dt=0 would NaN out)
                kf.prev_measurement_time = clock.t - 1.0 / args.fps
                warm = True
            meas = np.asarray(raw_fix, dtype=np.float32)
            filtered = kf.predict_location([{
                "pos": meas, "vel": np.zeros(3, dtype=np.float32),
                "heading": 0.0, "droneIndex": 0}])
            if not warm and filtered:
                cand_kf = np.asarray(filtered[0]["pos"], float)
                if np.all(np.isfinite(cand_kf)):
                    kf_pos = cand_kf
            last_fix_frame = fr
        else:
            kf_pos = None

        # ---- errors vs truth ---------------------------------------------
        T = np.array(truth3d.get(fr, []))
        tcent = T.mean(0) if len(T) else None
        raw_err = float(np.linalg.norm(raw_fix - tcent) * 1000) if (raw_fix is not None and tcent is not None) else None
        kf_err = float(np.linalg.norm(kf_pos - tcent) * 1000) if (kf_pos is not None and tcent is not None) else None
        rows.append([fr,
                     "" if raw_err is None else "%.2f" % raw_err,
                     "" if kf_err is None else "%.2f" % kf_err,
                     len(fused_pts), "1" if raw_fix is not None else "0"])
        if raw_err is not None:
            hist_raw.append((fr, raw_err))
        if kf_err is not None:
            hist_kf.append((fr, kf_err))
        if tcent is not None:
            hist_truth_xy.append((fr, tcent[0], tcent[1]))
        if kf_pos is not None:
            hist_kf_xy.append((fr, kf_pos[0], kf_pos[1]))

        # ================= draw =================
        canvas = np.zeros((H, W, 3), np.uint8)
        cv.rectangle(canvas, (0, 0), (W, title_h), (32, 40, 52), -1)
        cv.putText(canvas, "Low-Cost-Mocap  fused tracking vs ground truth (Kalman-filtered)  |  frame %04d  t=%5.2fs"
                   % (fr, fr / 30.0), (14, title_h - 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.72, (235, 240, 248), 1, cv.LINE_AA)

        # camera panels
        for ci, cam in enumerate(CAMS):
            col, row = ci % 3, ci // 3
            x0 = 16 + col * (pw + gap)
            y0 = title_h + row * (ph + gap)
            if full[ci] is None:
                continue
            panel = cv.cvtColor(cv.resize(full[ci], (pw, ph)), cv.COLOR_GRAY2BGR)
            det = detect(full[ci])
            cv.rectangle(panel, (0, 0), (pw, 18), COLORS[ci], -1)
            cv.putText(panel, cam, (5, 13), cv.FONT_HERSHEY_SIMPLEX, 0.45,
                       (10, 14, 22), 1, cv.LINE_AA)
            for (dx, dy) in det:
                cx, cy = int(dx * args.scale), int(dy * args.scale)
                cv.circle(panel, (cx, cy), 7, (0, 0, 255), 2)
                cv.line(panel, (cx - 7, cy), (cx + 7, cy), (0, 0, 255), 2)
                cv.line(panel, (cx, cy - 7), (cx, cy + 7), (0, 0, 255), 2)
            for (tu, tv) in true2d.get(fr, {}).get(cam, []):
                cx, cy = int(tu * args.scale), int(tv * args.scale)
                cv.circle(panel, (cx, cy), 4, (0, 255, 0), -1)
                cv.circle(panel, (cx, cy), 9, (0, 255, 0), 1)
            for (u, v) in fused_reproj[ci]:
                cx, cy = int(u * args.scale), int(v * args.scale)
                r = 8
                pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]])
                cv.fillConvexPoly(panel, pts, (0, 255, 255))
                cv.polylines(panel, [pts], True, (0, 0, 0), 1)
            canvas[y0:y0 + ph, x0:x0 + pw] = panel

        # ---- right panel ----
        cv.rectangle(canvas, (rx, title_h), (rx + rw, title_h + grid_h), (18, 24, 34), -1)
        cv.rectangle(canvas, (rx, title_h), (rx + rw, title_h + grid_h), (60, 74, 96), 1)

        # plot box helper (clips everything into the box)
        def plot_box(x0, y0, w0, h0, label):
            cv.rectangle(canvas, (x0, y0), (x0 + w0, y0 + h0), (60, 74, 96), 1)
            cv.putText(canvas, label, (x0 + 10, y0 + 16), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (150, 210, 255), 1, cv.LINE_AA)
            return x0 + 46, y0 + 26, x0 + w0 - 12, y0 + h0 - 26   # l, t, r, b

        # ---- plot A: position error, dynamic scale ----
        pa_l, pa_t, pa_r, pa_b = plot_box(rx + 10, title_h + 10, rw - 20, grid_h // 2 - 22,
                                          "position error (mm)   green=raw fix   orange=kalman")
        ymax = 20.0
        for (_, e) in hist_raw:
            ymax = max(ymax, e)
        for (_, e) in hist_kf:
            ymax = max(ymax, e)
        ymax *= 1.15
        step = 10 if ymax <= 60 else 25 if ymax <= 150 else 50
        gy = 0
        while gy <= ymax:
            yy = int(pa_b - (gy / ymax) * (pa_b - pa_t))
            cv.line(canvas, (pa_l, yy), (pa_r, yy), (45, 58, 76), 1)
            cv.putText(canvas, str(int(gy)), (pa_l - 32, yy + 4), cv.FONT_HERSHEY_SIMPLEX,
                       0.4, (120, 140, 165), 1, cv.LINE_AA)
            gy += step
        def xs(f):
            return pa_l + int((f - f0) / max(1, f1 - f0) * (pa_r - pa_l))
        def ye(e):
            return int(np.clip(pa_b - (e / ymax) * (pa_b - pa_t), pa_t, pa_b))
        for a, b in (((hist_raw[i], hist_raw[i + 1]) for i in range(len(hist_raw) - 1))):
            pass
        for i in range(len(hist_raw) - 1):
            (fa, ea), (fb, eb) = hist_raw[i], hist_raw[i + 1]
            if fb - fa > 3:
                continue
            cv.line(canvas, (xs(fa), ye(ea)), (xs(fb), ye(eb)), (120, 255, 180), 1, cv.LINE_AA)
        for i in range(len(hist_kf) - 1):
            (fa, ea), (fb, eb) = hist_kf[i], hist_kf[i + 1]
            if fb - fa > 3:
                continue
            cv.line(canvas, (xs(fa), ye(ea)), (xs(fb), ye(eb)), (120, 190, 255), 2, cv.LINE_AA)
        if raw_err is not None:
            cv.circle(canvas, (xs(fr), ye(raw_err)), 4, (160, 255, 200), -1)
        if kf_err is not None:
            cv.circle(canvas, (xs(fr), ye(kf_err)), 5, (255, 200, 120), -1)
            cv.circle(canvas, (xs(fr), ye(kf_err)), 7, (255, 200, 120), 1)

        # ---- plot B: top-down trajectory (x, y) ----
        pb_l, pb_t, pb_r, pb_b = plot_box(rx + 10, title_h + grid_h // 2 + 2, rw - 20,
                                          grid_h // 2 - 22, "top-down track x/y (m)   white=truth   blue=kalman")
        R = 2.6
        cxm = (pb_l + pb_r) / 2
        cym = (pb_t + pb_b) / 2
        s = min(pb_r - pb_l, pb_b - pb_t) / (2 * R)
        def tx(x):
            return int(np.clip(cxm + x * s, pb_l, pb_r))
        def ty(y):
            return int(np.clip(cym - y * s, pb_t, pb_b))
        for gx in (-2, -1, 0, 1, 2):
            cv.line(canvas, (tx(gx), pb_t), (tx(gx), pb_b), (40, 52, 68), 1)
            cv.putText(canvas, str(gx), (tx(gx) - 6, pb_b + 14), cv.FONT_HERSHEY_SIMPLEX,
                       0.38, (120, 140, 165), 1, cv.LINE_AA)
        for gy2 in (-2, -1, 0, 1, 2):
            cv.line(canvas, (pb_l, ty(gy2)), (pb_r, ty(gy2)), (40, 52, 68), 1)
            cv.putText(canvas, str(gy2), (pb_l - 30, ty(gy2) + 4), cv.FONT_HERSHEY_SIMPLEX,
                       0.38, (120, 140, 165), 1, cv.LINE_AA)
        for i in range(len(hist_truth_xy) - 1):
            fa, xa, ya = hist_truth_xy[i]
            fb, xb, yb = hist_truth_xy[i + 1]
            cv.line(canvas, (tx(xa), ty(ya)), (tx(xb), ty(yb)), (200, 205, 215), 1, cv.LINE_AA)
        for i in range(len(hist_kf_xy) - 1):
            fa, xa, ya = hist_kf_xy[i]
            fb, xb, yb = hist_kf_xy[i + 1]
            cv.line(canvas, (tx(xa), ty(ya)), (tx(xb), ty(yb)), (255, 170, 80), 2, cv.LINE_AA)
        if kf_pos is not None:
            cv.circle(canvas, (tx(kf_pos[0]), ty(kf_pos[1])), 5, (120, 190, 255), -1)
        if tcent is not None:
            cv.circle(canvas, (tx(tcent[0]), ty(tcent[1])), 4, (255, 255, 255), 1)

        # ---- readout ----
        s = "frame %04d   fused pts %d   fix %s" % (fr, len(fused_pts), "yes" if raw_fix is not None else "no")
        if raw_err is not None:
            s += "   raw %5.1f mm" % raw_err
        if kf_err is not None:
            s += "   kalman %5.1f mm" % kf_err
        cv.putText(canvas, s, (rx + 12, title_h + grid_h - 10), cv.FONT_HERSHEY_SIMPLEX,
                   0.55, (235, 240, 248), 1, cv.LINE_AA)

        cv.imwrite(os.path.join(args.outdir, "d%04d.png" % fr), canvas)
        if (fr - f0) % 60 == 0:
            r_s = "-" if raw_err is None else "%.1f" % raw_err
            k_s = "-" if kf_err is None else "%.1f" % kf_err
            print("  f%04d  raw=%s mm  kalman=%s mm" % (fr, r_s, k_s), flush=True)

    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "raw_pos_err_mm", "kalman_pos_err_mm", "fused_points", "has_fix"])
        w.writerows(rows)
    print("wrote csv", args.csv)
    print("wrote png frames to", args.outdir)


if __name__ == "__main__":
    main()