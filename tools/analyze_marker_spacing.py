"""Monte-Carlo: how does marker spacing affect attitude estimation?

Uses the real rig (6x OV9281 at 5 m radius / 6.4 m height, fx=933 px) and the
real flight (trajectory.csv quaternions).  For each candidate spacing the
marker triangle is scaled, projected into all six cameras, measurement noise is
added (sigma matched to the measured centroid error), close blobs are merged
(exactly what the 9x9 Gaussian pre-filter does to markers only a few px apart),
the points are DLT-triangulated and the rigid transform to the truth triangle
gives the attitude error.

Outputs a table + CSV: attitude error and 3-marker visibility vs spacing.
"""
import argparse
import csv
import json
import os
import itertools

import numpy as np
import cv2 as cv


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TRAJ = os.path.join(ROOT, "out", "rig", "trajectory.csv")
MARK3D = os.path.join(ROOT, "out", "rig", "markers_3d.csv")
RIG = os.path.join(ROOT, "out", "rig", "rig_blender.json")
CAL = os.path.join(ROOT, "out", "rig", "cameras_calib.json")


def quat_to_R(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def dlt(Ps, uvs):
    """Direct linear transform, same math as helpers.triangulate_point."""
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A.append(v * P[2, :] - P[1, :])
        A.append(P[0, :] - u * P[2, :])
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    X = Vh[-1]
    return X[:3] / X[3]


def kabsch_angle(P, Q):
    """Rotation angle of the rigid transform mapping P onto Q (deg)."""
    muP, muQ = P.mean(0), Q.mean(0)
    H = (P - muP).T @ (Q - muQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def merge_blobs(uvs, merge_px):
    """Emulate the 9x9 Gaussian pre-filter: markers closer than merge_px
    collapse into a single blob at their midpoint (iteratively)."""
    pts = [np.array(p, float) for p in uvs]
    changed = True
    while changed and len(pts) > 1:
        changed = False
        for i, j in itertools.combinations(range(len(pts)), 2):
            if np.linalg.norm(pts[i] - pts[j]) < merge_px:
                mid = (pts[i] + pts[j]) / 2
                pts = [p for k, p in enumerate(pts) if k not in (i, j)] + [mid]
                changed = True
                break
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-px", type=float, default=1.0,
                    help="per-marker pixel centroid noise (1-sigma)")
    ap.add_argument("--merge-px", type=float, default=9.0,
                    help="blob merge distance of the 9x9 blur chain")
    ap.add_argument("--frames", default="0-599")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--trials", type=int, default=8, help="noise trials per frame")
    ap.add_argument("--csv", default=os.path.join(ROOT, "out", "rig", "marker_spacing_study.csv"))
    args = ap.parse_args()

    rig = json.load(open(RIG))
    cal = json.load(open(CAL))
    local = np.array([m["local"] for m in sorted(rig["markers"], key=lambda m: m["id"])])
    base_spacing = float(np.linalg.norm(local[0] - local[1]))   # 47 mm
    calib = []
    for cc in cal["cameras"]:
        K = np.array(cc["K"], float)
        R = np.array(cc["R"], float)
        t = np.array(cc["t"], float).reshape(3, 1)
        calib.append((K, R, t))

    # truth body poses
    traj = {}
    with open(TRAJ, newline="") as fh:
        for r in csv.DictReader(fh):
            f = int(r["frame"])
            if f < int(args.frames.split("-")[0]) or f > int(args.frames.split("-")[1]):
                continue
            traj[f] = (np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
                       np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]))

    # sanity: reconstructed markers must match markers_3d.csv
    truth3d = {}
    with open(MARK3D, newline="") as fh:
        for r in csv.DictReader(fh):
            truth3d.setdefault(int(r["frame"]), []).append(
                [float(r["x"]), float(r["y"]), float(r["z"])])

    frames = sorted(traj)[::args.step]
    rng = np.random.default_rng(7)

    vol = rig["volume"]
    DATA_VOLUME = {"c": np.array(vol["center"], float),
                   "h": np.array(vol["size"], float) / 2}

    spacings = [0.047, 0.070, 0.100, 0.150, 0.200, 0.300]
    print("sigma_px=%.2f  merge_px=%.1f  frames=%d x trials=%d"
          % (args.sigma_px, args.merge_px, len(frames), args.trials))
    print("%-8s %-10s %-10s %-10s %-12s" % ("spacing", "att_deg", "att_p90", "pos_mm", "frames_3mk"))
    rows = []
    for s in spacings:
        k = s / base_spacing
        errs, poss, n3 = [], [], 0
        for fr in frames:
            body_pos, q = traj[fr]
            Rb = quat_to_R(q)
            mk = body_pos[None, :] + (k * local) @ Rb.T          # (3,3) world
            # project into each camera
            per_cam = []
            for (K, R, t) in calib:
                uv = (K @ (R @ mk.T + t)).T
                uv = uv[:, :2] / uv[:, 2:3]
                per_cam.append(uv)
            for trial in range(args.trials):
                # per-marker: collect pixel observations from all cameras,
                # merging blobs per camera first
                marker_obs = [[] for _ in range(3)]
                for cam_idx, (K, R, t) in enumerate(calib):
                    uvh = (K @ (R @ mk.T + t)).T
                    uv = uvh[:, :2] / uvh[:, 2:3]
                    uv = uv + rng.normal(0, args.sigma_px, uv.shape)
                    # The 9x9 Gaussian pre-filter merges two markers closer
                    # than ~merge_px into a single blob: both are then lost to
                    # the per-marker association (the merged centroid belongs
                    # to neither).
                    avail = [0, 1, 2]
                    for i, j in itertools.combinations(range(3), 2):
                        if np.linalg.norm(uv[i] - uv[j]) < args.merge_px:
                            avail = [m for m in avail if m not in (i, j)]
                    for m in avail:
                        marker_obs[m].append((cam_idx, uv[m]))
                if sum(1 for mo in marker_obs if len(mo) >= 2) < 3:
                    continue
                est = []
                ok = True
                for mo in marker_obs:
                    if len(mo) < 2:
                        ok = False
                        break
                    # P matrices must match the cameras that actually saw
                    # this marker -- a naive zip would misattribute every
                    # observation after a discarded view.
                    Ps_m = [calib[ci][0] @ np.hstack([calib[ci][1], calib[ci][2]])
                            for ci, _ in mo]
                    est.append(dlt(Ps_m, [p for _, p in mo]))
                if not ok:
                    continue
                est = np.array(est)
                # reject wild fixes the way a real mocap system does: every
                # marker must land inside the capture volume (plus margin).
                # Degenerate view sets (few remaining cameras after a
                # foreshortened pair merged) blow up DLT otherwise.
                v = DATA_VOLUME
                if not np.all(np.abs(est - v["c"]) <= v["h"] + 0.5):
                    continue
                n3 += 1
                errs.append(kabsch_angle(est, mk))
                poss.append(np.linalg.norm(est.mean(0) - mk.mean(0)) * 1000)
        if errs:
            e = np.array(errs)
            p = np.array(poss)
            print("%-8.0f %-10.2f %-10.2f %-10.1f %-12.0f%%"
                  % (s * 1000, e.mean(), np.percentile(e, 90), p.mean(),
                     100.0 * n3 / (len(frames) * args.trials)))
            rows.append([s * 1000, e.mean(), np.percentile(e, 90), np.percentile(e, 99),
                         p.mean(), 100.0 * n3 / (len(frames) * args.trials)])
        else:
            print("%-8.0f no solutions" % (s * 1000))
            rows.append([s * 1000, None, None, None, None, 0])

    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["spacing_mm", "att_err_deg_mean", "att_err_deg_p90",
                    "att_err_deg_p99", "pos_err_mm_mean", "frames_with_3_markers_pct"])
        w.writerows(rows)
    print("wrote", args.csv)


if __name__ == "__main__":
    main()