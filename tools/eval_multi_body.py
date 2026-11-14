"""Ideal-geometry diagnostic, NOT an actual-image/video accuracy measurement.

Chain under test is the real one:

    per-camera 2D detections
      -> cross-camera epipolar matching + DLT triangulation   (helpers.py)
      -> repository apex/base search, per body signature       (multi_body.py)
      -> repo KalmanFilter per body                           (KalmanFilter.py)

Detections come from ``out/rig3/detections_2d.csv`` (projected synthetic
observations), WITHOUT decoding/detecting the delivered videos.  For real
pixel detection and scoring use tools/evaluate_multi_video.py instead.
Legacy gate arguments here are unused; no custom temporal gate is applied.

Reports, per body: coverage, position error, and -- the number that matters
for multi-drone flight -- identity-keeping rate, i.e. how often the reported
body id is the one that is actually there.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ADAPTER = os.path.join(ROOT, "adapters", "low_cost_mocap")
API = os.path.join(ADAPTER, "low-cost-mocap", "computer_code", "api")
sys.path.insert(0, API)
sys.path.insert(0, ADAPTER)
os.environ.setdefault("MOCAP_VISIBLE_MODE", "1")
os.environ.setdefault("MOCAP_AUTOSCALE", "0")

from helpers import find_point_correspondance_and_object_points  # noqa: E402
from multi_body import BodyTracker, load_body_templates, locate_bodies  # noqa: E402
import KalmanFilter as KFmod  # noqa: E402

CAMS = ["CAM01", "CAM02", "CAM03", "CAM04", "CAM05", "CAM06"]


def _lpf_compat(self, data):
    a = np.asarray(data, dtype=float)
    if a.ndim == 0:
        a = a.reshape(1)
    return _lpf_orig(self, a)


_lpf_orig = KFmod.LowPassFilter.filter
KFmod.LowPassFilter.filter = _lpf_compat


class SimClock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t


def load_calib(rig_dir):
    rig = json.load(open(os.path.join(rig_dir, "rig_blender.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(rig_dir, "cameras_calib.json"), encoding="utf-8"))
    poses = []
    for cam, cc in zip(rig["cameras"], cal["cameras"]):
        poses.append({
            "R": np.array(cam["extrinsics_cv"]["R"], float),
            "t": np.array(cam["extrinsics_cv"]["t"], float).reshape(3, 1),
            "K": np.array(cc["K"], float),
        })
    return rig, poses


def load_truth(path):
    """body_id -> {frame: np.array([x,y,z])}"""
    truth = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            truth.setdefault(int(r["body_id"]), {})[int(r["frame"])] = np.array(
                [float(r["x"]), float(r["y"]), float(r["z"])])
    return truth


def load_detections(path):
    """frame -> cam_name -> [(u, v)]"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(int(r["frame"]), {}).setdefault(r["cam_name"], []).append(
                [float(r["u_px"]), float(r["v_px"])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig-dir", default=os.path.join(ROOT, "out", "rig3"))
    ap.add_argument("--trajectory", default=os.path.join(ROOT, "out", "rig", "trajectory3.csv"))
    ap.add_argument("--frames", default="0-599")
    ap.add_argument("--tol", type=float, default=0.030)
    ap.add_argument("--gate", type=float, default=1.20)
    ap.add_argument("--max-err", type=float, default=5.0)
    ap.add_argument("--reject", type=float, default=0.35)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-kalman", action="store_true")
    ap.add_argument("--csv-out", default=os.path.join(ROOT, "out", "rig3", "multi_body_error.csv"))
    args = ap.parse_args()

    f0, f1 = (int(x) for x in args.frames.split("-"))
    rig, poses = load_calib(args.rig_dir)
    templates = load_body_templates(os.path.join(args.rig_dir, "rig_blender.json"))
    truth = load_truth(args.trajectory)
    det = load_detections(os.path.join(args.rig_dir, "detections_2d.csv"))
    body_ids = [t["body_id"] for t in templates]
    print("bodies:", ", ".join(f"{t['body_id']}:{t['name']}" for t in templates))

    tracker = BodyTracker(templates, gate=args.gate)
    dummy = np.zeros((8, 8, 3), np.uint8)

    # Upstream locate_objects reports the *midpoint of the base pair*, not the
    # body origin, so the truth reference must be the same point: transform
    # the base-pair midpoint through the ground-truth pose.
    from scipy.spatial.transform import Rotation
    base_mid = {t["body_id"]: t["base_mid_local"] for t in templates}

    def truth_ref(b, fr):
        r = truth[b][fr]
        # trajectory3.csv stores qw,qx,qy,qz in the mesh/ENU body frame
        q = quat_by_body.get((b, fr))
        if q is None:
            return r
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return r + R @ base_mid[b]

    quat_by_body = {}
    with open(args.trajectory, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            quat_by_body[(int(r["body_id"]), int(r["frame"]))] = np.array(
                [float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])])

    kf = None
    clock = SimClock()
    if not args.no_kalman:
        KFmod.time = clock
        kf = KFmod.KalmanFilter(num_objects=len(templates))
        kf.prev_measurement_time = clock.t
        for k in kf.kalmans:
            k.processNoiseCov = np.eye(9, dtype=np.float32) * 1.0

    stats = {b: {"n": 0, "raw": [], "kf": [], "id_ok": 0, "id_n": 0,
                 "kf_id_ok": 0, "kf_id_n": 0} for b in body_ids}
    rows = []
    warm = 0

    for fr in range(f0, f1 + 1):
        clock.t += 1.0 / args.fps
        by_cam = det.get(fr, {})
        image_points = [by_cam.get(c) or [[None, None]] for c in CAMS]

        errs, obj, _ = find_point_correspondance_and_object_points(
            [list(ip) for ip in image_points], poses, [dummy] * len(CAMS))

        pts, evs = [], []
        for X, e in zip(obj, errs):
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

        objects = locate_bodies(np.array(pts) if pts else np.zeros((0, 3)),
                                np.array(evs) if evs else np.zeros(0),
                                templates, tracker, tol=args.tol,
                                fps=args.fps, frame=fr,
                                max_err=args.max_err, reject=args.reject)

        # ---- raw association vs truth ------------------------------------
        truth_fr = {b: truth_ref(b, fr) for b in body_ids if fr in truth[b]}
        raw_pos = {}
        for o in objects:
            b = o["droneIndex"]
            raw_pos[b] = np.asarray(o["pos"], float)
            if b in truth_fr:
                e = float(np.linalg.norm(raw_pos[b] - truth_fr[b]))
                stats[b]["n"] += 1
                stats[b]["raw"].append(e)
            # identity: is the reported id the closest body?
            if len(truth_fr) >= 2:
                nearest = min(truth_fr, key=lambda k: np.linalg.norm(raw_pos[b] - truth_fr[k]))
                stats[b]["id_n"] += 1
                stats[b]["id_ok"] += int(nearest == b)

        # ---- repo Kalman filter -------------------------------------------
        kf_pos = {}
        if kf is not None:
            warm += 1
            filt = kf.predict_location(objects)
            if warm > 5:
                for fo in filt:
                    b = fo["droneIndex"]
                    kf_pos[b] = np.asarray(fo["pos"], float)
                    if b in truth_fr:
                        e = float(np.linalg.norm(kf_pos[b] - truth_fr[b]))
                        stats[b]["kf"].append(e)
                    if len(truth_fr) >= 2:
                        nearest = min(truth_fr,
                                      key=lambda k: np.linalg.norm(kf_pos[b] - truth_fr[k]))
                        stats[b]["kf_id_n"] += 1
                        stats[b]["kf_id_ok"] += int(nearest == b)

        rows.append([fr] + [
            v for b in body_ids
            for v in (
                f"{raw_pos[b][0]:.5f},{raw_pos[b][1]:.5f},{raw_pos[b][2]:.5f}"
                if b in raw_pos else ",,"
            ).split(",")
        ])

    # ---- report ------------------------------------------------------------
    def summ(a):
        a = np.asarray(a, float)
        if a.size == 0:
            return "   n/a   "
        return (f"mean {1000*a.mean():7.2f}  p90 {1000*np.percentile(a,90):7.2f}  "
                f"max {1000*a.max():8.2f}  (mm, n={a.size})")

    print(f"\nframes {f0}-{f1} @ {args.fps:g} fps, tol {args.tol*1000:.0f} mm, gate {args.gate:g} m")
    for b in body_ids:
        s = stats[b]
        name = next(t["name"] for t in templates if t["body_id"] == b)
        cov = 100.0 * s["n"] / max(1, (f1 - f0 + 1))
        print(f"\nbody {b} ({name})  coverage {cov:5.1f}%")
        print(f"  raw  {summ(s['raw'])}")
        print(f"  kf   {summ(s['kf'])}")
        if s["id_n"]:
            print(f"  identity  raw {100.0*s['id_ok']/s['id_n']:6.2f}%   "
                  f"kf {100.0*s['kf_id_ok']/max(1,s['kf_id_n']):6.2f}%  "
                  f"(n={s['id_n']})")

    all_raw = np.array([e for b in body_ids for e in stats[b]["raw"]])
    all_kf = np.array([e for b in body_ids for e in stats[b]["kf"]])
    idn = sum(stats[b]["id_n"] for b in body_ids)
    idok = sum(stats[b]["id_ok"] for b in body_ids)
    print("\n--- overall ---")
    if all_raw.size:
        print(f"raw position error : mean {1000*all_raw.mean():.2f} mm, "
              f"p90 {1000*np.percentile(all_raw, 90):.2f} mm, "
              f"max {1000*all_raw.max():.2f} mm")
    if all_kf.size:
        print(f"kf  position error : mean {1000*all_kf.mean():.2f} mm, "
              f"p90 {1000*np.percentile(all_kf, 90):.2f} mm, "
              f"max {1000*all_kf.max():.2f} mm")
    if idn:
        print(f"identity keeping   : raw {100.0*idok/idn:.2f}%  (n={idn})")

    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + [f"body{b}_{ax}" for b in body_ids for ax in "xyz"])
        w.writerows(rows)
    print("csv:", args.csv_out)


if __name__ == "__main__":
    main()
