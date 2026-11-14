"""Ideal-geometry coverage check for the motion gate on the NEW rig geometry.
Confirms a 0.5 m gate does not destroy normal coverage on clean projected
points (the old near-equilateral B/C collapsed; the new strongly-isosceles
layout should coast cleanly through the same gate).
"""
import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap",
                                "low-cost-mocap", "computer_code", "api"))
sys.path.insert(0, ROOT)
os.environ.setdefault("MOCAP_VISIBLE_MODE", "1")

from helpers import find_point_correspondance_and_object_points  # noqa: E402
from multi_body import BodyTracker, load_body_templates, locate_bodies  # noqa: E402
from tools.eval_multi_body import load_calib, load_detections  # noqa: E402

CAMS = ["CAM01", "CAM02", "CAM03", "CAM04", "CAM05", "CAM06"]


def main():
    rig, poses = load_calib(os.path.join(ROOT, "out", "rig3"))
    templates = load_body_templates(os.path.join(ROOT, "out", "rig3", "rig_blender.json"))
    det = load_detections(os.path.join(ROOT, "out", "rig3", "detections_2d.csv"))
    truth, quat = {}, {}
    with open(os.path.join(ROOT, "out", "rig", "trajectory3.csv"), newline="") as handle:
        for row in csv.DictReader(handle):
            b, fr = int(row["body_id"]), int(row["frame"])
            truth.setdefault(b, {})[fr] = np.array([float(row[a]) for a in "xyz"])
            quat[(b, fr)] = np.array([float(row["qw"]), float(row["qx"]),
                                      float(row["qy"]), float(row["qz"])])
    base = {t["body_id"]: t["base_mid_local"] for t in templates}

    def reference(b, fr):
        q = quat[(b, fr)]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return truth[b][fr] + R @ base[b]

    dummy = np.zeros((8, 8, 3), np.uint8)
    for gate in (None, 0.5, 0.3):
        tracker = BodyTracker(templates, gate=gate) if gate else None
        stat = {b: {"n": 0, "e": []} for b in (0, 1, 2)}
        for frame in range(600):
            by_cam = det.get(frame, {})
            image_points = [by_cam.get(c) or [[None, None]] for c in CAMS]
            errs, obj, _ = find_point_correspondance_and_object_points(
                [list(p) for p in image_points], poses, [dummy] * 6)
            pts, ev = [], []
            for X, e in zip(obj, errs):
                if X is None:
                    continue
                Xa = np.asarray(X, float).reshape(-1)
                if Xa.size == 3 and np.all(np.isfinite(Xa)):
                    pts.append(Xa)
                    ev.append(float(np.asarray(e).ravel()[0]))
            objects = locate_bodies(np.array(pts) if pts else np.zeros((0, 3)),
                                    np.array(ev) if ev else np.zeros(0),
                                    templates, tracker=tracker, tol=0.025,
                                    frame=frame, dt=1 / 30)
            for o in objects:
                b = o["droneIndex"]
                stat[b]["n"] += 1
                stat[b]["e"].append(float(np.linalg.norm(o["pos"] - reference(b, frame))) * 1000)
        print(f"gate={gate}: coverage / median / p90 / max (mm)")
        for b in (0, 1, 2):
            e = np.asarray(stat[b]["e"])
            print(f"   body{b}: cov={100 * stat[b]['n'] / 600:5.1f}% "
                  f"med={np.median(e):6.1f} p90={np.percentile(e, 90):6.1f} "
                  f"max={e.max():6.1f}")


if __name__ == "__main__":
    main()