"""Frame-level stats for the multi-body matcher over the whole clip."""
import collections
import itertools
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap",
                                "low-cost-mocap", "computer_code", "api"))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
sys.path.insert(0, ROOT)
os.environ.setdefault("MOCAP_VISIBLE_MODE", "1")

from helpers import find_point_correspondance_and_object_points  # noqa: E402
from multi_body import load_body_templates, _score_triangle  # noqa: E402
from tools.eval_multi_body import load_calib, load_detections  # noqa: E402

CAMS = ["CAM01", "CAM02", "CAM03", "CAM04", "CAM05", "CAM06"]
rig_dir = os.path.join(ROOT, "out", "rig3")
rig, poses = load_calib(rig_dir)
templates = load_body_templates(os.path.join(rig_dir, "rig_blender.json"))
det = load_detections(os.path.join(rig_dir, "detections_2d.csv"))
dummy = np.zeros((8, 8, 3), np.uint8)

hist_pts = collections.Counter()
hist_bad = collections.Counter()
hist_match = collections.Counter()
best_res = []
err_lt5 = []

for fr in range(0, 600, 5):
    by_cam = det.get(fr, {})
    image_points = [by_cam.get(c) or [[None, None]] for c in CAMS]
    errs, obj, _ = find_point_correspondance_and_object_points(
        [list(ip) for ip in image_points], poses, [dummy] * 6)
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
    pts = np.array(pts) if pts else np.zeros((0, 3))
    evs = np.array(evs) if evs else np.zeros(0)
    hist_pts[len(pts)] += 1
    hist_bad[int((evs > 5).sum())] += 1
    err_lt5.append(int((evs < 5).sum()))

    if len(pts) >= 3:
        dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        good = evs < 5
        pts_g, evs_g = pts[good], evs[good]
        dist_g = np.linalg.norm(pts_g[:, None, :] - pts_g[None, :, :], axis=2)
        m = 0
        for t in templates:
            found = False
            for idx in itertools.combinations(range(len(pts_g)), 3):
                er, fit, _, _ = _score_triangle(t, t["locals"], pts_g, idx)
                if er < 0.06:
                    found = True
                    break
            m += int(found)
        hist_match[m] += 1

print("points/frame :", dict(sorted(hist_pts.items())))
print("points err>5 :", dict(sorted(hist_bad.items())))
print("bodies matched (edge<60mm, err-gated):", dict(sorted(hist_match.items())))
