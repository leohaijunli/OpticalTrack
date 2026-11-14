"""Debug the multi-body matcher on one frame: what does the epipolar pipeline
actually hand us, and why do the template gates reject it?"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap",
                                "low-cost-mocap", "computer_code", "api"))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
os.environ.setdefault("MOCAP_VISIBLE_MODE", "1")

from helpers import find_point_correspondance_and_object_points  # noqa: E402
from multi_body import load_body_templates, _match_template, _score_triangle  # noqa: E402
sys.path.insert(0, ROOT)
from tools.eval_multi_body import load_calib, load_detections  # noqa: E402

CAMS = ["CAM01", "CAM02", "CAM03", "CAM04", "CAM05", "CAM06"]

rig_dir = os.path.join(ROOT, "out", "rig3")
rig, poses = load_calib(rig_dir)
templates = load_body_templates(os.path.join(rig_dir, "rig_blender.json"))
det = load_detections(os.path.join(rig_dir, "detections_2d.csv"))
dummy = np.zeros((8, 8, 3), np.uint8)

for fr in (300, 305):
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
    pts = np.array(pts)
    evs = np.array(evs)
    print(f"\n=== frame {fr}: {len(pts)} triangulated points ===")
    print("err px: min %.2f  med %.2f  p90 %.2f  max %.2f"
          % (evs.min(), np.median(evs), np.percentile(evs, 90), evs.max()))
    print("points with err<5:", int((evs < 5).sum()),
          " err<2:", int((evs < 2).sum()), " err<1:", int((evs < 1).sum()))

    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    for t in templates:
        cands = _match_template(t, pts, dist, 0.030)
        cands_all = _match_template(t, pts, dist, 0.030, max_err=None, errs=None)
        print(f"body {t['body_id']} edges {np.round(t['edges'],3)} -> "
              f"{len(cands)} cands (err-gated) / {len(cands_all)} (ungated)")
        # what is the best edge residual ignoring gates?
        best = None
        for idx in __import__("itertools").combinations(range(len(pts)), 3):
            er, fit, _, _ = _score_triangle(t, t["locals"], pts, idx)
            if best is None or er < best[0]:
                best = (er, idx)
        print(f"   best edge_res = {best[0]*1000:.1f} mm at idx {best[1]} "
              f"(errs {np.round(evs[list(best[1])],2)})")
