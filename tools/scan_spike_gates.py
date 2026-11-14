"""Offline spike-killer scan on the OLD video cache.

The old rendered videos cannot be re-scored against the new marker geometry,
but their stored per-frame triangulated points + reprojection errors + truth
reference let us validate the new gates directly: run locate_bodies() with
the motion gate / error gate on the same points and re-score against the same
reference.  This proves whether a single-frame blip (stray floor highlight
forming a false triangle) is what produced the >2 m spikes, without waiting
for the new render.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
from multi_body import BodyTracker, load_body_templates, locate_bodies  # noqa: E402


def main():
    cache = os.path.join(ROOT, "out", "rig3", "tracking_from_video.json")
    records = json.load(open(cache, encoding="utf-8"))["records"]
    templates = load_body_templates(os.path.join(ROOT, "out", "rig3", "rig_blender.json"))

    def scan(gate, max_err):
        tracker = BodyTracker(templates, gate=gate) if gate else None
        spikes = total = 0
        per = {t["body_id"]: [0, 0] for t in templates}   # [count, >1m]
        for index, record in enumerate(records):
            pts = np.asarray(record["points"], float).reshape(-1, 3)
            errs = np.asarray(record["reprojection_mse_px2"], float).ravel()
            objects = (locate_bodies(pts, errs, templates, tracker=tracker,
                                     tol=0.025, frame=index, dt=1 / 30,
                                     max_err=max_err) if pts.size else [])
            refs = {int(k): np.asarray(v, float) for k, v in record["reference"].items()}
            for obj in objects:
                bid = obj["droneIndex"]
                if bid in refs:
                    per[bid][0] += 1
                    e = float(np.linalg.norm(np.asarray(obj["pos"]) - refs[bid]))
                    total += 1
                    if e > 1.0:
                        spikes += 1
                        per[bid][1] += 1
        return spikes, total, per

    print(f"{'gate(m)':>7} {'maxerr':>6} {'spikes>1m':>10} {'total':>6} "
          f"{'A':>8} {'B':>8} {'C':>8}")
    for gate in (None, 2.0, 1.0, 0.5, 0.3):
        for max_err in (None, 1000.0, 100.0, 25.0):
            spikes, total, per = scan(gate, max_err)
            row = f"{str(gate):>7} {str(max_err):>6} {spikes:>10} {total:>6}"
            for b in (0, 1, 2):
                row += f" {per[b][1]:>6}/{per[b][0]:<2}"
            print(row)


if __name__ == "__main__":
    main()