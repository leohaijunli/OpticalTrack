"""Compare both tracking pipelines against the SAME body-origin truth.

The historical filter estimates the base midpoint. Convert that estimate to
the origin with its measured rotation and the fixed marker offset before
scoring. Only common valid captures enter the paired comparison; availability
is reported separately. No ground truth enters either tracking pipeline.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adapters/low_cost_mocap"))
from multi_body import load_body_templates


def statistics(values):
    data = np.asarray(values, dtype=float)
    if not len(data):
        return {"count": 0}
    return {"count": len(data), "mean_mm": float(data.mean()),
            "p90_mm": float(np.percentile(data, 90)), "max_mm": float(data.max()),
            "rmse_mm": float(np.sqrt(np.mean(data ** 2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "out/rig3/baseline_midpoint/tracking_from_video.json")
    parser.add_argument("--current", type=Path, default=ROOT / "out/rig3/tracking_from_video.json")
    parser.add_argument("--rig", type=Path, default=ROOT / "out/rig3/rig_blender.json")
    parser.add_argument("--out", type=Path, default=ROOT / "out/rig3/reference_comparison.json")
    args = parser.parse_args()
    old = json.loads(args.baseline.read_text(encoding="utf-8"))
    new = json.loads(args.current.read_text(encoding="utf-8"))
    if new["summary"].get("position_reference_key") != "body_origin":
        raise ValueError("current cache must track the body origin")
    if old["summary"].get("position_reference_key", "base_midpoint") != "base_midpoint":
        raise ValueError("baseline cache must track the base midpoint")
    for key in ("geometry_sha256",):
        if old["summary"][key] != new["summary"][key]:
            raise ValueError("comparison geometry differs")
    if old["summary"]["provenance"]["video_sha256"] != new["summary"]["provenance"]["video_sha256"]:
        raise ValueError("comparison must use identical camera videos")

    baseline = {(r["frame"], o["droneIndex"]): o
                for r in old["records"] for o in r["filtered_objects"]}
    current = {(r["frame"], o["droneIndex"]): o
               for r in new["records"] for o in r["filtered_objects"]}
    truth = {(r["frame"], int(b)): np.asarray(p) for r in new["records"] for b, p in r["reference"].items()}
    result = {"comparison_reference": "same body-frame origin in ENU metres",
              "baseline_conversion": "filtered midpoint minus measured R times base_mid_local",
              "selection": "all common valid frames; missing estimates are not filled",
              "baseline": str(args.baseline), "current": str(args.current), "bodies": {}}
    for template in load_body_templates(args.rig):
        bid = template["body_id"]
        keys = sorted(k for k in baseline.keys() & current.keys() if k[1] == bid)
        before, after, events = [], [], []
        for key in keys:
            previous = baseline[key]
            old_origin = np.asarray(previous["pos"]) - np.asarray(previous["pose_r"]) @ template["base_mid_local"]
            old_error = float(np.linalg.norm(old_origin - truth[key]) * 1000)
            new_error = float(np.linalg.norm(np.asarray(current[key]["pos"]) - truth[key]) * 1000)
            before.append(old_error)
            after.append(new_error)
            if key[0] in (3, 5, 555, 560, 596):
                events.append({"frame": key[0], "before_origin_err_mm": old_error,
                               "after_origin_err_mm": new_error})
        result["bodies"][str(bid)] = {
            "name": template["name"], "common_frames": len(keys),
            "baseline_available": sum(k[1] == bid for k in baseline),
            "current_available": sum(k[1] == bid for k in current),
            "before_origin_error": statistics(before), "after_origin_error": statistics(after),
            "transition_frames": events}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Paired position comparison: SAME body-origin truth, common valid captures")
    for body in result["bodies"].values():
        a, b = body["before_origin_error"], body["after_origin_error"]
        print(f"{body['name']} n={body['common_frames']}: "
              f"mean {a.get('mean_mm', float('nan')):.3f} -> {b.get('mean_mm', float('nan')):.3f} mm; "
              f"max {a.get('max_mm', float('nan')):.3f} -> {b.get('max_mm', float('nan')):.3f} mm")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
