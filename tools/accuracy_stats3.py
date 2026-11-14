"""Position / speed / heading / 6-DOF pose accuracy on the actual video result.

Scores every sample the real-pixel evaluation actually produced; gaps (LOST)
are never filled.  Heading is the *undirected* base axis, measured as the
minimal angular distance on its period (pi): a 180 deg flip is the same axis.
6-DOF rotation is scored separately for rigid (extra-marker-confirmed) frames
and triangle-only frames, whose symmetric base order needs a temporal prior.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "adapters", "low_cost_mocap"))
from multi_body import load_body_templates, POSITION_REFERENCES, POSITION_REFERENCE_LABELS  # noqa: E402


def undirected(a):
    """Wrap an axis angle to [-pi/2, pi/2), retaining its signed direction."""
    return (a + math.pi / 2) % math.pi - math.pi / 2


def axis_error(a, b):
    """Minimal angular distance (rad) between two undirected axes (period pi)."""
    d = (a - b) % math.pi
    return min(d, math.pi - d)


def rotation_error_deg(estimated, truth):
    value = (np.trace(np.asarray(estimated) @ np.asarray(truth).T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def truth_tables(trajectory, templates, fps=30.0, position_reference="body_origin"):
    if position_reference not in POSITION_REFERENCES:
        raise ValueError(f"unknown position reference: {position_reference}")
    with open(trajectory, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    base_mid_local = {t["body_id"]: t["base_mid_local"] for t in templates}
    a0 = {t["body_id"]: t["locals"][t["base_pair"][0]] for t in templates}
    a1 = {t["body_id"]: t["locals"][t["base_pair"][1]] for t in templates}
    pos, quat = {}, {}
    for row in rows:
        b, fr = int(row["body_id"]), int(row["frame"])
        pos.setdefault(b, {})[fr] = np.array([float(row[x]) for x in "xyz"])
        quat[(b, fr)] = np.array([float(row["qw"]), float(row["qx"]),
                                  float(row["qy"]), float(row["qz"])])
    table = {b: {"origin": {}, "pos": {}, "speed": {}, "velocity": {}, "acceleration": {}, "heading": {},
                 "rotation": {}, "tilt": {}} for b in pos}
    for b, frames in pos.items():
        order = sorted(frames)
        prev = None
        prev_velocity = None
        for fr in order:
            p = frames[fr]
            q = quat[(b, fr)]
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            reference = p if position_reference == "body_origin" else p + R @ base_mid_local[b]
            table[b]["origin"][fr] = p
            table[b]["pos"][fr] = reference
            table[b]["rotation"][fr] = R
            table[b]["tilt"][fr] = math.degrees(math.acos(np.clip(R[2, 2], -1.0, 1.0)))
            # The repository reports the NEGATIVE world azimuth of the base
            # axis. Wrapping modulo pi must not identify +45 and -45 degrees:
            # those are perpendicular axes, not the same undirected line.
            world_dir = R @ (a1[b] - a0[b])
            table[b]["heading"][fr] = undirected(-math.atan2(world_dir[1], world_dir[0]))
            dt = (fr - prev[0]) / fps if prev else None
            v = (reference - prev[1]) / dt if prev else None
            table[b]["velocity"][fr] = v
            table[b]["speed"][fr] = None if v is None else float(np.linalg.norm(v))
            table[b]["acceleration"][fr] = ((v - prev_velocity) / dt
                                           if v is not None and prev_velocity is not None else None)
            prev = (fr, reference)
            prev_velocity = v
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=os.path.join(ROOT, "out", "rig3", "tracking_from_video.json"))
    parser.add_argument("--rig-dir", default=os.path.join(ROOT, "out", "rig3"))
    parser.add_argument("--trajectory", default=os.path.join(ROOT, "out", "scenario3.csv"))
    parser.add_argument("--out-prefix", default=os.path.join(ROOT, "out", "rig3", "accuracy3"))
    args = parser.parse_args()

    data = json.load(open(args.cache, encoding="utf-8"))
    records, summary = data["records"], data["summary"]
    templates = load_body_templates(os.path.join(args.rig_dir, "rig_blender.json"))
    reference = summary.get("position_reference_key", "base_midpoint")
    truth = truth_tables(args.trajectory, templates, fps=summary["fps"], position_reference=reference)
    body_ids = [t["body_id"] for t in templates]
    names = {t["body_id"]: t["name"] for t in templates}

    stats = {b: {"pos": [], "speed": [], "velocity": [], "acceleration": [], "heading": [], "cov_pos": 0,
                  "pose_origin": [], "pose_fit": [],
                  "pose_rot": {"rigid": [], "triangle3": []},
                  "pose_modes": {}, "pose_disambiguation": {}} for b in body_ids}
    rows = []
    for record in records:
        fr = record["frame"]
        filtered = {o["droneIndex"]: o for o in record["filtered_objects"]}
        for b in body_ids:
            row = {"frame": fr, "t": record["t"], "body_id": b,
                   "pos_avail": 0, "vel_avail": 0, "acc_avail": 0, "heading_avail": 0, "pose_avail": 0}
            pos_t = truth[b]["pos"].get(fr)
            spd_t = truth[b]["speed"].get(fr)
            hd_t = truth[b]["heading"].get(fr)
            if pos_t is not None:
                for a, v in zip("xyz", pos_t):
                    row[f"tpos_{a}"] = float(v)
            obj = filtered.get(b)
            if pos_t is None or obj is None:
                rows.append(row)
                continue
            if obj.get("position_reference", reference) != reference:
                raise ValueError("cache contains an estimate for a different position reference")
            pos_e = np.asarray(obj["pos"])
            stats[b]["cov_pos"] += 1
            stats[b]["pos"].append(float(np.linalg.norm(pos_e - pos_t)) * 1000)
            row["pos_avail"] = 1
            for a, v in zip("xyz", pos_e):
                row[f"epos_{a}"] = float(v)
            row["pos_err_mm"] = float(np.linalg.norm(pos_e - pos_t)) * 1000
            if spd_t is not None:
                ve = float(np.linalg.norm(np.asarray(obj["vel"])))
                stats[b]["speed"].append(ve - spd_t)
                row["vel_avail"] = 1
                row["tvel_mps"] = spd_t
                row["evel_mps"] = ve
                row["vel_err_mps"] = ve - spd_t
                vector_error = float(np.linalg.norm(
                    np.asarray(obj["vel"]) - truth[b]["velocity"][fr]))
                stats[b]["velocity"].append(vector_error)
                row["velocity_vector_err_mps"] = vector_error
            acc_t = truth[b]["acceleration"].get(fr)
            if acc_t is not None and "acc" in obj:
                acc_error = float(np.linalg.norm(np.asarray(obj["acc"]) - acc_t))
                stats[b]["acceleration"].append(acc_error)
                row["acc_avail"] = 1
                row["acceleration_vector_err_mps2"] = acc_error
                for axis, expected, estimated in zip("xyz", acc_t, obj["acc"]):
                    row[f"tacc_{axis}"] = float(expected)
                    row[f"eacc_{axis}"] = float(estimated)
            if hd_t is not None:
                he = axis_error(float(obj["heading"]), hd_t)
                stats[b]["heading"].append(math.degrees(he))
                row["heading_avail"] = 1
                row["theading_deg"] = math.degrees(hd_t)
                row["eheading_deg"] = math.degrees(undirected(float(obj["heading"])))
                row["heading_err_deg"] = math.degrees(he)
            pose = obj.get("pose_t")
            if pose is not None and fr in truth[b]["rotation"]:
                mode = obj.get("pose_mode", "triangle3")
                stats[b]["pose_modes"][mode] = stats[b]["pose_modes"].get(mode, 0) + 1
                source = obj.get("pose_disambiguation", "unreported")
                counts = stats[b]["pose_disambiguation"]
                counts[source] = counts.get(source, 0) + 1
                origin = float(np.linalg.norm(
                    np.asarray(pose) - truth[b]["origin"][fr])) * 1000
                rot = rotation_error_deg(obj["pose_r"], truth[b]["rotation"][fr])
                stats[b]["pose_origin"].append(origin)
                stats[b]["pose_rot"].setdefault(mode, []).append(rot)
                stats[b]["pose_fit"].append(float(obj.get("pose_rmse_m", 0.0)) * 1000)
                row["pose_avail"] = 1
                row["pose_mode"] = mode
                row["pose_disambiguation"] = source
                row["pose_origin_err_mm"] = origin
                row["pose_rot_err_deg"] = rot
                row["pose_fit_rmse_mm"] = stats[b]["pose_fit"][-1]
            rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    fields = ["frame", "t", "body_id", "pos_avail", "vel_avail", "acc_avail", "heading_avail",
              "pose_avail"]
    fields += [f"{p}_{a}" for p in ("tpos", "epos") for a in "xyz"]
    fields += [f"{p}_{a}" for p in ("tacc", "eacc") for a in "xyz"]
    fields += ["pos_err_mm", "tvel_mps", "evel_mps", "vel_err_mps", "velocity_vector_err_mps",
                "theading_deg", "eheading_deg", "heading_err_deg",
                "pose_mode", "pose_disambiguation", "pose_origin_err_mm", "pose_rot_err_deg",
                "pose_fit_rmse_mm"]
    fields += ["acceleration_vector_err_mps2"]
    with open(args.out_prefix + ".csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def summ(values, unit=1.0):
        v = np.asarray(values)
        if not v.size:
            return {"count": 0}
        return {"count": int(v.size),
                "mean": float(np.mean(v) * unit), "median": float(np.median(v) * unit),
                 "p90": float(np.percentile(v, 90) * unit), "max": float(np.max(v) * unit),
                 "rmse": float(np.sqrt(np.mean(v ** 2)) * unit),
                "std": float(np.std(v) * unit)}

    def speed_stats(values):
        v = np.asarray(values, dtype=float)
        if not v.size:
            return {"count": 0}
        return {"count": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
                "mae": float(np.mean(np.abs(v))),
                "rmse": float(np.sqrt(np.mean(v ** 2))),
                "p90_abs": float(np.percentile(np.abs(v), 90)),
                "max_abs": float(np.max(np.abs(v)))}

    result = {"fps": summary["fps"], "frame_count": len(records),
              "geometry_sha256": summary.get("geometry_sha256", {}),
              "position_reference_key": reference,
              "position_reference": POSITION_REFERENCE_LABELS[reference],
              "velocity_reference": "backward difference of the selected reference position",
              "acceleration_reference": "second backward difference of the selected reference position",
              "kalman_state": ["px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "az"],
              "heading_reference": "negative world base-axis azimuth, period pi",
              "pose_reference": "body-frame origin and rotation; raw pose passed through Kalman",
              "bodies": {}}
    for b in body_ids:
        s = stats[b]
        rigid = s["pose_rot"].get("rigid", [])
        triangle = s["pose_rot"].get("triangle3", [])
        result["bodies"][str(b)] = {
            "name": names[b],
            "position_coverage_percent": 100 * s["cov_pos"] / len(records),
            "position_err_mm": summ(s["pos"]),
             "speed_err_mps": speed_stats(s["speed"]),
              "velocity_vector_err_mps": summ(s["velocity"]),
             "acceleration_vector_err_mps2": summ(s["acceleration"]),
            "heading_err_deg": summ(s["heading"]),
            "pose": {
                "coverage_percent": 100 * len(s["pose_origin"]) / len(records),
                 "mode_counts": dict(sorted(s["pose_modes"].items())),
                 "disambiguation_counts": dict(sorted(s["pose_disambiguation"].items())),
                "origin_err_mm": summ(s["pose_origin"]),
                "fit_rmse_mm": summ(s["pose_fit"]),
                "rigid_rotation_err_deg": summ(rigid),
                "triangle3_rotation_err_deg": summ(triangle),
            },
        }
    with open(args.out_prefix + ".json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)

    print("position/speed/heading/pose accuracy (Kalman, actual video pixels):")
    for b in body_ids:
        r = result["bodies"][str(b)]
        p, sp, h = r["position_err_mm"], r["speed_err_mps"], r["heading_err_deg"]
        print(f"{r['name']}  cov {r['position_coverage_percent']:5.1f}%")
        if p.get("count"):
            print(f"   position {p['mean']:6.1f}+/-{p['std']:5.1f} mm  "
                  f"(med {p['median']:5.1f}, p90 {p['p90']:5.1f}, max {p['max']:6.0f}, n{p['count']})")
        if sp.get("count"):
            print(f"   speed    MAE {sp['mae']:5.2f}  RMSE {sp['rmse']:5.2f} m/s  "
                  f"(bias {sp['mean']:+.2f}, max {sp['max_abs']:4.2f}, n{sp['count']})")
        if h.get("count"):
            print(f"   heading  {h['mean']:5.1f}+/-{h['std']:5.1f} deg  "
                  f"(p90 {h['p90']:5.1f}, n{h['count']})")
        pose = r["pose"]
        rigid, triangle = pose["rigid_rotation_err_deg"], pose["triangle3_rotation_err_deg"]
        if pose["origin_err_mm"].get("count"):
            print(f"   pose     cov {pose['coverage_percent']:5.1f}% {pose['mode_counts']}  "
                  f"origin {pose['origin_err_mm']['mean']:.2f} mm, "
                  f"rot rigid {rigid.get('mean', float('nan')):.3f} deg / "
                  f"triangle {triangle.get('mean', float('nan')):.2f} deg")
    print("wrote", args.out_prefix + ".csv", "and .json")


if __name__ == "__main__":
    main()
