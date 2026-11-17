"""Plot body-origin position-error distributions and compare video evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def series(records, body):
    filtered, raw = [], []
    for record in records:
        truth = np.asarray(record["reference"][str(body)], dtype=float)
        for field, result in (("filtered_objects", filtered), ("objects", raw)):
            found = next((obj for obj in record[field] if obj["droneIndex"] == body), None)
            result.append([np.nan] * 3 if found is None else
                          ((np.asarray(found["pos"]) - truth) * 1000).tolist())
    return (np.asarray(filtered, dtype=float).reshape(-1, 3),
            np.asarray(raw, dtype=float).reshape(-1, 3))


def summarize(vectors, frames, fps):
    vectors = np.asarray(vectors, dtype=float)
    frames = np.asarray(frames)
    if (frames.ndim != 1 or not frames.size or vectors.shape != (len(frames), 3)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("nonempty matching (N,3) errors/frame IDs and positive finite FPS are required")
    if np.any(np.diff(frames) != 1):
        raise ValueError("use contiguous frame IDs to measure missing duration")
    valid = np.isfinite(vectors).all(axis=1)
    values = np.linalg.norm(vectors[valid], axis=1)
    longest = current = 0
    for available in valid:
        current = 0 if available else current + 1
        longest = max(longest, current)
    stats = {"valid_frames": int(valid.sum()), "missing_frames": int((~valid).sum()),
             "coverage_percent": float(valid.mean() * 100),
             "missing_frame_ids": np.asarray(frames)[~valid].tolist(),
             "longest_missing_frames": longest, "longest_missing_duration_s": longest / fps}
    if not len(values):
        stats.update(dict.fromkeys(("mean_mm", "rmse_mm", "p50_mm", "p90_mm", "p95_mm", "p99_mm",
                                    "max_mm", "over_5mm_valid_percent", "over_10mm_valid_percent")))
        stats.update(bias_xyz_mm=[None] * 3, std_xyz_mm=[None] * 3,
                     within_5mm_all_frames_percent=0.0, largest_errors=[])
        return stats
    stats.update({"mean_mm": float(values.mean()), "rmse_mm": float(np.sqrt(np.mean(values ** 2))),
                  "p50_mm": float(np.percentile(values, 50)), "p90_mm": float(np.percentile(values, 90)),
                  "p95_mm": float(np.percentile(values, 95)), "p99_mm": float(np.percentile(values, 99)),
                  "max_mm": float(values.max()),
                  "bias_xyz_mm": vectors[valid].mean(axis=0).tolist(),
                  "std_xyz_mm": vectors[valid].std(axis=0).tolist(),
                  "over_5mm_valid_percent": float(np.mean(values > 5) * 100),
                  "over_10mm_valid_percent": float(np.mean(values > 10) * 100),
                  "within_5mm_all_frames_percent": float(np.sum(values <= 5) / len(frames) * 100)})
    valid_indices = np.flatnonzero(valid)
    stats["largest_errors"] = [{"frame": int(frames[i]), "t": float(frames[i] / fps),
                                "error_mm": float(np.linalg.norm(vectors[i])),
                                "xyz_mm": vectors[i].tolist()}
                               for i in valid_indices[np.argsort(values)[-5:][::-1]]]
    return stats


def setup_plots():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#94a3b8", "axes.labelcolor": "#334155",
                         "xtick.color": "#475569", "ytick.color": "#475569",
                         "figure.facecolor": "white"})


def number(value, signed=False):
    return "—" if value is None else (f"{value:+.3f}" if signed else f"{value:.3f}")


def comparison_configuration(current, baseline, allow_centroid_change=False, allow_change=()):
    """Reject comparisons whose configuration differs, except where asked.

    ``allow_change`` names estimator parameters that a run is supposed to
    change (a tuning A/B).  Everything else must still match, so an accidental
    difference in calibration, frame range or detector setup is still caught.
    """
    current, baseline = dict(current), dict(baseline)
    mode_current = current.pop("centroid_mode", "integer")
    mode_baseline = baseline.pop("centroid_mode", "integer")
    if mode_current not in ("integer", "subpixel") or mode_baseline not in ("integer", "subpixel"):
        raise ValueError("unsupported centroid mode in comparison input")
    permitted = {}
    for key in allow_change:
        if key not in current and key not in baseline:
            raise ValueError(f"--allow-change {key}: neither input records this parameter")
        if current.get(key) != baseline.get(key):
            permitted[key] = {"current": current.get(key), "baseline": baseline.get(key)}
        current.pop(key, None)
        baseline.pop(key, None)
    if current != baseline:
        differing = sorted(key for key in set(current) | set(baseline)
                           if current.get(key) != baseline.get(key))
        raise ValueError("comparison requires matching configuration apart from an explicitly "
                         f"allowed centroid change; still differing: {', '.join(differing)}")
    if mode_current != mode_baseline:
        if not allow_centroid_change:
            raise ValueError("different centroid modes require --allow-centroid-change")
        permitted["centroid_mode"] = {"current": mode_current, "baseline": mode_baseline}
    return permitted


def plot_distributions(bodies, data, previous, stats, frame_count, fps, prefix, camera_count,
                       current_label="优化后", baseline_label="优化前"):
    setup_plots()
    colors = ["#2563eb", "#d97706", "#059669"]
    maximum = max((np.linalg.norm(v[np.isfinite(v).all(axis=1)], axis=1).max()
                   for v in [*data.values(), *previous.values()] if np.isfinite(v).all(axis=1).any()), default=5)
    limit = max(5.0, float(np.ceil(maximum)))
    bins = np.linspace(0, limit, int(np.ceil(limit / .5)) + 1)
    fig, axes = plt.subplots(2, len(bodies), figsize=(5 * len(bodies), 8.3), squeeze=False)
    for column, body in enumerate(bodies):
        key = body["body_id"]
        color = colors[column % len(colors)]
        current = np.linalg.norm(data[key][np.isfinite(data[key]).all(axis=1)], axis=1)
        old = np.linalg.norm(previous[key][np.isfinite(previous[key]).all(axis=1)], axis=1)
        stat = stats[str(key)]["filtered"]
        ax = axes[0, column]
        if len(current):
            ax.hist(current, bins=bins, weights=np.full(len(current), 100 / len(current)),
                    color=color, alpha=.65, label=current_label)
        if len(old):
            ax.hist(old, bins=bins, weights=np.full(len(old), 100 / len(old)), histtype="step",
                    color="#334155", linestyle="--", linewidth=1.5, label=baseline_label)
        ax.set_title(f"{body['name']}  |  有效 {stat['valid_frames']}/{frame_count} 帧\n"
                     f"均值 {number(stat['mean_mm'])} mm，RMSE {number(stat['rmse_mm'])} mm",
                     fontsize=12, pad=12)
        ax.set(xlabel="机体原点位置误差 (mm)", ylabel="有效样本比例 (%)", xlim=(0, limit))
        ax.grid(axis="y", alpha=.18)
        ax = axes[1, column]
        for values, label, line_color, style in ((old, baseline_label, "#334155", "--"),
                                                 (current, current_label, color, "-")):
            if len(values):
                ax.step(np.r_[0, np.sort(values), limit],
                        np.r_[0, np.arange(1, len(values) + 1) * 100 / len(values), 100],
                        where="post", label=label, color=line_color, linestyle=style, linewidth=1.8)
        ax.axhline(95, color="#94a3b8", linestyle=":", linewidth=.9)
        ax.set(xlabel="机体原点位置误差 (mm)", ylabel="累计比例 (%)", xlim=(0, limit), ylim=(0, 102))
        ax.grid(alpha=.18)
        ax.set_title(f"P95 {number(stat['p95_mm'])}   P99 {number(stat['p99_mm'])}   "
                     f"最大 {number(stat['max_mm'])} mm", fontsize=11, pad=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.99, .955), frameon=False, ncol=2)
    fig.suptitle(f"{current_label}定位误差分布与累计分布  |  {camera_count} 相机 · {frame_count} 帧 · {fps:g} fps 素材",
                 fontsize=16, y=.995)
    fig.text(.02, .025, "统计参考点：机体几何原点；误差为 Kalman 输出与同帧真值的三维距离。\n"
              "直方图与累计分布按有效样本统计；无解帧计入覆盖率，不按零误差计分。未使用真值反向校正估计。",
             fontsize=10, color="#475569")
    fig.tight_layout(rect=(0, .08, 1, .93), h_pad=2.5, w_pad=2)
    fig.savefig(str(prefix) + ".png", dpi=170)
    plt.close(fig)

    finite = np.vstack([v[np.isfinite(v).all(axis=1)] for v in data.values()])
    component_limit = max(5.0, float(np.ceil(np.abs(finite).max()))) if finite.size else 5.0
    component_bins = np.linspace(-component_limit, component_limit, int(component_limit * 4) + 1)
    fig, axes = plt.subplots(1, len(bodies), figsize=(5 * len(bodies), 4.5), squeeze=False)
    for column, body in enumerate(bodies):
        ax = axes[0, column]
        valid = data[body["body_id"]]
        valid = valid[np.isfinite(valid).all(axis=1)]
        for index, (axis, color) in enumerate(zip("XYZ", colors)):
            if len(valid):
                ax.hist(valid[:, index], bins=component_bins, weights=np.full(len(valid), 100 / len(valid)),
                        histtype="step", color=color, label=axis, linewidth=1.6)
        stat = stats[str(body["body_id"])]["filtered"]
        bias = stat["bias_xyz_mm"]
        ax.set_title(f"{body['name']}\n平均偏差 XYZ = {', '.join(number(v, True) for v in bias)} mm",
                     fontsize=11, pad=12)
        ax.set(xlabel="估计位置 − 真值 (mm)", ylabel="有效样本比例 (%)", xlim=(-component_limit, component_limit))
        ax.axvline(0, color="#94a3b8", linestyle=":", linewidth=.8)
        ax.grid(axis="y", alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.99, .955), ncol=3, frameon=False)
    fig.suptitle(f"{current_label}定位的 XYZ 分量误差分布", fontsize=16, y=.995)
    fig.tight_layout(rect=(0, .02, 1, .87), w_pad=2)
    fig.savefig(str(prefix) + "_xyz.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "out/rig3/tracking_optimized.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "out/rig3/tracking_from_video.json")
    parser.add_argument("--out-prefix", type=Path, default=ROOT / "out/rig3/position_error_distribution")
    parser.add_argument("--allow-centroid-change", action="store_true",
                        help="compare integer/subpixel modes; all other parameters must still match")
    parser.add_argument("--allow-change", action="append", default=[], metavar="KEY",
                        help="configuration key this comparison is meant to change (repeatable); "
                             "every other key must still match")
    args = parser.parse_args()
    current = json.loads(args.cache.read_text(encoding="utf-8"))
    previous = json.loads(args.baseline.read_text(encoding="utf-8"))
    changes = comparison_configuration(current["summary"]["configuration"],
                                       previous["summary"]["configuration"], args.allow_centroid_change,
                                       args.allow_change)
    for key in ("geometry_sha256", "selected_frames", "provenance", "truth_source"):
        if current["summary"][key] != previous["summary"][key]:
            raise ValueError(f"comparison requires matching {key}")
    if current["summary"]["position_reference_key"] != "body_origin":
        raise ValueError("this report requires the body-origin reference")
    frames = current["summary"]["selected_frames"]
    if any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError("use a contiguous sequence to report loss duration")
    fps = current["summary"]["fps"]
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError("a nonempty sequence with positive FPS is required")
    for label, cache in (("current", current), ("baseline", previous)):
        if [record.get("frame") for record in cache["records"]] != frames:
            raise ValueError(f"{label} records do not match their declared frame sequence")
    bodies = current["summary"]["bodies"]
    data, old_data, statistics, rows = {}, {}, {}, []
    for body in bodies:
        key = body["body_id"]
        filtered, raw = series(current["records"], key)
        old_filtered, old_raw = series(previous["records"], key)
        data[key], old_data[key] = filtered, old_filtered
        stats = {"name": body["name"], "filtered": summarize(filtered, frames, fps),
                 "raw": summarize(raw, frames, fps), "baseline_filtered": summarize(old_filtered, frames, fps)}
        same_valid = np.array_equal(np.isfinite(filtered), np.isfinite(old_filtered))
        paired = np.isfinite(filtered).all(axis=1) & np.isfinite(old_filtered).all(axis=1)
        paired_raw = np.isfinite(raw).all(axis=1) & np.isfinite(old_raw).all(axis=1)
        stats["comparison"] = {"availability_identical": same_valid, "paired_frames": int(paired.sum()),
                               "raw_availability_identical": np.array_equal(np.isfinite(raw), np.isfinite(old_raw)),
                               "max_filtered_change_mm": float(np.linalg.norm(filtered[paired] - old_filtered[paired], axis=1).max()) if paired.any() else None,
                               "max_raw_change_mm": float(np.linalg.norm(raw[paired_raw] - old_raw[paired_raw], axis=1).max()) if paired_raw.any() else None}
        paired_current = np.linalg.norm(filtered[paired], axis=1)
        paired_baseline = np.linalg.norm(old_filtered[paired], axis=1)
        stats["comparison"].update({
            "paired_current_mean_mm": float(paired_current.mean()) if paired.any() else None,
            "paired_baseline_mean_mm": float(paired_baseline.mean()) if paired.any() else None,
            "paired_current_p95_mm": float(np.percentile(paired_current, 95)) if paired.any() else None,
            "paired_baseline_p95_mm": float(np.percentile(paired_baseline, 95)) if paired.any() else None})
        statistics[str(key)] = stats
        row = {"body_id": key, "name": body["name"]}
        row.update({k: v for k, v in stats["filtered"].items() if not isinstance(v, (list, dict))})
        rows.append(row)
    report = {"cache": str(args.cache), "baseline": str(args.baseline),
              "frame_count": len(frames), "fps_capture": fps,
              "reference": "body_origin; calibrated world; millimetres",
              "note": "Euclidean filtered position error; quantiles conditional on valid tracking",
              "algorithm_sha256": current["summary"]["algorithm_sha256"], "bodies": statistics,
              "configuration_changes": changes}
    report["all_bodies_available_percent"] = float(100 * np.logical_and.reduce(
        [np.isfinite(vectors).all(axis=1) for vectors in data.values()]).mean())
    report["baseline_all_bodies_available_percent"] = float(100 * np.logical_and.reduce(
        [np.isfinite(vectors).all(axis=1) for vectors in old_data.values()]).mean())
    for key, path in (("cache_sha256", args.cache), ("baseline_sha256", args.baseline), ("script_sha256", Path(__file__))):
        with path.open("rb") as handle:
            report[key] = hashlib.file_digest(handle, "sha256").hexdigest()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with args.out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = {"integer": "整数质心", "subpixel": "浮点质心"}
    centroid = changes.get("centroid_mode")
    current_label, baseline_label = ("优化后", "优化前") if not centroid else (
        labels[centroid["current"]], labels[centroid["baseline"]])
    plot_distributions(bodies, data, old_data, statistics, len(frames), fps, args.out_prefix,
                       len(current["summary"]["camera_names"]), current_label, baseline_label)
    for stats in statistics.values():
        values = stats["filtered"]
        print(f"{stats['name']}: coverage={values['coverage_percent']:.2f}%, mean={number(values['mean_mm'])}, "
              f"P95={number(values['p95_mm'])}, P99={number(values['p99_mm'])}, max={number(values['max_mm'])} mm; "
              f"delta={number(stats['comparison']['max_filtered_change_mm'])} mm")
    print("wrote", args.out_prefix, "(.json/.csv/.png and _xyz.png)")


if __name__ == "__main__":
    main()
