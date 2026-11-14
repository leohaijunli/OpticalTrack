"""Validate the rendered mocap frames against the ground-truth detections.

For each requested frame this script

  1. decodes the six mono PNGs,
  2. finds the bright marker blobs (the markers are the only saturated pixels),
  3. matches them to the predicted pixels in `out/rig3/detections_2d.csv`,
  4. prints the pixel error per camera.

Because image f####.png, the CSV row with frame = #### and the 6-DOF truth all
share the same 0-based frame index, this doubles as a frame-synchronisation
check at moving portions of the trajectory. Shared render manifests and
encoded frame counts separately verify the declared capture clock.

Usage
    python tools/validate_frames.py --frames 0,150,300,450,599
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FRAMES = os.path.join(ROOT, "out", "frames")
CALIB = os.path.join(ROOT, "out", "rig3", "cameras_calib.json")
DET = os.path.join(ROOT, "out", "rig3", "detections_2d.csv")
W, H = 1280, 800

# --rig-dir / --frames-dir override CALIB / DET / the frames subdir



def decode(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    if len(raw) != W * H:
        raise RuntimeError(f"{path}: got {len(raw)} bytes, expected {W * H}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(H, W)


def blobs(img, thresh=200):
    """Connected components of pixels above `thresh`, returned as centroids."""
    mask = img >= thresh
    ys, xs = np.nonzero(mask)
    seen = np.zeros_like(mask)
    out = []
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(pts) >= 2:
            arr = np.array(pts)
            w = img[arr[:, 0], arr[:, 1]].astype(np.float64) - thresh
            if w.sum() > 0:
                out.append(((arr[:, 1] * w).sum() / w.sum(), (arr[:, 0] * w).sum() / w.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0,150,300,450,599")
    ap.add_argument("--radius", type=float, default=4.0,
                    help="max distance (px) between a detection and a blob")
    ap.add_argument("--rig-dir", default=None,
                    help="directory with cameras_calib.json + detections_2d.csv")
    ap.add_argument("--frames-dir", default=None,
                    help="frames subdir template, e.g. 'CAM{cam}_t650'")
    args = ap.parse_args()

    if args.rig_dir:
        calib_path = os.path.join(args.rig_dir, "cameras_calib.json")
        det_path = os.path.join(args.rig_dir, "detections_2d.csv")
    else:
        calib_path, det_path = CALIB, DET
    # {cam} receives the full camera name, e.g. "CAM02"
    frames_pat = args.frames_dir or "{cam}_t650"

    calib = json.load(open(calib_path))
    cams = [c["name"] for c in calib["cameras"]]
    det = {}
    for r in csv.DictReader(open(det_path)):
        det.setdefault((int(r["frame"]), r["cam_name"]), []).append(
            (int(r["marker"]), float(r["u_px"]), float(r["v_px"]))
        )

    frames = [int(f) for f in args.frames.split(",")]
    total = matched = 0
    errs = []
    per_cam = {c: [0, 0, []] for c in cams}

    for f in frames:
        for cam in cams:
            path = os.path.join(FRAMES, frames_pat.format(cam=cam), f"f{f:04d}.png")
            if not os.path.exists(path):
                print(f"missing {path}")
                continue
            img = decode(path)
            found = blobs(img)
            for _, u, v in det.get((f, cam), []):
                total += 1
                if not found:
                    continue
                d = [np.hypot(bx - u, by - v) for bx, by in found]
                j = int(np.argmin(d))
                if d[j] <= args.radius:
                    matched += 1
                    errs.append(d[j])
                    per_cam[cam][0] += 1
                    per_cam[cam][2].append(d[j])
                per_cam[cam][1] += 1

    errs = np.array(errs)
    print(f"frames checked     : {frames}")
    print(f"detections matched : {matched}/{total} ({100.0 * matched / max(total, 1):.1f}%)")
    if len(errs):
        print(f"centroid error     : mean {errs.mean():.3f} px, "
              f"rms {np.sqrt((errs ** 2).mean()):.3f} px, max {errs.max():.2f} px")
    print("per camera:")
    for cam, (m, n, e) in per_cam.items():
        s = f"mean {np.mean(e):.3f} px" if e else "-"
        print(f"  {cam}: {m}/{n} matched, {s}")
    return 0 if total and matched == total else 1


if __name__ == "__main__":
    sys.exit(main())
