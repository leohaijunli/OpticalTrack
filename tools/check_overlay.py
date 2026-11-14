"""Overlay the CSV 2D detections for one frame/camera on a rendered still.

Answers one question: does what Blender rendered agree with what the dataset
CSV claims the cameras saw?  Any offset here becomes a triangulation bias.
"""
import csv, os, sys
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    frame = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    cam_name = sys.argv[2] if len(sys.argv) > 2 else "CAM01"
    still = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        ROOT, "out", "frames", f"still_{cam_name}_vis", f"f{frame:04d}.png")
    csvp = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        ROOT, "out", "rig3", "detections_2d.csv")
    out = os.path.join(ROOT, "out", f"overlay_{cam_name}_f{frame:04d}.png")

    img = cv2.imread(still, cv2.IMREAD_GRAYSCALE)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rows = []
    with open(csvp, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["frame"]) == frame and r["cam_name"] == cam_name:
                rows.append(r)
    for r in rows:
        u, v = float(r["u_px"]), float(r["v_px"])
        cv2.drawMarker(vis, (int(round(u)), int(round(v))), (0, 0, 255),
                       cv2.MARKER_CROSS, 11, 1)
        cv2.putText(vis, f"b{r['body_id']}m{r['marker']}", (int(u) + 6, int(v) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out, vis)
    # zoomed 3x crop around the detections
    if rows:
        us = [float(r["u_px"]) for r in rows]
        vs = [float(r["v_px"]) for r in rows]
        x0, x1 = int(min(us)) - 60, int(max(us)) + 60
        y0, y1 = int(min(vs)) - 60, int(max(vs)) + 60
        x0, y0 = max(0, x0), max(0, y0)
        crop = vis[y0:min(vis.shape[0], y1), x0:min(vis.shape[1], x1)]
        cv2.imwrite(out.replace(".png", "_zoom.png"),
                    cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    print(f"{len(rows)} detections on {cam_name} frame {frame} -> {out}")


if __name__ == "__main__":
    main()
