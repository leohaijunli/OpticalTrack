"""Encode the multi-drone rendered frames into the delivery clips.

Same encoder settings as the single-body dataset (libx264 / crf 16 / slow /
yuv420p / +faststart); frame f of every camera is the same scene instant,
so the six clips stay synchronised by construction.

    in : out/frames/CAM0x_t650/f0000.png ... f0599.png
    out: out/video/0x_cam0x_t650_20s.mp4        clean mono plate
         out/video/0x_cam0x_t650_20s_hud.mp4   camera id / frame / timecode
"""
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
F = os.path.join(ROOT, "out", "frames")
V = os.path.join(ROOT, "out", "video")
FONT = "C\\:/Windows/Fonts/consola.ttf"
FONTB = "C\\:/Windows/Fonts/consolab.ttf"

START, COUNT, FPS = 0, 600, 30
CAMERAS = tuple(f"CAM{i:02d}" for i in range(1, 7))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_render_inputs(tag):
    """Require a complete, common-source six-camera sequence before encoding."""
    manifests = []
    expected = {f"f{frame:04d}.png" for frame in range(START, START + COUNT)}
    for camera in CAMERAS:
        directory = Path(F) / f"{camera}_{tag}"
        with (directory / "render_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (manifest.get("status") != "complete" or manifest.get("rendered_count") != COUNT
                or manifest.get("shot") != camera or manifest.get("frames") != [START, START + COUNT - 1]
                or manifest.get("resolution") != [1280, 800] or manifest.get("fps") != FPS
                or manifest.get("band") != "vis"):
            raise ValueError(f"{camera}: render is incomplete or does not match the delivery settings")
        files = list(directory.glob("f[0-9][0-9][0-9][0-9].png"))
        if {path.name for path in files} != expected or any(path.stat().st_size == 0 for path in files):
            raise ValueError(f"{camera}: missing, empty, or extra frames")
        if manifests and manifest["source_sha256"] != manifests[0]["source_sha256"]:
            raise ValueError(f"{camera}: render sources differ from CAM01")
        manifests.append(manifest)
    return manifests


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if p.returncode != 0:
        print(p.stderr[-1500:])
        raise RuntimeError(f"encoder failed with exit code {p.returncode}")


def encode_clip(command, destination):
    partial = os.path.splitext(destination)[0] + ".partial.mp4"
    try:
        run(command + [partial])
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", partial],
                               check=True, capture_output=True, text=True, encoding="utf-8")
        stream = json.loads(probe.stdout)["streams"][0]
        if (int(stream["nb_frames"]) != COUNT or Fraction(stream["avg_frame_rate"]) != FPS
                or [stream["width"], stream["height"]] != [1280, 800]
                or stream["pix_fmt"] != "yuv420p"):
            raise RuntimeError(f"encoded clip has incorrect frame count, clock, or format: {destination}")
        os.replace(partial, destination)
        return {"file": os.path.basename(destination), "sha256": file_sha256(destination)}
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def main():
    os.makedirs(V, exist_ok=True)
    tag = sys.argv[1] if len(sys.argv) > 1 else "t650"
    manifests = validate_render_inputs(tag)
    delivery = {"tag": tag, "resolution": [1280, 800], "fps": FPS,
                "frames": [START, START + COUNT - 1], "frame_count": COUNT,
                "source_sha256": manifests[0]["source_sha256"], "clips": []}
    label = "Tarot Ironman 650 x3  -  4 markers / body (rigid ID)"
    for camera in CAMERAS:
        c = camera[-2:]
        src = os.path.join(F, f"CAM{c}_{tag}", "f%04d.png")
        print(f"[encode] CAM{c}", flush=True)
        clean = os.path.join(V, f"{c}_cam{c}_{tag}_20s.mp4")
        clean_record = encode_clip(["ffmpeg", "-y", "-start_number", str(START), "-framerate", str(FPS),
             "-i", src, "-frames:v", str(COUNT),
             "-c:v", "libx264", "-preset", "slow", "-crf", "16",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart"], clean)

        vf = (
            f"drawtext=fontfile='{FONTB}':text='CAM {c}  |  OV9281 mono  1280x800'"
            f":x=36:y=28:fontsize=22:fontcolor=white@0.92,"
            f"drawtext=fontfile='{FONT}':text='M27280M07S 2.8 mm  |  HFOV 68.9 deg"
            f"  |  global shutter':x=36:y=62:fontsize=15:fontcolor=white@0.58,"
            f"drawtext=fontfile='{FONT}':text='f %{{eif\\:n\\:d\\:4}}   t %{{pts\\:hms}}'"
            f":x=w-tw-36:y=32:fontsize=17:fontcolor=white@0.80,"
            f"drawtext=fontfile='{FONT}':text='{label}'"
            f":x=36:y=h-46:fontsize=15:fontcolor=white@0.58,"
            f"drawtext=fontfile='{FONT}':text='20 s @ 30 fps'"
            f":x=w-tw-36:y=h-46:fontsize=15:fontcolor=white@0.58")
        hud = os.path.join(V, f"{c}_cam{c}_{tag}_20s_hud.mp4")
        hud_record = encode_clip(["ffmpeg", "-y", "-start_number", str(START), "-framerate", str(FPS),
             "-i", src, "-frames:v", str(COUNT), "-vf", vf,
             "-c:v", "libx264", "-preset", "slow", "-crf", "16",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart"], hud)
        delivery["clips"].append({"camera": camera, "clean": clean_record, "hud": hud_record})
    manifest_path = os.path.join(V, f"{tag}_manifest.json")
    with open(manifest_path + ".partial", "w", encoding="utf-8") as handle:
        json.dump(delivery, handle, indent=2)
    os.replace(manifest_path + ".partial", manifest_path)
    print("done ->", V)


if __name__ == "__main__":
    main()
