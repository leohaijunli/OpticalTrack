"""Multi-camera video source for the Low-Cost-Mocap adapter.

Drop-in replacement for the ``pseyepy.Camera`` object that
``api/helpers.py`` uses as its live-camera source.  The point of this module
is that *nothing downstream knows where the pixels come from*: the same class
replays the rendered Blender clips, a folder of PNG frames, six RTSP cameras,
or six USB devices.

API compatibility (the only members helpers.py touches):

    Camera.RES_SMALL          -> nominal resolution tag
    Camera(fps, resolution,
           gain, exposure)    -> opens every configured source
    cam.exposure / cam.gain   -> mutable lists (len = number of cameras)
    cam.read()                -> (frames, timestamps)
                                 frames: list of BGR ndarrays, one per camera
                                 timestamps: 1-D float array of frame times

Selecting the source
--------------------
``MOCAP_SOURCES`` is a comma separated list, one entry per camera, in CAM01..
order.  An entry can be:

    out/video/01_cam01_t650_20s.mp4        a video file
    rtsp://10.0.0.11:8554/cam01            any stream OpenCV can open
    http://host:8080/video                 MJPEG-over-HTTP
    0                                      a local capture device index
    out/frames/CAM01_t650                  a directory of PNG/JPG frames

When ``MOCAP_SOURCES`` is unset the module falls back to the rendered dataset:
the ``0?_cam0?_visible_20s.mp4`` clips in ``MOCAP_VIDEO_DIR``.

Timing
------
* ``MOCAP_SYNC_MODE=pace`` (default for files): frames are served on a fixed
  clock at the clip framerate, so replay looks like a live 30 fps capture.
* ``MOCAP_SYNC_MODE=native`` (default for streams/devices): frames are served
  as fast as the source produces them, timestamps come from the source.

File replay is frame-index synchronised.  Generic OpenCV RTSP/USB reads use
host-arrival timestamps, not hardware exposure timestamps: they do not prove
camera synchronisation.  Real precision mocap requires shared trigger/PTP
and calibrated camera timestamps from a suitable capture backend.
"""
from __future__ import annotations

import glob
import os
import re
import time

import numpy as np
import cv2 as cv

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_STREAM_PREFIX = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def default_project_root() -> str:
    # adapters/low_cost_mocap/video_replay.py -> <repo>/  (two levels up)
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def discover_clips(source_dir: str | None = None, tag: str | None = None) -> list[str]:
    """Rendered dataset clips (clean, non-HUD) ordered CAM01..CAM06."""
    if not source_dir:
        source_dir = os.environ.get(
            "MOCAP_VIDEO_DIR",
            os.path.join(default_project_root(), "out", "video"))
    tag = tag or os.environ.get("MOCAP_VIDEO_TAG", "visible")
    clips = sorted(glob.glob(os.path.join(source_dir, f"0?_cam0?_{tag}_20s.mp4")))
    numbers = []
    for path in clips:
        match = re.fullmatch(r"(\d{2})_cam(\d{2})_.*_20s\.mp4", os.path.basename(path))
        if match is None or match[1] != match[2]:
            raise ValueError(f"inconsistent camera numbering: {path}")
        numbers.append(int(match[1]))
    if numbers != list(range(1, 7)):
        raise FileNotFoundError(f"Expected exactly CAM01..CAM06 for tag={tag!r} in {source_dir}")
    return clips


# --------------------------------------------------------------------- readers
class _Reader:
    """One camera: open, read a BGR frame, report a capture timestamp."""

    kind = "reader"

    def __init__(self, spec: str):
        self.spec = spec

    def read(self):
        raise NotImplementedError

    def timestamp(self) -> float | None:
        return None

    @property
    def fps(self) -> float:
        return 30.0

    def release(self):
        pass


class _VideoFileReader(_Reader):
    kind = "file"

    def __init__(self, spec: str, loop: bool = True):
        super().__init__(spec)
        self.loop = loop
        self.cap = cv.VideoCapture(spec)
        if not self.cap.isOpened():
            raise OSError(f"cannot open video source: {spec}")
        self._fps = float(self.cap.get(cv.CAP_PROP_FPS)) or 30.0
        self._frame = -1
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return max(1.0, self._fps)

    def read(self):
        ok, img = self.cap.read()
        if not ok or img is None:
            if not self.loop:
                return None
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
            self._frame = -1
            if not ok:
                return None
        self._frame += 1
        return _as_bgr(img)

    def timestamp(self) -> float:
        return self._frame / self.fps

    def release(self):
        self.cap.release()


class _StreamReader(_Reader):
    """RTSP / HTTP(MJPEG) / UDP.  Reopens itself when the feed stalls."""

    kind = "stream"

    def __init__(self, spec: str, reopen_after: float = 1.0):
        super().__init__(spec)
        self._reopen_after = reopen_after
        self._open()
        self._fail_since = None

    def _open(self):
        # FFMPEG backend is the only one that reliably handles rtsp/http
        self.cap = cv.VideoCapture(self.spec, cv.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv.VideoCapture(self.spec)
        self._fps = float(self.cap.get(cv.CAP_PROP_FPS)) or 30.0

    @property
    def fps(self) -> float:
        return max(1.0, self._fps)

    def read(self):
        ok, img = self.cap.read()
        now = time.monotonic()
        if not ok or img is None:
            if self._fail_since is None:
                self._fail_since = now
            elif now - self._fail_since > self._reopen_after:
                self.cap.release()
                self._open()
                self._fail_since = None
            return None
        self._fail_since = None
        return _as_bgr(img)

    def timestamp(self) -> float | None:
        ms = self.cap.get(cv.CAP_PROP_POS_MSEC)
        return float(ms) / 1000.0 if ms and ms > 0 else None

    def release(self):
        self.cap.release()


class _DeviceReader(_Reader):
    kind = "device"

    def __init__(self, index: int, backend=None):
        super().__init__(str(index))
        self.cap = (cv.VideoCapture(index, backend) if backend is not None
                    else cv.VideoCapture(index))
        if not self.cap.isOpened():
            raise OSError(f"cannot open capture device {index}")
        self._fps = float(self.cap.get(cv.CAP_PROP_FPS)) or 30.0

    @property
    def fps(self) -> float:
        return max(1.0, self._fps)

    def read(self):
        ok, img = self.cap.read()
        return _as_bgr(img) if (ok and img is not None) else None

    def timestamp(self) -> float | None:
        ms = self.cap.get(cv.CAP_PROP_POS_MSEC)
        return float(ms) / 1000.0 if ms and ms > 0 else None

    def release(self):
        self.cap.release()


class _ImageDirReader(_Reader):
    kind = "dir"

    def __init__(self, path: str, loop: bool = True):
        super().__init__(path)
        self.loop = loop
        files = [f for f in sorted(os.listdir(path))
                 if f.lower().endswith(_IMAGE_EXT) and os.path.isfile(os.path.join(path, f))]
        if not files:
            raise FileNotFoundError(f"no image sequence in {path}")
        self.files = [os.path.join(path, f) for f in files]
        self._i = -1
        self.frame_count = len(self.files)

    def read(self):
        self._i += 1
        if self._i >= len(self.files):
            if not self.loop:
                return None
            self._i = 0
        img = cv.imread(self.files[self._i], cv.IMREAD_UNCHANGED)
        return _as_bgr(img) if img is not None else None

    def timestamp(self) -> float:
        return self._i / self.fps


def _as_bgr(img):
    """Every reader hands back 3-channel BGR: helpers.py assumes BGR."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    return img


def open_reader(spec: str, loop: bool = True) -> _Reader:
    s = spec.strip()
    if s.isdigit():
        return _DeviceReader(int(s))
    if s.lower().startswith(_STREAM_PREFIX):
        return _StreamReader(s)
    if os.path.isdir(s):
        return _ImageDirReader(s, loop=loop)
    return _VideoFileReader(s, loop=loop)


def _resolve_path(p: str) -> str:
    """Relative paths in MOCAP_SOURCES are relative to the project root, not
    to whatever directory the server happened to be launched from."""
    if os.path.isabs(p) or p.isdigit() or p.lower().startswith(_STREAM_PREFIX):
        return p
    if os.path.exists(p):
        return p
    root = default_project_root()
    cand = os.path.join(root, p)
    return cand if os.path.exists(cand) else p


def resolve_sources(explicit: str | None = None) -> list[str]:
    """Camera order is CAM01..CAM06, matching the calibration file."""
    raw = explicit if explicit is not None else os.environ.get("MOCAP_SOURCES", "")
    if raw.strip():
        return [_resolve_path(p.strip()) for p in raw.split(",") if p.strip()]
    return discover_clips()


class VideoReplay:
    """pseyepy-compatible multi-camera source, backed by any video source."""

    RES_SMALL = (1280, 800)

    def __init__(self, fps: int = 90, resolution=RES_SMALL,
                 gain: int = 10, exposure: int = 100,
                 source: str | None = None,
                 realtime: bool | None = None,
                 loop: bool = True):
        self.fps = int(fps)
        self._loop = loop

        specs = resolve_sources(source)
        self.sources = specs
        self.readers = [open_reader(s, loop=loop) for s in specs]
        self.num_cameras = len(self.readers)
        self.kinds = [r.kind for r in self.readers]

        # helpers.py reads the camera count off this list length and
        # edit_settings() writes into both.
        self.gain = [int(gain)] * self.num_cameras
        self.exposure = [int(exposure)] * self.num_cameras

        # Pacing only makes sense for files we control the timing of; a live
        # stream already arrives at its own rate.
        if realtime is None:
            mode = os.environ.get("MOCAP_SYNC_MODE", "").lower()
            if mode == "pace":
                realtime = True
            elif mode == "native":
                realtime = False
            else:
                realtime = all(k in ("file", "dir") for k in self.kinds)
        self._realtime = bool(realtime)

        self._fps_pace = max(1.0, min(r.fps for r in self.readers))
        if all(k in ("file", "dir") for k in self.kinds):
            rates = [r.fps for r in self.readers]
            counts = [r.frame_count for r in self.readers]
            if max(rates) - min(rates) > 1e-3 or len(set(counts)) != 1:
                self.release()
                raise ValueError("synchronised replay requires equal frame rates and frame counts")
        self._frame = -1
        self._next_t = None
        self._last = [None] * self.num_cameras

    # ------------------------------------------------------------------ utils
    def __len__(self) -> int:
        return self.num_cameras

    def describe(self) -> str:
        return ", ".join(f"CAM{i + 1:02d}={k}:{s}"
                         for i, (k, s) in enumerate(zip(self.kinds, self.sources)))

    # ------------------------------------------------------------------- read
    def read(self):
        """Return (frames, timestamps); blocks to emulate a real capture."""
        if self._realtime:
            now = time.monotonic()
            if self._next_t is None:
                self._next_t = now
            else:
                self._next_t += 1.0 / self._fps_pace
                wait = self._next_t - now
                if wait > 0:
                    time.sleep(wait)

        frames, stamps = [], []
        for i, r in enumerate(self.readers):
            img = r.read()
            if img is None and r.kind in ("file", "dir"):
                if not self._loop:
                    raise StopIteration("synchronised clip exhausted")
                raise OSError(f"unreadable replay frame from camera {i + 1}")
            self._last[i] = img
            frames.append(img)
            # Never label a cached/stalled frame as a fresh observation.
            ts = (r.timestamp() if r.kind in ("file", "dir") else time.monotonic())
            stamps.append(float("nan") if img is None else ts)

        self._frame += 1
        return frames, np.asarray(stamps, dtype=np.float64)

    def release(self) -> None:
        for r in self.readers:
            try:
                r.release()
            except Exception:
                pass

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


# ------------------------------------------------------------------- CLI check
if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else None
    cam = VideoReplay(source=src)
    print("sources:", cam.describe())
    print("pacing:", "fixed clock" if cam._realtime else "native")
    for _ in range(3):
        frames, ts = cam.read()
        print("frames:", [(f.shape if f is not None else None) for f in frames],
              "ts:", np.round(ts, 4).tolist())
    cam.release()
