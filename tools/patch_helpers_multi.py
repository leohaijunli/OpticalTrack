"""Wire multi-target localisation into helpers.py.

Two edits:
  * set_num_objects(n) also loads the body templates when n > 1
  * _camera_read() calls locate_bodies() instead of locate_objects() for n > 1
"""
import io
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
H = os.path.join(ROOT, "adapters", "low_cost_mocap", "low-cost-mocap",
                 "computer_code", "api", "helpers.py")
s = io.open(H, encoding="utf-8").read()

# ---------------------------------------------------------------- 1. import
old = """from video_replay import VideoReplay as Camera
from Singleton import Singleton"""
new = """from video_replay import VideoReplay as Camera
from Singleton import Singleton

# Multi-target localisation: one rigid body per drone, told apart by the
# pairwise marker distances in its body spec (see adapters/low_cost_mocap/
# multi_body.py).  Single-body runs keep using the upstream locate_objects().
try:
    from multi_body import (load_body_templates, locate_bodies, BodyTracker,
                            default_rig_path)
    _MULTI_OK = True
except Exception as _e:            # pragma: no cover - optional accelerator
    _MULTI_OK = False
    _MULTI_ERR = _e"""
assert old in s, "import block not found"
s = s.replace(old, new, 1)

# ------------------------------------------------------- 2. instance state
old = """        self.num_objects = None

        self.kalman_filter = None"""
new = """        self.num_objects = None

        # multi-body state
        self.body_templates = None
        self.body_tracker = None
        self.frame_index = 0

        self.kalman_filter = None"""
assert old in s, "state block not found"
s = s.replace(old, new, 1)

# ------------------------------------------------- 3. set_num_objects loads rig
old = """    def set_num_objects(self, num_objects):
        self.num_objects = num_objects
        self.drone_armed = [False for i in range(0, self.num_objects)]"""
new = """    def set_num_objects(self, num_objects):
        self.num_objects = num_objects
        self.drone_armed = [False for i in range(0, self.num_objects)]

        # n > 1 means the scene holds several rigid bodies: load their marker
        # layouts once, here, so the hot loop stays allocation free.
        self.body_templates = None
        self.body_tracker = None
        if num_objects > 1 and _MULTI_OK:
            path = (os.environ.get("MOCAP_BODY_SPECS")
                    or os.environ.get("MOCAP_RIG_JSON")
                    or default_rig_path())
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", path)
            if os.path.exists(path):
                self.body_templates = load_body_templates(path)
                if len(self.body_templates) < num_objects:
                    print(f"[multi] {path} defines {len(self.body_templates)} "
                          f"bodies, num_objects={num_objects}")
                self.body_tracker = BodyTracker(self.body_templates)
                print("[multi] bodies:",
                      ", ".join(f"{t['body_id']}:{t['name']}"
                                f"({t['frame']})" for t in self.body_templates))
            else:
                print(f"[multi] body spec not found: {path}")"""
assert old in s, "set_num_objects not found"
s = s.replace(old, new, 1)

# ------------------------------------------------------- 4. hot loop branch
old = """                    if self.is_locating_objects:
                        objects = locate_objects(object_points, errors)"""
new = """                    if self.is_locating_objects:
                        if self.body_templates and len(self.body_templates) > 1:
                            objects = locate_bodies(
                                object_points, errors, self.body_templates,
                                self.body_tracker,
                                tol=float(os.environ.get("MOCAP_BODY_TOL", 0.030)),
                                fps=float(os.environ.get("MOCAP_RATE", 30.0)),
                                frame=self.frame_index)
                        else:
                            objects = locate_objects(object_points, errors)"""
assert old in s, "locate_objects call not found"
s = s.replace(old, new, 1)

# ------------------------------------------------------- 5. frame counter
old = """    def _camera_read(self):
        frames, _ = self.cameras.read()"""
new = """    def _camera_read(self):
        frames, _ = self.cameras.read()
        self.frame_index += 1"""
assert old in s, "camera_read not found"
s = s.replace(old, new, 1)

io.open(H, "w", encoding="utf-8", newline="\n").write(s)
print("helpers.py: multi-target localisation wired in")
