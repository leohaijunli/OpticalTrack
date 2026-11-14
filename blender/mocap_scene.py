"""
mocap_scene.py — OptiTrack-style motion capture volume, rendered with Blender 4.5.

Consumes the Rust-generated data:
  out/scenario.json      -> per-frame drone pose (MAVLink-derived)
  out/rig/rig_blender.json -> camera poses / OV9281 intrinsics, 3 LED markers

One scene and one timeline: the six mocap cameras view the same animated
bodies at each frame index. Frame f is t = f / FPS (0-based).

Modes
  build                 only build and save out/scene.blend
  still  --shots NAME   render one frame for a quick look
  render --shots a,b,c  render the requested shots

Shots
  hero      wide volume shot, visible light, frustums + calibration rig
  closeup   chase camera on the Crazyflie, visible light, shallow DOF
  cam01..   the actual mocap camera POV
"""

import bpy
import hashlib
import json
import math
import os
import sys
import time
from mathutils import Vector, Quaternion, Euler, Matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "out")
FRAMES = os.path.join(OUT, "frames")

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    key = "--" + name
    if key in argv:
        i = argv.index(key)
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            return argv[i + 1]
    return default


def flag(name):
    return ("--" + name) in argv


SCEN = os.path.join(OUT, os.environ.get("MOCAP_SCENARIO", "scenario.json"))
RIGJ = os.environ.get("MOCAP_RIG") or os.path.join(OUT, "rig", "rig_blender.json")
RIGJ = RIGJ if os.path.isabs(RIGJ) else os.path.join(ROOT, RIGJ)
scenario = json.load(open(SCEN, "r", encoding="utf-8"))
rig = json.load(open(RIGJ, "r", encoding="utf-8"))
BODIES = scenario.get("bodies") or [
    {"body_id": 0, "frame": "crazyflie", "samples": scenario["samples"]}]

# Marker geometry lives in the rig file (one entry per rigid body).  Merge it
# into the scenario bodies by body_id so a body spec is: trajectory + frame +
# marker layout, all data driven.  A scenario without a matching rig body
# falls back to the legacy flat `markers` list.
_rig_bodies = {int(b["body_id"]): b for b in rig.get("bodies", [])}
for _b in BODIES:
    _spec = _rig_bodies.get(int(_b.get("body_id", 0)))
    if _spec is not None:
        _b.setdefault("markers", _spec["markers"])
        _b.setdefault("body_radius_m", _spec.get("body_radius_m", 0.05))
    if "markers" not in _b:
        _b["markers"] = rig.get("markers", [])
SAMPLES = BODIES[0]["samples"]
FPS = int(scenario["meta"]["fps"])
NF = len(SAMPLES)
DUR = NF / FPS

APPEARANCE_PATH = arg("appearance-config", os.path.join(ROOT, "blender", "profiles", "visible_led.json"))
if not os.path.isabs(APPEARANCE_PATH):
    APPEARANCE_PATH = os.path.join(ROOT, APPEARANCE_PATH)
with open(APPEARANCE_PATH, encoding="utf-8") as handle:
    APPEARANCE = json.load(handle)

os.makedirs(FRAMES, exist_ok=True)


# --------------------------------------------------------------------------- utils
def srgb(c):
    """Convert an sRGB-ish authoring value to linear for Blender."""
    return tuple(
        (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4) for x in c
    )


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_mat(name, base=(0.5, 0.5, 0.5), rough=0.5, metal=0.0,
             emit=None, emit_strength=0.0, alpha=1.0, specular=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*srgb(base), 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Specular IOR Level"].default_value = specular
    if emit is not None:
        e = srgb(emit)
        bsdf.inputs["Emission Color"].default_value = (*e, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit_strength
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        for attr, val in (("blend_method", "BLEND"), ("surface_render_method", "BLENDED")):
            try:
                setattr(m, attr, val)
            except Exception:
                pass
        m.use_backface_culling = False
    m.diffuse_color = (*srgb(base), alpha)
    return m


MATS = {"vis": {}, "ir": {}}


def build_materials():
    # ---- visible light look: a working hangar whose matte, low-reflectance
    # surfaces stay well below the detector threshold.  Only the active LED
    # markers are allowed to saturate, exactly like a real capture hall where
    # the floor and walls are deliberately kept dark.
    v = MATS["vis"]
    v["floor"] = make_mat("floor_vis", (0.072, 0.072, 0.076), 0.90)
    v["wall"] = make_mat("wall_vis", (0.066, 0.069, 0.073), 0.65, 0.15)
    v["rib"] = make_mat("rib_vis", (0.048, 0.050, 0.054), 0.60, 0.20)
    v["roof"] = make_mat("roof_vis", (0.030, 0.031, 0.034), 0.85)
    v["steel"] = make_mat("steel_vis", (0.058, 0.061, 0.066), 0.60, 0.35)
    v["door"] = make_mat("door_vis", (0.082, 0.085, 0.090), 0.65, 0.15)
    v["lamp"] = make_mat("lamp_vis", (0.9, 0.9, 0.9), 0.5, 0.0,
                         emit=(1.0, 0.97, 0.90), emit_strength=12.0)
    v["truss"] = make_mat("truss_vis", (0.20, 0.21, 0.22), 0.50, 0.45)
    v["cam_body"] = make_mat("cam_body_vis", (0.055, 0.058, 0.065), 0.34, 0.75)
    v["cam_lens"] = make_mat("cam_lens_vis", (0.02, 0.03, 0.05), 0.06, 0.2)
    v["cam_led"] = make_mat("cam_led_vis", (0.5, 0.03, 0.02), 0.4, 0.0,
                            emit=(1.0, 0.10, 0.03), emit_strength=3.0)
    v["drone_frame"] = make_mat("drone_frame_vis", (0.055, 0.057, 0.063), 0.42)
    v["drone_pcb"] = make_mat("drone_pcb_vis", (0.02, 0.16, 0.09), 0.55)
    v["drone_battery"] = make_mat("drone_batt_vis", (0.10, 0.105, 0.12), 0.55)
    v["drone_motor"] = make_mat("drone_motor_vis", (0.55, 0.56, 0.60), 0.28, 0.95)
    v["drone_prop"] = make_mat("drone_prop_vis", (0.16, 0.17, 0.19), 0.35, 0.1, alpha=0.85)
    v["led_marker"] = make_mat("led_marker_vis", (0.88, 0.88, 0.88), 0.45, 0.0,
                              emit=(1.0, 1.0, 1.0), emit_strength=5.0)
    v["pad"] = make_mat("pad_vis", (0.055, 0.057, 0.060), 0.94)
    v["pad_ring"] = make_mat("pad_ring_vis", (0.10, 0.10, 0.11), 0.94)
    # Coverage cones are drawn as thin emissive edges only; a translucent solid
    # stacks up into opaque blue fog once a dozen cones overlap in shot.
    v["frustum"] = make_mat("frustum_vis", (0.02, 0.1, 0.22), 0.5, 0.0, alpha=0.0)
    v["frustum_edge"] = make_mat("frustum_edge_vis", (0.05, 0.22, 0.45), 0.4, 0.0,
                                 emit=(0.12, 0.38, 0.70), emit_strength=0.30)
    v["wand"] = make_mat("wand_vis", (0.30, 0.31, 0.32), 0.62, 0.10)
    v["floor_mark"] = make_mat("floor_mark_vis", (0.030, 0.031, 0.033), 0.92)
    v["label"] = make_mat("label_vis", (0.9, 0.9, 0.9), 0.5,
                          emit=(1, 1, 1), emit_strength=0.4)
    v["volume_edge"] = make_mat("volume_edge_vis", (0.25, 0.45, 0.70), 0.5, 0.0,
                                emit=(0.22, 0.48, 0.80), emit_strength=0.45)

    # A capture appearance profile controls actual material radiance.  No
    # masks or ground-truth positions are used to erase non-marker pixels.
    for role, material in APPEARANCE["materials"].items():
        v[role] = make_mat(
            role + "_capture_vis", tuple(material["base"]),
            rough=material["roughness"], metal=material["metallic"],
            specular=material["specular"], emit=material.get("emission"),
            emit_strength=material.get("emission_strength", 0.0))

    # ---- 850 nm IR look: everything is near-black, only emitters glow
    r = MATS["ir"]
    dark = (0.012, 0.012, 0.013)
    r["floor"] = make_mat("floor_ir", (0.010, 0.010, 0.011), 0.92)
    r["wall"] = make_mat("wall_ir", (0.006, 0.006, 0.007), 0.95)
    r["rib"] = make_mat("rib_ir", (0.006, 0.006, 0.007), 0.95)
    r["roof"] = make_mat("roof_ir", (0.004, 0.004, 0.005), 0.95)
    r["steel"] = make_mat("steel_ir", (0.012, 0.012, 0.013), 0.9, 0.3)
    r["door"] = make_mat("door_ir", (0.008, 0.008, 0.009), 0.95)
    r["lamp"] = make_mat("lamp_ir", (0.01, 0.01, 0.01), 0.9)
    r["truss"] = make_mat("truss_ir", (0.018, 0.018, 0.019), 0.8, 0.3)
    r["cam_body"] = make_mat("cam_body_ir", (0.02, 0.02, 0.021), 0.8, 0.2)
    r["cam_lens"] = make_mat("cam_lens_ir", (0.004, 0.004, 0.005), 0.3)
    r["cam_led"] = make_mat("cam_led_ir", (0.15, 0.15, 0.16), 0.5,
                            emit=(0.85, 0.87, 0.95), emit_strength=1.4)
    r["drone_frame"] = make_mat("drone_frame_ir", (0.030, 0.030, 0.031), 0.85)
    r["drone_pcb"] = make_mat("drone_pcb_ir", dark, 0.85)
    r["drone_battery"] = make_mat("drone_batt_ir", (0.016, 0.016, 0.017), 0.85)
    r["drone_motor"] = make_mat("drone_motor_ir", (0.035, 0.035, 0.036), 0.7, 0.4)
    r["drone_prop"] = make_mat("drone_prop_ir", (0.014, 0.014, 0.015), 0.8, 0.0, alpha=0.7)
    r["drone_status"] = make_mat("drone_status_ir", dark, 0.85)
    r["led_marker"] = make_mat("led_marker_ir", (1, 1, 1), 0.2, 0.0,
                               emit=(1.0, 1.0, 1.0), emit_strength=90.0)
    r["pad"] = make_mat("pad_ir", (0.012, 0.012, 0.013), 0.9)
    r["pad_ring"] = make_mat("pad_ring_ir", (0.03, 0.03, 0.03), 0.8)
    r["frustum"] = make_mat("frustum_ir", (0, 0, 0), 0.5, 0.0, alpha=0.0)
    r["frustum_edge"] = make_mat("frustum_edge_ir", (0, 0, 0), 0.5, 0.0, alpha=0.0)
    r["wand"] = make_mat("wand_ir", (0.05, 0.05, 0.05), 0.85)
    r["floor_mark"] = make_mat("floor_mark_ir", (0.012, 0.012, 0.013), 0.9)
    r["label"] = make_mat("label_ir", (0.05, 0.05, 0.05), 0.9)
    r["volume_edge"] = make_mat("volume_edge_ir", (0.02, 0.02, 0.02), 0.9)


def set_role(obj, role):
    obj["role"] = role
    m = MATS["vis"].get(role)
    if m is not None:
        obj.data.materials.clear()
        obj.data.materials.append(m)
    return obj


# --------------------------------------------------------------------------- prims
def add_box(name, size, loc=(0, 0, 0), rot=(0, 0, 0), role=None, parent=None,
            bevel=0.0012):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        md = o.modifiers.new("bev", "BEVEL")
        md.width = bevel
        md.segments = 2
    if role:
        set_role(o, role)
    if parent:
        o.parent = parent
    return o


def add_cyl(name, radius, depth, loc=(0, 0, 0), rot=(0, 0, 0), role=None,
            parent=None, verts=24):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth,
                                        location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    if role:
        set_role(o, role)
    if parent:
        o.parent = parent
    return o


def add_sphere(name, radius, loc=(0, 0, 0), role=None, parent=None, seg=16, ring=8):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=ring, radius=radius,
                                         location=loc)
    o = bpy.context.active_object
    o.name = name
    bpy.ops.object.shade_smooth()
    if role:
        set_role(o, role)
    if parent:
        o.parent = parent
    return o


def add_tube(name, p1, p2, radius, role, parent=None, verts=12):
    p1, p2 = Vector(p1), Vector(p2)
    d = p2 - p1
    length = d.length
    if length < 1e-6:
        return None
    mid = (p1 + p2) * 0.5
    o = add_cyl(name, radius, length, loc=mid, role=role, parent=parent, verts=verts)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.normalized().to_track_quat("Z", "Y")
    return o


# --------------------------------------------------------------------------- room
def hangar_size():
    """Room dimensions from the layout's ``hangar`` block (metres)."""
    size = (rig.get("hangar") or {}).get("size") or (22.0, 18.0, 7.5)
    return float(size[0]), float(size[1]), float(size[2])


def build_room():
    """Empty hangar: concrete slab, corrugated steel walls, roof trusses and a
    sliding door. The capture rig stands in the middle of the clear span.

    Every dimension is read from the layout's ``hangar`` / ``volume`` blocks,
    so shortening the flight space (and lowering the ceiling to keep the
    cameras under 5 m) cannot leave a staircase-sized hangar behind.
    """
    W, D, H = hangar_size()
    hangar = rig.get("hangar") or {}
    door = hangar.get("door") or {"x": 0.0, "width": 8.0, "height": 4.5}
    door_x, door_w, door_h = (float(door["x"]), float(door["width"]),
                              float(door["height"]))

    add_box("Floor", (W, D, 0.08), (0, 0, -0.04), role="floor", bevel=0.0)
    add_box("Wall_N", (W, 0.12, H), (0, D / 2, H / 2), role="wall", bevel=0.0)
    add_box("Wall_S", (W, 0.12, H), (0, -D / 2, H / 2), role="wall", bevel=0.0)
    add_box("Wall_E", (0.12, D, H), (W / 2, 0, H / 2), role="wall", bevel=0.0)
    add_box("Wall_W", (0.12, D, H), (-W / 2, 0, H / 2), role="wall", bevel=0.0)
    add_box("Roof", (W, D, 0.10), (0, 0, H), role="roof", bevel=0.0)

    # saw-cut expansion joints so the slab reads as concrete rather than a
    # featureless plane at this distance
    for fx in (-0.4, -0.2, 0.0, 0.2, 0.4):
        x = fx * W
        add_box(f"FloorJointX_{fx:+.1f}", (0.05, D, 0.003), (x, 0, 0.003),
                role="floor_mark", bevel=0.0)
    for fy in (-0.4, -0.2, 0.0, 0.2, 0.4):
        y = fy * D
        add_box(f"FloorJointY_{fy:+.1f}", (W, 0.05, 0.003), (0, y, 0.003),
                role="floor_mark", bevel=0.0)

    # corrugated wall ribs
    rib_step = 1.05
    n_rib_x = max(1, int((W / 2 - 0.2) / rib_step))
    for i in range(-n_rib_x, n_rib_x + 1):
        x = i * rib_step
        for wy, nm in ((D / 2 - 0.07, "N"), (-D / 2 + 0.07, "S")):
            add_box(f"Rib_{nm}{i}", (0.07, 0.07, H), (x, wy, H / 2), role="rib", bevel=0.0)
    n_rib_y = max(1, int((D / 2 - 0.2) / rib_step))
    for j in range(-n_rib_y, n_rib_y + 1):
        y = j * rib_step
        for wx, nm in ((W / 2 - 0.07, "E"), (-W / 2 + 0.07, "W")):
            add_box(f"Rib_{nm}{j}", (0.07, 0.07, H), (wx, y, H / 2), role="rib", bevel=0.0)

    # roof trusses + purlins (everything stays above the camera rail)
    truss_half = D / 2 - 0.9
    n_truss = max(2, int(round(truss_half / 1.2)))
    truss_ys = [-truss_half + (2 * truss_half) * j / n_truss
                for j in range(n_truss + 1)]
    n_web = max(1, int((W / 2) / 2.6))
    for j, y in enumerate(truss_ys):
        add_tube(f"RoofTruss_{j}", (-W / 2, y, H - 0.30), (W / 2, y, H - 0.30), 0.075, "steel")
        add_tube(f"RoofTie_{j}", (-W / 2, y, H - 0.70), (W / 2, y, H - 0.70), 0.035, "steel")
        for k in range(-n_web, n_web + 1):
            x0 = k * 2.6
            add_tube(f"RoofWeb_{j}_{k}", (x0, y, H - 0.70), (x0 + 1.3, y, H - 0.30), 0.024, "steel")
    n_purlin = max(1, int((W / 2) / 1.3))
    for i in range(-n_purlin, n_purlin + 1):
        add_tube(f"Purlin_{i}", (i * 1.3, -D / 2, H - 0.20), (i * 1.3, D / 2, H - 0.20), 0.030, "steel")

    # structural columns along the walls
    column_x = [-W / 2 + 1.0 + i * (W - 2.0) / 4 for i in range(5)]
    for i, x in enumerate(column_x):
        for y in (-D / 2 + 0.18, D / 2 - 0.18):
            add_box(f"Column_{i}_{y > 0}", (0.32, 0.32, H), (x, y, H / 2), role="steel", bevel=0.004)

    # sliding hangar door on the south wall (size from the layout)
    add_box("HangarDoor", (door_w, 0.16, door_h),
            (door_x, -D / 2 + 0.16, door_h / 2), role="door", bevel=0.0)
    for k in range(1, max(1, int(door_h / 0.85)) + 1):
        add_box(f"DoorSeam_{k}", (door_w, 0.04, 0.045),
                (door_x, -D / 2 + 0.25, 0.85 * k), role="rib", bevel=0.0)
    frame_h = door_h + 0.4
    add_box("DoorFrame_L", (0.22, 0.34, frame_h),
            (door_x - door_w / 2 - 0.1, -D / 2 + 0.2, frame_h / 2),
            role="steel", bevel=0.0)
    add_box("DoorFrame_R", (0.22, 0.34, frame_h),
            (door_x + door_w / 2 + 0.1, -D / 2 + 0.2, frame_h / 2),
            role="steel", bevel=0.0)
    add_box("DoorRail", (door_w + 0.6, 0.5, 0.10),
            (door_x, -D / 2 + 0.35, door_h + 0.45), role="steel", bevel=0.0)

    # ceiling luminaires (emissive panels, the matching lights are in build_lights)
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        x, y = sx * W * 0.23, sy * D * 0.28
        add_box(f"LampPanel_{i}", (3.2, 0.9, 0.10), (x, y, H - 0.85), role="lamp", bevel=0.0)
        add_box(f"LampHood_{i}", (3.5, 1.2, 0.16), (x, y, H - 0.72), role="steel", bevel=0.0)

    # calibrated capture volume outline on the slab
    vcx, vcy, vcz = rig["volume"]["center"]
    vsx, vsy, vsz = rig["volume"]["size"]
    for (x, y, sx, sy) in [
        (vcx, vcy - vsy / 2, vsx, 0.025), (vcx, vcy + vsy / 2, vsx, 0.025),
        (vcx - vsx / 2, vcy, 0.025, vsy), (vcx + vsx / 2, vcy, 0.025, vsy),
    ]:
        add_box("Mark", (sx, sy, 0.004), (x, y, 0.004), role="floor_mark", bevel=0.0)

    # calibration reference L-frames, outside the volume but inside the rig
    cfx, cfy = vcx + vsx / 2 + 0.85, vcy + vsy / 2 + 0.65
    for i, (cx, cy, ang) in enumerate([(-cfx, -cfy, 0.0), (cfx, -cfy, math.pi / 2),
                                       (-cfx, cfy, -math.pi / 2)]):
        for j, (ox, oy) in enumerate([(0.30, 0.05), (0.05, 0.30)]):
            add_box(f"CalFrame_{i}_{j}", (0.55 if j == 0 else 0.10, 0.10 if j == 0 else 0.55, 0.02),
                    (cx, cy, 0.012), (0, 0, ang), role="steel")
        for k in range(3):
            add_sphere(f"CalDot_{i}_{k}", 0.017,
                       (cx + math.cos(ang) * (0.10 + 0.14 * k),
                        cy + math.sin(ang) * (0.10 + 0.14 * k), 0.024),
                       role="wand", seg=12, ring=6)

    # take-off pad under the first body's first sample
    pad_x, pad_y = BODIES[0]["samples"][0]["p"][:2]
    add_cyl("Pad", 0.38, 0.012, (pad_x, pad_y, 0.008), role="pad", verts=48)
    add_cyl("PadRing", 0.30, 0.010, (pad_x, pad_y, 0.010), role="pad_ring", verts=48)
    add_cyl("PadInner", 0.10, 0.011, (pad_x, pad_y, 0.011), role="pad", verts=32)

    # calibration wand lying on the slab
    w0 = Vector((vcx + vsx / 2 + 0.45, vcy + vsy / 2 + 0.45, 0.055))
    w1 = w0 + Vector((0.62, 0.18, 0))
    add_tube("WandBar", w0, w1, 0.012, "wand")
    for t, name in ((0.0, "WandDotA"), (1.0, "WandDotB")):
        p = w0.lerp(w1, t)
        add_sphere(name, 0.012, p, role="wand", seg=14, ring=7)
        add_cyl(name + "_stem", 0.006, 0.05, (p.x, p.y, p.z - 0.04), role="wand", verts=8)


def build_truss():
    """Ceiling-mounted camera frame.

    Everything structural lives *above and outboard* of the six mounts:

      * a stiffening rail ring at radius R + 0.55, 0.45 m above the mounts,
      * a short arm from the ring to each camera's back plate,
      * a drop rod from the roof purlins to that arm.

    No floor stands and no columns. Because every mocap camera looks down at
    46 deg, the top of its frame is still 23 deg below the horizon, so nothing
    above the mount height can ever enter a delivered POV shot.
    """
    pts = [Vector(c["pos"]) for c in rig["cameras"]]
    n = len(pts)
    R = sum((p.xy.length for p in pts)) / len(pts)
    Z = sum(p.z for p in pts) / len(pts)
    RAIL_R = R + 0.55
    RAIL_Z = Z + 0.45
    # The roof plate hangs just under the slab; the room height comes from the
    # layout so the drop rods shorten with the ceiling instead of overshooting.
    ROOF_Z = hangar_size()[2] - 0.08

    ring = []
    for i, p in enumerate(pts):
        out = Vector((p.x, p.y, 0.0)).normalized()
        ring.append(Vector((p.x + out.x * 0.55, p.y + out.y * 0.55, RAIL_Z)))

    # rail ring
    for i in range(n):
        add_tube(f"TrussRail_{i}", ring[i], ring[(i + 1) % n], 0.030, "truss")

    for i, p in enumerate(pts):
        out = Vector((p.x, p.y, 0.0)).normalized()
        r = ring[i]

        # roof plate + drop rod
        add_box(f"CeilPlate_{i}", (0.30, 0.30, 0.024),
                (r.x, r.y, ROOF_Z - 0.012), role="steel")
        add_tube(f"CeilRod_{i}", (r.x, r.y, ROOF_Z - 0.024),
                 (r.x, r.y, RAIL_Z + 0.03), 0.020, "steel")
        for dx, dy in ((-0.10, -0.10), (0.10, -0.10), (-0.10, 0.10), (0.10, 0.10)):
            add_tube(f"CeilBolt_{i}_{dx:+.0f}{dy:+.0f}",
                     (r.x + dx, r.y + dy, ROOF_Z), (r.x + dx, r.y + dy, ROOF_Z - 0.05),
                     0.008, "steel")

        # rail -> camera back plate
        arm = p + Vector((out.x, out.y, 0.0)) * 0.075 + Vector((0, 0, 0.105))
        add_tube(f"MountArm_{i}", (r.x, r.y, RAIL_Z), arm, 0.022, "steel")
        add_box(f"MountClamp_{i}", (0.075, 0.075, 0.035),
                (arm.x, arm.y, arm.z + 0.02), role="steel", bevel=0.003)

        # short diagonal brace from the roof plate to the ring vertex
        add_tube(f"CeilBrace_{i}", (r.x + out.x * 0.55, r.y + out.y * 0.55, ROOF_Z - 0.05),
                 (r.x, r.y, RAIL_Z), 0.016, "steel")


# --------------------------------------------------------------------------- cameras
CAM_ROOTS = {}


def build_cameras():
    for cam in rig["cameras"]:
        name = cam["name"]
        root = bpy.data.objects.new(f"RIG_{name}", None)
        bpy.context.collection.objects.link(root)
        root.location = Vector(cam["pos"])
        q = cam["quat"]
        root.rotation_mode = "QUATERNION"
        root.rotation_quaternion = Quaternion((q[0], q[1], q[2], q[3]))
        CAM_ROOTS[name] = root

        # fixture geometry, local frame: lens points down -Z.
        # Arducam OV9281 module on an M12 lens holder: 40 x 40 mm board, no IR
        # illuminator ring (this is a visible-light mono camera).
        add_box(f"{name}_Body", (0.040, 0.040, 0.014), (0, 0, 0.0),
                role="cam_body", parent=root, bevel=0.002)
        add_box(f"{name}_Board", (0.044, 0.044, 0.003), (0, 0, 0.009),
                role="cam_body", parent=root, bevel=0.001)
        add_cyl(f"{name}_Barrel", 0.0145, 0.020, (0, 0, -0.017), role="cam_body",
                parent=root, verts=28)
        add_cyl(f"{name}_Glass", 0.0115, 0.003, (0, 0, -0.028), role="cam_lens",
                parent=root, verts=28)
        add_box(f"{name}_BackPlate", (0.070, 0.070, 0.010), (0, 0, 0.016),
                role="steel", parent=root, bevel=0.002)

        # Blender camera for the actual sensor view
        cd = bpy.data.cameras.new(f"CAMDATA_{name}")
        cd.lens = cam["lens_mm"]
        cd.sensor_width = cam["sensor_width_mm"]
        cd.sensor_fit = "HORIZONTAL"
        cd.clip_start = 0.05
        cd.clip_end = 120.0
        co = bpy.data.objects.new(name, cd)
        bpy.context.collection.objects.link(co)
        co.parent = root
        co.matrix_parent_inverse = Matrix.Identity(4)
        co.location = (0, 0, 0)
        co["role"] = "sensor"
        # ceiling hanger is built in build_truss(): nothing is attached below
        # or beside the lens, so no stand can occlude the view.


FRUSTUMS = []
FIXTURES = []


def collect_fixtures():
    FIXTURES.clear()
    for o in bpy.data.objects:
        if o.name.startswith("CAM") and "_" in o.name and o.type == "MESH":
            FIXTURES.append(o)
    # Hangar structure and rig hardware are decorative here: let them receive
    # light but not throw hard shadows across the capture volume.
    decor = ("CAM", "Truss", "Column", "Base_", "Rib_", "Roof", "Purlin", "Door",
             "HangarDoor", "Lamp", "Wand", "CalFrame", "CalDot", "Mark", "Pad",
             "VolumeEdge", "Frustum")
    for o in bpy.data.objects:
        if o.type == "MESH" and o.name.startswith(decor):
            o.visible_shadow = False
    if flag("noshadow"):
        for o in bpy.data.objects:
            if o.type == "MESH":
                o.visible_shadow = False


def build_frustums():
    """Coverage cones drawn as thin emissive edges (no translucent solid: a
    dozen overlapping faces turn the whole shot into opaque blue fog)."""
    for cam in rig["cameras"]:
        pos = Vector(cam["pos"])
        q = cam["quat"]
        rot = Quaternion((q[0], q[1], q[2], q[3]))
        M = rot.to_matrix()
        right = M @ Vector((1, 0, 0))
        up = M @ Vector((0, 1, 0))
        fwd = M @ Vector((0, 0, -1))
        dist = 4.6
        hw = math.tan(math.radians(cam["hfov_deg"]) * 0.5) * dist
        hh = hw * float(rig["resolution"][1]) / float(rig["resolution"][0])
        base = pos + fwd * dist
        corners = [
            base - right * hw - up * hh,
            base + right * hw - up * hh,
            base + right * hw + up * hh,
            base - right * hw + up * hh,
        ]
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            e = add_tube(f"FrustumEdge_{cam['name']}_{i}", a, b, 0.0016, "frustum_edge")
            e.visible_shadow = False
            FRUSTUMS.append(e)
        # the four side rails from the lens out to the far corners
        for i in range(4):
            e = add_tube(f"FrustumRail_{cam['name']}_{i}", pos + fwd * 0.12,
                         corners[i], 0.0010, "frustum_edge")
            e.visible_shadow = False
            FRUSTUMS.append(e)


def build_volume_box():
    """Wireframe of the calibrated capture volume, from the rig JSON."""
    cx, cy, cz = rig["volume"]["center"]
    sx, sy, sz = rig["volume"]["size"]
    lo = Vector((cx - sx / 2, cy - sy / 2, cz - sz / 2))
    hi = Vector((cx + sx / 2, cy + sy / 2, cz + sz / 2))
    corners = [
        Vector((lo.x, lo.y, lo.z)), Vector((hi.x, lo.y, lo.z)),
        Vector((hi.x, hi.y, lo.z)), Vector((lo.x, hi.y, lo.z)),
        Vector((lo.x, lo.y, hi.z)), Vector((hi.x, lo.y, hi.z)),
        Vector((hi.x, hi.y, hi.z)), Vector((lo.x, hi.y, hi.z)),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, (a, b) in enumerate(edges):
        e = add_tube(f"VolumeEdge_{i}", corners[a], corners[b], 0.0045, "volume_edge")
        e.visible_shadow = False
        FRUSTUMS.append(e)



# --------------------------------------------------------------------------- drone
DRONE = None
PROPS = []
DRONES = []   # (root_empty, samples, props) per body



def build_tarot650(spec, prefix="T650"):
    """Tarot Ironman 650: 650 mm wheelbase X-frame, motors at +-0.230 m,
    15 inch props, tall landing skids.  spec["frame"] picks the small rig
    details (mast vs plate markers) but the geometry is shared."""
    global DRONE
    DRONE = bpy.data.objects.new(prefix + "_BODY", None)
    bpy.context.collection.objects.link(DRONE)

    arm_r = 0.230            # motor radius from the centre (650 mm diagonal)
    arm_z = 0.020            # arm plane above the origin (CG)

    # centre plates (lower + upper), Tarot twin-plate style
    add_cyl(prefix + "_PlateLow", 0.088, 0.0025, (0, 0, 0.000), role="drone_frame",
            parent=DRONE, verts=32)
    add_cyl(prefix + "_PlateUp", 0.088, 0.0025, (0, 0, 0.062), role="drone_frame",
            parent=DRONE, verts=32)
    add_cyl(prefix + "_SpacerF", 0.004, 0.062, (0.070, 0.0, 0.031), role="drone_frame",
            parent=DRONE, verts=10)
    add_cyl(prefix + "_SpacerB", 0.004, 0.062, (-0.070, 0.0, 0.031), role="drone_frame",
            parent=DRONE, verts=10)
    # battery on the lower plate
    add_box(prefix + "_Battery", (0.155, 0.065, 0.048), (0.0, 0.0, 0.028),
            role="drone_battery", parent=DRONE, bevel=0.004)

    # X arms + motors + props
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, -1), (-1, 1)]):
        mx, my = sx * arm_r, sy * arm_r
        ang = math.atan2(my, mx)
        cx, cy = mx * 0.42, my * 0.42
        add_cyl(prefix + "_Arm%d" % i, 0.008, 0.30, (cx, cy, arm_z),
                (0, math.pi / 2, ang), role="drone_frame", parent=DRONE, verts=10)
        add_cyl(prefix + "_Motor%d" % i, 0.021, 0.032, (mx, my, arm_z + 0.026),
                role="drone_motor", parent=DRONE, verts=16)

        prop = bpy.data.objects.new(prefix + "_Prop%d" % i, None)
        bpy.context.collection.objects.link(prop)
        prop.parent = DRONE
        prop.location = (mx, my, arm_z + 0.060)
        PROPS.append((prop, 1 if i % 2 == 0 else -1))
        # 15 inch two-blade prop
        for b in range(2):
            add_box(prefix + "_Blade%d_%d" % (i, b), (0.186, 0.021, 0.0028),
                    (0, 0, 0), (0, 0, b * math.pi), role="drone_prop",
                    parent=prop, bevel=0.0008)
        add_cyl(prefix + "_Hub%d" % i, 0.009, 0.012, (0, 0, 0.002), role="drone_prop",
                parent=prop, verts=12)

    # landing gear: 4 legs + 2 skids
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        add_cyl(prefix + "_Leg%d" % i, 0.006, 0.235,
                (sx * 0.090, sy * 0.090, -0.098), role="drone_frame",
                parent=DRONE, verts=8)
    for i, sy in enumerate([1, -1]):
        add_cyl(prefix + "_Skid%d" % i, 0.006, 0.36,
                (0.0, sy * 0.090, -0.212), (0, math.pi / 2, 0), role="drone_frame",
                parent=DRONE, verts=8)

    # Active markers from the body spec.  Every marker sits on a mast that is
    # bolted to the airframe at m["mount"]: a retro-reflective dot floating in
    # mid-air would be a modelling error, and it also tells the viewer how the
    # rig was actually built.
    for m in spec["markers"]:
        x, y, z = m["local"]
        ax, ay, az = m.get("mount", (x, y, 0.065))
        r = m["radius_m"]
        # base bracket: straddles the plate surface so the mast foot is
        # visibly through-bolted, never hovering above it
        add_cyl(prefix + "_MountPad%d" % m["id"], 0.0115, 0.0060,
                (ax, ay, az + 0.0010), role="drone_frame", parent=DRONE, verts=14)
        # carbon mast from the bracket up to the marker
        p0 = Vector((ax, ay, az))
        p1 = Vector((x, y, z - r * 0.8))
        add_tube(prefix + "_Mast%d" % m["id"], p0, p1, 0.0040, "drone_frame",
                 parent=DRONE, verts=8)
        # clamp collar right under the sphere, aligned with the mast
        d = (Vector((x, y, z)) - p0)
        if d.length > 1e-6:
            add_cyl(prefix + "_Collar%d" % m["id"], 0.0058, 0.0045,
                    (x, y, z - r * 0.85),
                    d.normalized().to_track_quat("Z", "Y").to_euler(),
                    role="drone_frame", parent=DRONE, verts=12)
        add_sphere(prefix + "_Marker%d" % m["id"], r, (x, y, z),
                   role="led_marker", parent=DRONE, seg=12, ring=6)

    # Low-output board indicators are distinct from the tracking LEDs.
    for i, (x, y) in enumerate([(0.060, 0.040), (-0.060, -0.040)]):
        add_sphere(prefix + "_Status%d" % i, 0.0022, (x, y, 0.0655), role="drone_status",
                   parent=DRONE, seg=8, ring=4)



def build_crazyflie(spec, prefix="CF"):
    """Crazyflie 2.1-class airframe (legacy single-body dataset)."""
    global DRONE
    DRONE = bpy.data.objects.new(prefix + "_BODY", None)
    bpy.context.collection.objects.link(DRONE)

    # central stack
    add_box(prefix + "_PCB_Top", (0.054, 0.054, 0.0035), (0, 0, 0.0045),
            role="drone_pcb", parent=DRONE, bevel=0.0008)
    add_box(prefix + "_PCB_Bot", (0.050, 0.050, 0.0030), (0, 0, -0.0015),
            role="drone_pcb", parent=DRONE, bevel=0.0008)
    add_box(prefix + "_Battery", (0.036, 0.023, 0.014), (0.002, 0, 0.0135),
            role="drone_battery", parent=DRONE, bevel=0.0015)
    add_box(prefix + "_Header", (0.020, 0.012, 0.004), (-0.010, 0.016, 0.008),
            role="drone_frame", parent=DRONE, bevel=0.0006)

    motor_xy = 0.0325
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        mx, my = sx * motor_xy, sy * motor_xy
        ang = math.atan2(my, mx)
        cx, cy = mx * 0.5, my * 0.5
        add_box(prefix + "_Arm%d" % i, (0.048, 0.0055, 0.0022), (cx, cy, 0.0025),
                (0, 0, ang), role="drone_frame", parent=DRONE, bevel=0.0006)
        add_cyl(prefix + "_Motor%d" % i, 0.0058, 0.011, (mx, my, 0.0045),
                role="drone_motor", parent=DRONE, verts=16)
        # 45 mm two-blade prop on its own empty so it can spin
        prop = bpy.data.objects.new(prefix + "_Prop%d" % i, None)
        bpy.context.collection.objects.link(prop)
        prop.parent = DRONE
        prop.location = (mx, my, 0.0115)
        PROPS.append((prop, 1 if i % 2 == 0 else -1))
        for b in range(2):
            add_box(prefix + "_Blade%d_%d" % (i, b), (0.042, 0.008, 0.0012),
                    (0, 0, 0), (0, 0, b * math.pi), role="drone_prop",
                    parent=prop, bevel=0.0004)
        add_cyl(prefix + "_Hub%d" % i, 0.0035, 0.0035, (0, 0, 0.0012),
                role="drone_prop", parent=prop, verts=12)
        # landing foot
        add_cyl(prefix + "_Foot%d" % i, 0.0022, 0.008, (mx, my, -0.0045),
                role="drone_frame", parent=DRONE, verts=8)

    # active markers, each on a short standoff off the PCB
    for m in spec["markers"]:
        x, y, z = m["local"]
        ax, ay, az = m.get("mount", (x, y, 0.0065))
        r = m["radius_m"]
        add_cyl(prefix + "_MountPad%d" % m["id"], 0.0038, 0.0022,
                (ax, ay, az + 0.0011), role="drone_frame", parent=DRONE, verts=10)
        add_tube(prefix + "_Mast%d" % m["id"],
                 (ax, ay, az + 0.0022), (x, y, z - r * 0.8),
                 0.0016, "drone_frame", parent=DRONE, verts=6)
        add_sphere(prefix + "_Marker%d" % m["id"], r, (x, y, z),
                   role="led_marker", parent=DRONE, seg=12, ring=6)


def build_drone():
    """Build every body of the scenario (multi-target capable)."""
    global DRONE
    for b in BODIES:
        is_cf = b.get("frame", "crazyflie") == "crazyflie"
        before = len(PROPS)
        frame = b.get("frame", "crazyflie")
        if frame == "crazyflie":
            builder = build_crazyflie
        else:
            builder = build_tarot650
        builder(b, prefix=("CF" if is_cf else "T650%d" % b["body_id"]))
        root = DRONE
        props = PROPS[before:]
        DRONES.append((root, b["samples"], props))
    DRONE = DRONES[0][0]


def animate_body(root, samples, props):
    root.rotation_mode = "QUATERNION"
    for i, s in enumerate(samples):
        root.location = Vector(s["p"])
        q = s["q"]
        root.rotation_quaternion = Quaternion((q[0], q[1], q[2], q[3]))
        root.keyframe_insert("location", frame=i)
        root.keyframe_insert("rotation_quaternion", frame=i)
    for fc in root.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    for prop, direction in props:
        prop.rotation_mode = "XYZ"
        prop.rotation_euler = (0, 0, 0)
        prop.keyframe_insert("rotation_euler", index=2, frame=0)
        prop.rotation_euler = (0, 0, direction * math.pi * 2 * NF * 0.55)
        prop.keyframe_insert("rotation_euler", index=2, frame=NF - 1)
        for fc in prop.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def animate():
    for root, samples, props in DRONES:
        animate_body(root, samples, props)


# --------------------------------------------------------------------------- lights
LIGHTS = []


def add_light(name, kind, loc, energy, size=3.0, color=(1, 1, 1), rot=(0, 0, 0),
              aim=None):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color
    if kind == "AREA":
        ld.size = size
    lo = bpy.data.objects.new(name, ld)
    bpy.context.collection.objects.link(lo)
    lo.location = loc
    if aim is not None:
        d = Vector(aim) - Vector(loc)
        lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    else:
        lo.rotation_euler = rot
    lo["energy_vis"] = energy
    LIGHTS.append(lo)
    return lo


def build_lights():
    # four ceiling luminaires, matching the emissive panels in build_room.
    # They sit above the camera mounts, so they light the volume without ever
    # appearing in a downward-looking POV frame.
    W, D, H = hangar_size()
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        add_light(f"Ceiling_{i}", "AREA", (sx * W * 0.23, sy * D * 0.28, H - 0.88),
                  700, size=3.0, color=(1.0, 0.97, 0.92))
    # soft fill through the open door side
    door = (rig.get("hangar") or {}).get("door") or {"x": 0.0}
    add_light("DoorFill", "AREA", (float(door["x"]), -D / 2 + 0.4, min(4.2, H - 1.5)),
              250, size=7.0, color=(0.82, 0.88, 1.0), aim=(0, 0, 1.3))
    # low modelling lights so the airframe is not a silhouette
    add_light("StudioA", "AREA", (3.6, -3.2, 3.0), 130, size=4.0,
              color=(1.0, 0.98, 0.95), aim=(0, 0, 1.2))
    add_light("StudioB", "AREA", (-3.4, 2.6, 3.4), 95, size=4.0,
              color=(0.86, 0.92, 1.0), aim=(0, 0, 1.2))


def setup_world(band):
    w = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    if band == "ir":
        bg.inputs[0].default_value = (0.004, 0.004, 0.005, 1)
        bg.inputs[1].default_value = 0.60
    else:
        bg.inputs[0].default_value = (0.012, 0.013, 0.017, 1)
        bg.inputs[1].default_value = 1.0


# --------------------------------------------------------------------------- hero cams
def build_story_cameras():
    # The hero shot frames the whole calibrated volume instead of chasing one
    # airframe: with three bodies a fixed look-at keeps all of them in view and
    # the framing stable for the length of the clip.
    focus = bpy.data.objects.new("FOCUS", None)
    bpy.context.collection.objects.link(focus)
    focus.location = Vector(rig["volume"]["center"])

    hero_d = bpy.data.cameras.new("HERO")
    hero_d.lens = 28.0
    hero_d.sensor_width = 36.0
    hero = bpy.data.objects.new("HERO", hero_d)
    bpy.context.collection.objects.link(hero)
    hero.location = (5.55, -5.75, 3.35)
    t = hero.constraints.new("TRACK_TO")
    t.target = focus
    t.track_axis = "TRACK_NEGATIVE_Z"
    t.up_axis = "UP_Y"
    hero.keyframe_insert("location", frame=0)
    hero.location = (4.65, -4.95, 2.65)
    hero.keyframe_insert("location", frame=int((NF - 1) * 0.52))
    hero.location = (5.35, -5.55, 3.20)
    hero.keyframe_insert("location", frame=NF - 1)
    for fc in hero.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"

    # chase camera for the close-up
    cd = bpy.data.cameras.new("CLOSEUP")
    cd.lens = 62.0
    cd.sensor_width = 36.0
    cd.dof.use_dof = True
    cd.dof.focus_object = DRONE
    cd.dof.aperture_fstop = 2.8
    close = bpy.data.objects.new("CLOSEUP", cd)
    bpy.context.collection.objects.link(close)
    cc = close.constraints.new("COPY_LOCATION")
    cc.target = DRONE
    cc.use_offset = True
    close.location = (1.25, -1.05, 0.55)
    t2 = close.constraints.new("TRACK_TO")
    t2.target = focus
    t2.track_axis = "TRACK_NEGATIVE_Z"
    t2.up_axis = "UP_Y"



# --------------------------------------------------------------------------- render
def apply_band(band):
    for obj in bpy.data.objects:
        role = obj.get("role")
        if role and obj.type == "MESH":
            m = MATS[band].get(role)
            if m is not None:
                obj.data.materials.clear()
                obj.data.materials.append(m)
    scale = 1.0 if band == "vis" else 0.30
    for lo in LIGHTS:
        lo.data.energy = lo["energy_vis"] * scale
    setup_world(band)
    for o in FRUSTUMS:
        o.hide_render = band == "ir"


def setup_compositor(band, sensor_view=False):
    sc = bpy.context.scene
    sc.use_nodes = True
    nt = sc.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    if sensor_view and band == "vis" and not APPEARANCE["sensor"]["cosmetic_glow"]:
        nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
        return
    glare = nt.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    glare.quality = "HIGH"
    if band == "ir":
        glare.threshold = 0.05
        glare.size = 10
        glare.mix = 0.30
        bw = nt.nodes.new("CompositorNodeRGBToBW")
        nt.links.new(rl.outputs["Image"], glare.inputs["Image"])
        nt.links.new(glare.outputs["Image"], bw.inputs["Image"])
        nt.links.new(bw.outputs["Val"], comp.inputs["Image"])
    else:
        glare.threshold = 0.30
        glare.size = 9
        glare.mix = 0.22
        nt.links.new(rl.outputs["Image"], glare.inputs["Image"])
        nt.links.new(glare.outputs["Image"], comp.inputs["Image"])


def setup_render(band, samples=16, motion_blur=False, sensor_view=False):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.render.fps = FPS
    sc.frame_start = 0
    sc.frame_end = NF - 1
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 12
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.exposure = APPEARANCE["sensor"]["exposure_stops"] if sensor_view else 0.0
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    ee = sc.eevee
    ee.taa_render_samples = samples
    try:
        ee.use_raytracing = False
    except Exception:
        pass
    try:
        ee.use_shadows = True
    except Exception:
        pass
    sc.render.use_motion_blur = motion_blur
    sc.render.motion_blur_shutter = 0.25
    setup_compositor(band, sensor_view)


def shot_obj(shot):
    return {"hero": "HERO", "closeup": "CLOSEUP"}.get(shot, shot)


def render_shot(shot, band, w, h, outdir, samples=16, motion_blur=False, frames=None):
    sc = bpy.context.scene
    apply_band(band)
    setup_render(band, samples, motion_blur, sensor_view=shot.upper().startswith("CAM"))
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    sc.render.resolution_percentage = 100

    if shot == "hero":
        sc.camera = bpy.data.objects["HERO"]
        for o in FRUSTUMS:
            o.hide_render = band == "ir"
    elif shot == "closeup":
        sc.camera = bpy.data.objects["CLOSEUP"]
        for o in FRUSTUMS:
            o.hide_render = True
    else:
        sc.camera = bpy.data.objects[shot]
        for o in FRUSTUMS:
            o.hide_render = True
        for o in FIXTURES:
            o.hide_render = o.name.startswith(shot + "_")

    os.makedirs(outdir, exist_ok=True)
    # The delivered POV cameras are OV9281 mono sensors: write true grayscale
    # PNGs for them, RGB only for the documentation shots (hero / closeup).
    sc.render.image_settings.color_mode = "BW" if shot.upper().startswith("CAM") else "RGB"
    sc.render.filepath = os.path.join(outdir, "f####")
    if frames:
        a, b = frames.split("-")
        sc.frame_start, sc.frame_end = int(a), int(b)
    else:
        sc.frame_start, sc.frame_end = 0, NF - 1

    def digest(path):
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    manifest = {
        "shot": shot, "band": band, "resolution": [w, h], "fps": FPS,
        "frames": [sc.frame_start, sc.frame_end], "appearance": APPEARANCE,
        "source_sha256": {"scenario": digest(SCEN), "rig": digest(RIGJ),
                          "scene_script": digest(__file__), "appearance": digest(APPEARANCE_PATH)},
        "status": "rendering"}
    manifest_path = os.path.join(outdir, "render_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    t0 = time.time()
    selected = arg("frame-list")
    if selected:
        selected = [int(value) for value in selected.split(",")]
        if not selected or any(frame < 0 or frame >= NF for frame in selected):
            raise ValueError("frame-list entries must be within the scenario timeline")
        for frame in selected:
            sc.frame_set(frame)
            sc.render.filepath = os.path.join(outdir, f"f{frame:04d}.png")
            bpy.ops.render.render(write_still=True)
        manifest["selected_frames"] = selected
    else:
        bpy.ops.render.render(animation=True)
    dt = time.time() - t0
    n = sc.frame_end - sc.frame_start + 1
    if selected:
        n = len(selected)
    manifest["status"] = "complete"
    manifest["rendered_count"] = n
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[shot] {shot:<8} band={band:<3} {w}x{h} frames={n} "
          f"time={dt:.1f}s ({dt / max(n, 1):.2f}s/frame) -> {outdir}")


def main():
    mode = arg("mode", "build")
    if mode == "build":
        clear()
        build_materials()
        build_room()
        build_truss()
        build_cameras()
        build_drone()
        animate()
        build_lights()
        build_story_cameras()
        build_frustums()
        build_volume_box()
        collect_fixtures()
        setup_world("vis")
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = NF - 1
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "scene.blend"))
        print(f"[build] scene saved, {NF} frames @ {FPS} fps, {DUR:.2f}s")
        return

    # render modes rebuild in-process (the scene build is only a few seconds)
    clear()
    build_materials()
    build_room()
    build_truss()
    build_cameras()
    build_drone()
    animate()
    build_lights()
    build_story_cameras()
    build_frustums()
    build_volume_box()
    collect_fixtures()

    shots = (arg("shots", "hero") or "").split(",")
    band = arg("band", "vis")
    w = int(arg("width", rig["resolution"][0]))
    h = int(arg("height", rig["resolution"][1]))
    samples = int(arg("samples", 16))
    frames = arg("frames", None)
    tag = arg("tag", band)

    if mode == "still":
        for shot in shots:
            if shot_obj(shot) not in bpy.data.objects:
                print("unknown shot", shot)
                continue
            render_shot(shot, band, w, h,
                        os.path.join(FRAMES, f"still_{shot}_{tag}"), samples,
                        motion_blur=flag("mblur"), frames=frames or "1-1")
        return

    for shot in shots:
        if shot_obj(shot) not in bpy.data.objects:
            print("unknown shot", shot)
            continue
        render_shot(shot, band, w, h, os.path.join(FRAMES, f"{shot}_{tag}"),
                    samples, motion_blur=flag("mblur"), frames=frames)


main()









