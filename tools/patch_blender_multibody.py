"""Multi-body support for blender/mocap_scene.py.

* model registry: build_crazyflie / build_tarot650 dispatched by the body spec
* scenario "bodies" support (N drones, each animated from its own samples)
* legacy single-body scenario still works unchanged
"""
import io, re

p = r"C:\Users\leeli\Downloads\opticaltrack\blender\mocap_scene.py"
s = io.open(p, encoding="utf-8").read()

# ---- 1. globals: bodies list ------------------------------------------------
old = '''SCEN = os.path.join(OUT, "scenario.json")'''
new = '''SCEN = os.path.join(OUT, os.environ.get("MOCAP_SCENARIO", "scenario.json"))'''
assert old in s
s = s.replace(old, new, 1)

old = '''SAMPLES = scenario["samples"]'''
new = '''BODIES = scenario.get("bodies") or [
    {"body_id": 0, "frame": "crazyflie", "samples": scenario["samples"]}]
SAMPLES = BODIES[0]["samples"]'''
assert old in s
s = s.replace(old, new, 1)

# ---- 2. rename the current builder and add the registry + Tarot builder -----
old = "def build_drone():\n    global DRONE"
new = "def build_crazyflie(spec, prefix=\"CF\"):\n    global DRONE"
assert old in s
s = s.replace(old, new, 1)

# markers of this body come from the spec, not the global rig
old = '''    # three active IR markers, positions come from the Rust rig definition
    for m in rig["markers"]:'''
new = '''    # active markers, positions come from the Rust body spec
    for m in spec["markers"]:'''
assert old in s
s = s.replace(old, new, 1)

# ---- 3. registry + tarot650 builders, inserted before build_drone -----------
TAROT = '''
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

    # active markers from the Rust body spec
    for m in spec["markers"]:
        x, y, z = m["local"]
        add_cyl(prefix + "_MarkerBase%d" % m["id"], 0.006, 0.003, (x, y, z - 0.0045),
                role="drone_frame", parent=DRONE, verts=12)
        add_sphere(prefix + "_Marker%d" % m["id"], m["radius_m"], (x, y, z),
                   role="led_marker", parent=DRONE, seg=12, ring=6)

    # a couple of status LEDs on the top plate
    for i, (x, y) in enumerate([(0.060, 0.040), (-0.060, -0.040)]):
        add_sphere(prefix + "_Status%d" % i, 0.0022, (x, y, 0.0655), role="led_marker",
                   parent=DRONE, seg=8, ring=4)


BUILDERS = {
    "crazyflie": build_crazyflie,
    "tarot650_x": build_tarot650,
    "tarot650_fwd": build_tarot650,
    "tarot650_rev": build_tarot650,
}


def build_drone():
    """Build every body of the scenario (multi-target capable)."""
    for b in BODIES:
        builder = BUILDERS.get(b.get("frame", "crazyflie"), build_crazyflie)
        builder(b, prefix=("CF" if b.get("frame", "crazyflie") == "crazyflie"
                           else "T650%d" % b["body_id"]))


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
    if len(BODIES) == 1 and DRONES[0][0].name == "CF_BODY":
        animate_body(DRONES[0][0], SAMPLES, PROPS)
        return
    for root, samples, props in DRONES:
        animate_body(root, samples, props)


'''
# replace "def build_drone():" marker we renamed earlier with registry block + it
old2 = "def build_crazyflie(spec, prefix=\"CF\"):\n    global DRONE"
new2 = TAROT + "\ndef build_crazyflie(spec, prefix=\"CF\"):\n    global DRONE"
assert old2 in s
s = s.replace(old2, new2, 1)

# ---- 4. animation: track bodies --------------------------------------------
old = '''def animate():
    DRONE.rotation_mode = "QUATERNION"'''
assert old in s
# remove the old animate body (it is superseded); keep function name
start = s.index("def animate():")
end = s.index("# --------------------------------------------------------------------------- lights")
old_anim = s[start:end]
new_anim = '''def animate():
    for root, samples, props in DRONES:
        animate_body(root, samples, props)


'''
s = s[:start] + new_anim + s[end:]

# ---- 5. multi-body roots: replace DRONE global handling --------------------
old = '''PROPS = []'''
new = '''PROPS = []
DRONES = []   # (root_empty, samples, props) per body'''
assert old in s
s = s.replace(old, new, 1)

old = '''def build_drone():
    """Build every body of the scenario (multi-target capable)."""
    for b in BODIES:
        builder = BUILDERS.get(b.get("frame", "crazyflie"), build_crazyflie)
        builder(b, prefix=("CF" if b.get("frame", "crazyflie") == "crazyflie"
                           else "T650%d" % b["body_id"]))'''
new = '''def build_drone():
    """Build every body of the scenario (multi-target capable)."""
    global DRONE
    for b in BODIES:
        is_cf = b.get("frame", "crazyflie") == "crazyflie"
        PROPS.clear() if False else None
        before = len(PROPS)
        builder = BUILDERS.get(b.get("frame", "crazyflie"), build_crazyflie)
        builder(b, prefix=("CF" if is_cf else "T650%d" % b["body_id"]))
        root = DRONE
        props = PROPS[before:]
        DRONES.append((root, b["samples"], props))
    DRONE = DRONES[0][0]'''
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("mocap_scene.py: multi-body support added")
