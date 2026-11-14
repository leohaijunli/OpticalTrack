"""Give every marker a physical mount point on the airframe.

A mocap marker that floats in mid-air is wrong.  Real rigs bolt a short mast
(carbon rod / standoff) to the frame and glue the retro-reflective marker to
its tip.  The dataset must therefore carry, per marker, the point where the
mast attaches to the airframe; Blender draws the rod between the two.

`mount_local` is in the same body frame as `local`, metres.
"""
import io
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RIG = os.path.join(ROOT, "rust", "mocap-sim", "src", "rig.rs")

s = io.open(RIG, encoding="utf-8").read()

# ---------------------------------------------------------------- 1. struct
old = """pub struct Marker {
    pub id: usize,
    pub local: V3,
    pub radius_m: f64,
    pub normal_local: V3,
}"""
new = """pub struct Marker {
    pub id: usize,
    pub local: V3,
    pub radius_m: f64,
    pub normal_local: V3,
    /// Where the marker's mast attaches to the airframe (body frame, metres).
    /// Blender draws a rod from here to `local`; a marker that is not bolted
    /// to the frame is a modelling error, not a shortcut.
    pub mount_local: V3,
}"""
assert old in s, "Marker struct not found"
s = s.replace(old, new, 1)

# ------------------------------------------------------- 2. crazyflie layout
# tiny 5 mm standoffs straight down onto the PCB
s = s.replace(
    """            local: v3(0.030, -0.0235, 0.0135),
            radius_m: 0.010,
            normal_local: v3(0.0, 0.0, 1.0),""",
    """            local: v3(0.030, -0.0235, 0.0135),
            radius_m: 0.010,
            normal_local: v3(0.0, 0.0, 1.0),
            mount_local: v3(0.030, -0.0235, 0.0065),""", 1)
s = s.replace(
    """            local: v3(0.030, 0.0235, 0.0135),
            radius_m: 0.010,
            normal_local: v3(0.0, 0.0, 1.0),""",
    """            local: v3(0.030, 0.0235, 0.0135),
            radius_m: 0.010,
            normal_local: v3(0.0, 0.0, 1.0),
            mount_local: v3(0.030, 0.0235, 0.0065),""", 1)
s = s.replace(
    """            local: v3(-0.0355, 0.0, 0.0155),
            radius_m: 0.010,
            normal_local: v3(0.0, 0.0, 1.0),""",
    """            local: v3(-0.0355, 0.0, 0.0155),
            radius_m: 0.010,
            normal_local: v3(0.0, 0.0, 1.0),
            mount_local: v3(-0.0355, 0.0, 0.0085),""", 1)

# ------------------------------------------------------------ 3. tri() helper
old = """fn tri(a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64), r: f64) -> Vec<Marker> {
    vec![
        Marker { id: 0, local: v3(a.0, a.1, a.2), radius_m: r, normal_local: v3(0.0, 0.0, 1.0) },
        Marker { id: 1, local: v3(b.0, b.1, b.2), radius_m: r, normal_local: v3(0.0, 0.0, 1.0) },
        Marker { id: 2, local: v3(c.0, c.1, c.2), radius_m: r, normal_local: v3(0.0, 0.0, 1.0) },
    ]
}"""
new = """fn tri(
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    c: (f64, f64, f64),
    r: f64,
    mounts: [(f64, f64, f64); 3],
) -> Vec<Marker> {
    let p = [a, b, c];
    (0..3)
        .map(|i| Marker {
            id: i,
            local: v3(p[i].0, p[i].1, p[i].2),
            radius_m: r,
            normal_local: v3(0.0, 0.0, 1.0),
            mount_local: v3(mounts[i].0, mounts[i].1, mounts[i].2),
        })
        .collect()
}"""
assert old in s, "tri() not found"
s = s.replace(old, new, 1)

# ------------------------------------------- 4. tarot650 mount points per rig
old = """        0 => tri(
            (0.230, -0.230, 0.165),
            (0.230, 0.230, 0.165),
            (-0.290, 0.0, 0.185),
            0.012,
        ),"""
new = """        0 => tri(
            (0.230, -0.230, 0.165),
            (0.230, 0.230, 0.165),
            (-0.290, 0.0, 0.185),
            0.012,
            // front pair: vertical masts on the motor-bell tops
            // tail: diagonal mast off the rear edge of the top plate
            [(0.230, -0.230, 0.062), (0.230, 0.230, 0.062), (-0.075, 0.0, 0.064)],
        ),"""
assert old in s
s = s.replace(old, new, 1)

old = """        1 => tri(
            (0.050, -0.140, 0.205),
            (0.050, 0.140, 0.205),
            (-0.190, 0.0, 0.205),
            0.012,
        ),"""
new = """        1 => tri(
            (0.050, -0.140, 0.205),
            (0.050, 0.140, 0.205),
            (-0.190, 0.0, 0.205),
            0.012,
            // all three masts stand on the upper centre plate
            [(0.050, -0.140, 0.065), (0.050, 0.140, 0.065), (-0.150, 0.0, 0.065)],
        ),"""
assert old in s
s = s.replace(old, new, 1)

old = """        2 => tri(
            (-0.070, -0.180, 0.205),
            (-0.070, 0.180, 0.205),
            (0.240, 0.0, 0.205),
            0.012,
        ),"""
new = """        2 => tri(
            (-0.070, -0.180, 0.205),
            (-0.070, 0.180, 0.205),
            (0.240, 0.0, 0.205),
            0.012,
            [(-0.070, -0.180, 0.065), (-0.070, 0.180, 0.065), (0.075, 0.0, 0.065)],
        ),"""
assert old in s
s = s.replace(old, new, 1)

io.open(RIG, "w", encoding="utf-8", newline="\n").write(s)
print("rig.rs: mount_local added to every marker")

# --------------------------------------------------- 5. emit mount in rig json
MAIN = os.path.join(ROOT, "rust", "mocap-sim", "src", "main.rs")
m = io.open(MAIN, encoding="utf-8").read()
old = '''                "{{\\"id\\":{},\\"local\\":[{:.4},{:.4},{:.4}],\\"local_ned_body\\":[{:.4},{:.4},{:.4}],\\"radius_m\\":{:.4}}}"'''
assert old in m, "marker json format string not found"
new = '''                "{{\\"id\\":{},\\"local\\":[{:.4},{:.4},{:.4}],\\"mount\\":[{:.4},{:.4},{:.4}],\\"local_ned_body\\":[{:.4},{:.4},{:.4}],\\"radius_m\\":{:.4}}}"'''
m = m.replace(old, new, 1)

old = """                m.id,
                m.local.x, m.local.y, m.local.z,
                rig::marker_in_ned_body(m).x,"""
new = """                m.id,
                m.local.x, m.local.y, m.local.z,
                m.mount_local.x, m.mount_local.y, m.mount_local.z,
                rig::marker_in_ned_body(m).x,"""
assert old in m, "marker json args not found"
m = m.replace(old, new, 1)
io.open(MAIN, "w", encoding="utf-8", newline="\n").write(m)
print("main.rs: mount emitted in rig_blender.json")
