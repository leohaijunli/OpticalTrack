"""Ray-cast a few pixels in each mocap camera POV and report what is hit.

Used to prove nothing structural (a column, truss rail, roof tie) is sitting in
front of the lens, which would render as a black polygon in the middle of the
frame.
"""
import bpy
from mathutils import Vector

sc = bpy.context.scene
sc.frame_set(300)
deps = bpy.context.evaluated_depsgraph_get()
W, H = 1280, 800

PIXELS = [(640, 400), (640, 600), (320, 400), (960, 400), (640, 200), (160, 700), (1120, 700)]

for cam in [o for o in bpy.data.objects if o.type == "CAMERA" and o.name.startswith("CAM")]:
    lens = cam.data.lens
    sw = cam.data.sensor_width
    tan_h = (sw * 0.5) / lens
    tan_v = tan_h * H / W
    M = cam.matrix_world
    origin = M.translation
    worst = None
    print(f"--- {cam.name}  pos={tuple(round(x, 2) for x in origin)}")
    for (u, v) in PIXELS:
        d = Vector(((u / W * 2 - 1) * tan_h, (1 - v / H * 2) * tan_v, -1.0))
        d = (M.to_3x3() @ d).normalized()
        # In POV renders every camera fixture (CAMxx_*) is hidden, so ignore
        # them here as well; we only care about the hangar and rig structure.
        ok, loc, nor, idx, obj, mat = sc.ray_cast(deps, origin, d)
        guard = 0
        while ok and obj is not None and (
            obj.name.startswith("CAM")
            or obj.name.startswith("Frustum")
            or obj.name.startswith("VolumeEdge")
        ) and guard < 32:
            origin2 = loc + d * 1e-3
            ok, loc, nor, idx, obj, mat = sc.ray_cast(deps, origin2, d)
            guard += 1
        name = obj.name if obj else "-"
        dist = (loc - M.translation).length if ok else 0.0
        if ok and (worst is None or dist < worst[0]):
            worst = (dist, name)
        print(f"  px({u:4d},{v:3d}) -> {name:16s} {dist:7.2f} m")
    print(f"  nearest: {worst}")
