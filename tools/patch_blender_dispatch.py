import io

p = r"C:\Users\leeli\Downloads\opticaltrack\blender\mocap_scene.py"
s = io.open(p, encoding="utf-8").read()

# 1. remove the module-level BUILDERS dict (it forward-references
#    build_crazyflie which is defined later -> NameError at import time)
old = (
    "BUILDERS = {\n"
    "    \"crazyflie\": build_crazyflie,\n"
    "    \"tarot650_x\": build_tarot650,\n"
    "    \"tarot650_fwd\": build_tarot650,\n"
    "    \"tarot650_rev\": build_tarot650,\n"
    "}\n"
    "\n"
)
assert old in s, "BUILDERS block not found"
s = s.replace(old, "", 1)

# 2. dispatch by frame name inside build_drone
old2 = '        builder = BUILDERS.get(b.get("frame", "crazyflie"), build_crazyflie)'
new2 = (
    '        frame = b.get("frame", "crazyflie")\n'
    "        if frame == \"crazyflie\":\n"
    "            builder = build_crazyflie\n"
    "        else:\n"
    "            builder = build_tarot650"
)
assert old2 in s, "dispatch line not found"
s = s.replace(old2, new2, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("BUILDERS dispatch fixed")
