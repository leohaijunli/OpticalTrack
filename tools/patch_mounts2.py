import io

p = r"C:\Users\leeli\Downloads\opticaltrack\rust\mocap-sim\src\rig.rs"
s = io.open(p, encoding="utf-8").read()

pairs = [
("""            // front pair: masts start inboard on the arms (clear of the prop
            // hubs) and lean out/up to the marker on the arm tip
            // tail: diagonal mast off the rear edge of the top plate
            [(0.160, -0.160, 0.028), (0.160, 0.160, 0.028), (-0.075, 0.0, 0.064)],""",
"""            [
                tarot_mount(0.230, -0.230),
                tarot_mount(0.230, 0.230),
                tarot_mount(-0.290, 0.0),
            ],"""),
("""            // all three masts stand on the upper centre plate
            [(0.050, -0.140, 0.065), (0.050, 0.140, 0.065), (-0.150, 0.0, 0.065)],""",
"""            [
                tarot_mount(0.050, -0.140),
                tarot_mount(0.050, 0.140),
                tarot_mount(-0.190, 0.0),
            ],"""),
("""            [(-0.070, -0.180, 0.065), (-0.070, 0.180, 0.065), (0.075, 0.0, 0.065)],""",
"""            [
                tarot_mount(-0.070, -0.180),
                tarot_mount(-0.070, 0.180),
                tarot_mount(0.240, 0.0),
            ],"""),
]
for old, new in pairs:
    assert old in s, old[:60]
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("mounts now derived from airframe structure")
