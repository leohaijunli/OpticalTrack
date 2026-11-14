import io

p = r"C:\Users\leeli\Downloads\opticaltrack\rust\mocap-sim\src\rig.rs"
s = io.open(p, encoding="utf-8").read()

old = """    mounts: [(f64, f64, f64); 3],
) -> Vec<Marker> {"""
new = """    mounts: [V3; 3],
) -> Vec<Marker> {"""
assert old in s
s = s.replace(old, new, 1)

old = """            mount_local: v3(mounts[i].0, mounts[i].1, mounts[i].2),"""
new = """            mount_local: mounts[i],"""
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("tri() now takes V3 mounts")
