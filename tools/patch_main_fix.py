import io
p = r"C:\Users\leeli\Downloads\opticaltrack\rust\mocap-sim\src\main.rs"
s = io.open(p, encoding="utf-8").read()

bad = "(i, format!(\"tarot650_x tarot650_fwd tarot650_rev\".split(' ').nth(i).unwrap_or(\"tarot650\")),"
good = ("(i,\n                         [\"tarot650_x\", \"tarot650_fwd\", \"tarot650_rev\"][i].to_string(),")
assert bad in s, "frames block not found"
s = s.replace(bad, good, 1)

old_q = "q: quat(qq[0], qq[1], qq[2], qq[3]),"
new_q = "q: crate::math::quat(qq[0], qq[1], qq[2], qq[3]),"
assert old_q in s, "quat call not found"
s = s.replace(old_q, new_q, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched main.rs")
