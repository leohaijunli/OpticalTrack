import io, json

# ---- fix the Rust writer's insertion anchor for future runs ----
p = r"C:\Users\leeli\Downloads\opticaltrack\rust\mocap-sim\src\main.rs"
s = io.open(p, encoding="utf-8").read()
old = 'if let Some(pos) = text.find("  \\"markers\\"") {'
new = 'if let Some(pos) = text.find("\\"markers\\"") {'
if old in s:
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("rust writer anchor fixed")

# ---- patch the already-generated rig3 json ----
j = r"C:\Users\leeli\Downloads\opticaltrack\out\rig3\rig_blender.json"
text = io.open(j, encoding="utf-8").read()
if '"bodies"' not in text:
    specs = [
        (0, "T650-A", "tarot650_x", 0.24, (0.460, 0.569),
         [(0, 0.230, -0.230, 0.165), (1, 0.230, 0.230, 0.165), (2, -0.290, 0.0, 0.185)]),
        (1, "T650-B", "tarot650_fwd", 0.17, (0.280, 0.278),
         [(0, 0.050, -0.140, 0.205), (1, 0.050, 0.140, 0.205), (2, -0.190, 0.0, 0.205)]),
        (2, "T650-C", "tarot650_rev", 0.17, (0.360, 0.358),
         [(0, -0.070, -0.180, 0.205), (1, -0.070, 0.180, 0.205), (2, 0.240, 0.0, 0.205)]),
    ]
    def ned(x, y, z):
        return (x, -y, -z)
    parts = []
    for (bid, name, frame, br, sig, marks) in specs:
        ms = ",".join(
            '{"id":%d,"local":[%.4f,%.4f,%.4f],"local_ned_body":[%.4f,%.4f,%.4f],"radius_m":0.0120}'
            % (m[0], m[1], m[2], m[3], *ned(m[1], m[2], m[3])) for m in marks)
        parts.append('    {"body_id":%d,"name":"%s","frame":"%s","body_radius_m":%.3f,'
                     '"signature":[%.4f,%.4f],"markers":[%s]}'
                     % (bid, name, frame, br, sig[0], sig[1], ms))
    block = '  "bodies": [\n' + ",\n".join(parts) + '\n  ],\n'
    pos = text.find('"markers"')
    text = text[:pos] + block + text[pos:]
    io.open(j, "w", encoding="utf-8", newline="\n").write(text)
    print("rig3 json: bodies inserted")

d = json.load(open(j, encoding="utf-8"))
for b in d["bodies"]:
    print("  body", b["body_id"], b["name"], b["frame"], "sig", [round(x, 3) for x in b["signature"]],
          "markers", len(b["markers"]))
