import io

p = r"C:\Users\leeli\Downloads\opticaltrack\rust\mocap-sim\src\main.rs"
s = io.open(p, encoding="utf-8").read()

Q = chr(92) + chr(34)   # \" in the Rust source


def splice(s, start_marker, end_marker, new_text, include_start=True, include_end=False):
    i = s.index(start_marker)
    j = s.index(end_marker, i)
    a = i if include_start else i
    b = j if include_end else j
    return s[:a] + new_text + s[b:], i, j


# ---- 1. reader: from the doc comment to just before write_scenario_multi ----
i = s.index("/// Multi-body scenario reader.")
j = s.index("fn write_scenario_multi(")

new_reader = (
    "/// Multi-body scenario reader.  Returns (body_id, frame key, samples) per\n"
    "/// body; falls back to the legacy single-body \"samples\" list as body 0.\n"
    "fn read_scenario_bodies(path: &str) -> Vec<(usize, String, Vec<Sample>)> {\n"
    "    let text = std::fs::read_to_string(path).expect(\"read scenario\");\n"
    "    if let Some(bpos) = text.find(\"" + Q + "bodies" + Q + "\") {\n"
    "        let tail = &text[bpos..];\n"
    "        let region = match tail.find(\"\\n  ],\") {\n"
    "            Some(i) => &tail[..i],\n"
    "            None => tail,\n"
    "        };\n"
    "        let mut marks: Vec<usize> = Vec::new();\n"
    "        let mut from = 0usize;\n"
    "        while let Some(rel) = region[from..].find(\"" + Q + "body_id" + Q + "\") {\n"
    "            marks.push(from + rel);\n"
    "            from += rel + 9;\n"
    "        }\n"
    "        let mut out: Vec<(usize, String, Vec<Sample>)> = Vec::new();\n"
    "        for (k, &m) in marks.iter().enumerate() {\n"
    "            let seg_end = marks.get(k + 1).copied().unwrap_or(region.len());\n"
    "            let chunk = &region[m..seg_end];\n"
    "            let id = grab_num(chunk, \"body_id\") as usize;\n"
    "            let frame = grab_str(chunk, \"frame\").unwrap_or_else(|| \"crazyflie\".to_string());\n"
    "            let samples = parse_samples_object(chunk);\n"
    "            if !samples.is_empty() {\n"
    "                out.push((id, frame, samples));\n"
    "            }\n"
    "        }\n"
    "        if !out.is_empty() {\n"
    "            return out;\n"
    "        }\n"
    "    }\n"
    "    vec![(0usize, \"crazyflie\".to_string(), read_scenario(path))]\n"
    "}\n"
    "\n"
    "fn after_key<'a>(s: &'a str, key: &str) -> Option<&'a str> {\n"
    "    let k = format!(\"" + Q + "{}" + Q + ":\", key);\n"
    "    let i = s.find(&k)?;\n"
    "    s.get(i + k.len()..)\n"
    "}\n"
    "\n"
    "fn grab_num(s: &str, key: &str) -> f64 {\n"
    "    after_key(s, key)\n"
    "        .and_then(|r| {\n"
    "            r.trim_start()\n"
    "                .split(|c: char| c == ',' || c == '}')\n"
    "                .next()\n"
    "        })\n"
    "        .and_then(|t| t.trim().parse::<f64>().ok())\n"
    "        .unwrap_or(0.0)\n"
    "}\n"
    "\n"
    "fn grab_str(s: &str, key: &str) -> Option<String> {\n"
    "    let r = after_key(s, key)?;\n"
    "    let t = r.trim_start();\n"
    "    if t.starts_with('\"') {\n"
    "        let e = t[1..].find('\"')? + 1;\n"
    "        Some(t[1..e].to_string())\n"
    "    } else {\n"
    "        None\n"
    "    }\n"
    "}\n"
    "\n"
)
s = s[:i] + new_reader + s[j:]

# ---- 2. arr3 / arr4 blocks: replace by index -------------------------------
i = s.index("        let arr3 = |key: &str| -> [f64; 3] {")
j = s.index("        let arr4 = |key: &str| -> [f64; 4] {")
new_arr3 = (
    "        let arr3 = |key: &str| -> [f64; 3] {\n"
    "            let mut v = [0.0; 3];\n"
    "            if let Some(seg) = after_key(obj, key) {\n"
    "                if let Some(l) = seg.find('[') {\n"
    "                    if let Some(r) = seg[l..].find(']') {\n"
    "                        if let Some(inner) = seg.get(l + 1..l + r) {\n"
    "                            for (j2, part) in inner.split(',').enumerate().take(3) {\n"
    "                                v[j2] = part.trim().parse::<f64>().unwrap_or(0.0);\n"
    "                            }\n"
    "                        }\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "            v\n"
    "        };\n"
)
s = s[:i] + new_arr3 + s[j:]

i = s.index("        let arr4 = |key: &str| -> [f64; 4] {")
j = s.index("        out.push(Sample {")
new_arr4 = (
    "        let arr4 = |key: &str| -> [f64; 4] {\n"
    "            let mut v = [0.0; 4];\n"
    "            if let Some(seg) = after_key(obj, key) {\n"
    "                if let Some(l) = seg.find('[') {\n"
    "                    if let Some(r) = seg[l..].find(']') {\n"
    "                        if let Some(inner) = seg.get(l + 1..l + r) {\n"
    "                            for (j2, part) in inner.split(',').enumerate().take(4) {\n"
    "                                v[j2] = part.trim().parse::<f64>().unwrap_or(0.0);\n"
    "                            }\n"
    "                        }\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "            v\n"
    "        };\n"
)
s = s[:i] + new_arr4 + s[j:]

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("reader + arr helpers rewritten")
