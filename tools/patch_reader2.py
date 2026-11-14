import io

p = r"C:\Users\leeli\Downloads\opticaltrack\rust\mocap-sim\src\main.rs"
s = io.open(p, encoding="utf-8").read()

Q = chr(92) + chr(34)   # \" as it appears in the Rust source

# Replace ONLY read_scenario_bodies (leave parse_samples_object untouched).
i = s.index("/// Multi-body scenario reader.")
j = s.index("fn parse_samples_object(")

new = (
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
s = s[:i] + new + s[j:]
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("read_scenario_bodies replaced (parse_samples_object untouched)")
