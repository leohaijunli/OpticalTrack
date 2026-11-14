"""Multi-body support for main.rs: demo --bodies N, multi scenario/CSV writers,
and a body-aware `simulate`.  Splices new functions + modifies the two commands."""
import io, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "rust", "mocap-sim", "src", "main.rs")

src = io.open(SRC, encoding="utf-8").read()
orig_len = len(src)

# ---------------------------------------------------------------- 1. demo cmd
old_demo = '''        "demo" => {
            let out = get("out", "logs/demo_crazyflie.mavlink");
            let fl = demo::build();
            println!("flight: {}", demo::summary(&fl));
            let t = get("probe-t", "0.079").parse::<f64>().unwrap_or(0.079);
            let h = 1e-4;
            let p = demo::demo_pos(t);
            let v = demo::demo_pos(t + h).sub(demo::demo_pos(t - h)).mul(1.0 / (2.0 * h));
            let (pt, vt) = fl.track.eval(t);
            println!("debug t={t:.4} geo p={:?} v={:?}", p.arr(), v.arr());
            println!("debug     track p={:?} v={:?}", pt.arr(), vt.arr());
            match demo::write_log(&dialect, &out, &fl) {
                Ok(n) => println!("wrote {out} ({:.1} KiB, MAVLink 2)", n as f64 / 1024.0),
                Err(e) => eprintln!("write failed: {e}"),
            }
        }'''
new_demo = '''        "demo" => {
            let out = get("out", "logs/demo_crazyflie.mavlink");
            let n_bodies = get("bodies", "1").parse::<usize>().unwrap_or(1);
            let fl = demo::build();
            println!("flight: {}", demo::summary(&fl));
            let t = get("probe-t", "0.079").parse::<f64>().unwrap_or(0.079);
            let h = 1e-4;
            let p = demo::demo_pos(t);
            let v = demo::demo_pos(t + h).sub(demo::demo_pos(t - h)).mul(1.0 / (2.0 * h));
            let (pt, vt) = fl.track.eval(t);
            println!("debug t={t:.4} geo p={:?} v={:?}", p.arr(), v.arr());
            println!("debug     track p={:?} v={:?}", pt.arr(), vt.arr());
            if n_bodies <= 1 {
                match demo::write_log(&dialect, &out, &fl) {
                    Ok(n) => println!("wrote {out} ({:.1} KiB, MAVLink 2)", n as f64 / 1024.0),
                    Err(e) => eprintln!("write failed: {e}"),
                }
                return;
            }
            // ---- multi-body: N Tarot 650 quads, one MAVLink system each ----
            let specs = demo::tarot650_specs();
            let specs = &specs[..n_bodies.min(specs.len())];
            let flights = demo::build_multi(specs);
            for (i, f) in flights.iter().enumerate() {
                println!("body {}: {}", i, demo::summary(f));
            }
            let refs: Vec<(u8, &demo::DemoFlight)> =
                flights.iter().enumerate().map(|(i, f)| ((i + 1) as u8, f)).collect();
            match demo::write_log_multi(&dialect, &out, &refs) {
                Ok(n) => println!(
                    "wrote {out} ({:.1} KiB, MAVLink 2, {} systems sysid 1..{})",
                    n as f64 / 1024.0,
                    refs.len(),
                    refs.len()
                ),
                Err(e) => eprintln!("write failed: {e}"),
            }
            // scenario straight from the analytic truth (the log is for the
            // MAVLink ingest path; the dataset ground truth is exact).
            if has("scenario") {
                let scen = get("scenario", "out/scenario3.json");
                let fps = get("fps", "30").parse::<f64>().unwrap_or(30.0);
                let bodies: Vec<(usize, String, Vec<Sample>)> = flights
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        (i, format!("tarot650_x tarot650_fwd tarot650_rev".split(' ').nth(i).unwrap_or("tarot650")),
                         flight::samples_from(&f.track, None, fps, f.duration))
                    })
                    .collect();
                let refs2: Vec<(usize, String, &[Sample])> =
                    bodies.iter().map(|(i, fr, s)| (*i, fr.clone(), s.as_slice())).collect();
                write_scenario_multi(&scen, &refs2, fps, &out);
                let csv = scen.replace("scenario3.json", "rig/trajectory3.csv").replace(
                    &format!("scenario{}.json", n_bodies),
                    &format!("rig/trajectory{}.csv", n_bodies),
                );
                write_trajectory_csv_multi(&csv, &refs2);
                println!("scenario: {} bodies x {} frames -> {scen}", bodies.len(), bodies[0].2.len());
                println!("trajectory: {csv}");
            }
        }'''
assert old_demo in src, "demo cmd block not found"
src = src.replace(old_demo, new_demo, 1)

# ------------------------------------------------------- 2. simulate cmd
old_sim = '''        "simulate" => {
            let scen = get("scenario", "out/scenario.json");
            let dir = get("out", "out/rig");
            let data = read_scenario(&scen);
            let cams = rig::default_rig();
            let markers = rig::crazyflie_markers();
            std::fs::create_dir_all(&dir).expect("create out dir");
            write_rig_json(&dir, &cams, &markers, &data);
            let n_det = write_detections(&dir, &data, &cams, &markers);
            println!(
                "rig: {} cameras, {} markers, {} detections -> {dir}",
                cams.len(),
                markers.len(),
                n_det
            );
        }'''
new_sim = '''        "simulate" => {
            let scen = get("scenario", "out/scenario.json");
            let dir = get("out", "out/rig");
            let cams = rig::default_rig();
            std::fs::create_dir_all(&dir).expect("create out dir");
            let bodies = read_scenario_bodies(&scen);
            if bodies.len() > 1 {
                let specs = rig::tarot650_bodies();
                let n_det = write_detections_multi(&dir, &cams, &specs, &bodies);
                write_rig_json_multi(&dir, &cams, &specs, &bodies);
                println!(
                    "rig: {} cameras, {} bodies, {} detections -> {dir}",
                    cams.len(),
                    bodies.len(),
                    n_det
                );
            } else {
                let (bid, frame, data) = &bodies[0];
                let markers = if frame == "crazyflie" {
                    rig::crazyflie_markers()
                } else {
                    rig::tarot650_markers(frame.replace("tarot650", "").parse::<usize>().unwrap_or(0))
                };
                write_rig_json(&dir, &cams, &markers, data);
                let n_det = write_detections(&dir, data, &cams, &markers);
                println!(
                    "rig: {} cameras, {} markers, {} detections -> {dir}",
                    cams.len(),
                    markers.len(),
                    n_det
                );
            }
        }'''
assert old_sim in src, "simulate cmd block not found"
src = src.replace(old_sim, new_sim, 1)

# ------------------------------------------------ 3. new multi-body functions
ADD = r'''

/// Multi-body scenario reader.  Returns (body_id, frame key, samples) per
/// body; falls back to the legacy single-body "samples" list as body 0.
fn read_scenario_bodies(path: &str) -> Vec<(usize, String, Vec<Sample>)> {
    let text = std::fs::read_to_string(path).expect("read scenario");
    if let Some(pos) = text.find("\"bodies\"") {
        let mut out = Vec::new();
        for chunk in text[pos..].split("{").skip(1) {
            let grab = |key: &str| -> Option<String> {
                let k = format!("\"{}\"", key);
                let i = chunk.find(&k)?;
                let rest = &chunk[i + k.len()..];
                let s = rest.find(":")? + 1;
                let e = rest[s..].find(",")? + s;
                Some(rest[s..e].trim().trim_matches('"').to_string())
            };
            let id = match grab("body_id").and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => v,
                None => continue,
            };
            let frame = grab("frame").unwrap_or_else(|| "crazyflie".to_string());
            let samples = parse_samples_object(chunk);
            if !samples.is_empty() {
                out.push((id, frame, samples));
            }
            if out.len() >= 8 {
                break;
            }
        }
        if !out.is_empty() {
            return out;
        }
    }
    vec![(0usize, "crazyflie".to_string(), read_scenario(path))]
}

fn parse_samples_object(chunk: &str) -> Vec<Sample> {
    // chunk starts just after "{": find the "samples" [ ... ] array inside it
    let start = match chunk.find("\"samples\"") {
        Some(i) => i,
        None => return Vec::new(),
    };
    let lb = match chunk[start..].find('[') {
        Some(i) => start + i,
        None => return Vec::new(),
    };
    // matching close bracket
    let bytes = chunk.as_bytes();
    let mut depth = 0usize;
    let mut rb = lb;
    for (i, &c) in bytes.iter().enumerate().skip(lb) {
        if c == b'[' {
            depth += 1;
        } else if c == b']' {
            depth -= 1;
            if depth == 0 {
                rb = i;
                break;
            }
        }
    }
    let arr = &chunk[lb + 1..rb];
    let mut out = Vec::new();
    for obj in arr.split("{").skip(1) {
        let num = |key: &str| -> f64 {
            let k = format!("\"{}\":", key);
            match obj.find(&k) {
                Some(i) => obj[i + k.len()..]
                    .trim()
                    .trim_end_matches(',')
                    .trim()
                    .parse::<f64>()
                    .unwrap_or(0.0),
                None => 0.0,
            }
        };
        let arr3 = |key: &str| -> [f64; 3] {
            let k = format!("\"{}\":", key);
            let mut v = [0.0; 3];
            if let Some(i) = obj.find(&k) {
                let seg = &obj[i + k.len()..];
                if let Some(l) = seg.find('[') {
                    if let Some(r) = seg[l..].find(']') {
                        for (j, part) in seg[l + 1..l + r].split(',').enumerate().take(3) {
                            v[j] = part.trim().parse::<f64>().unwrap_or(0.0);
                        }
                    }
                }
            }
            v
        };
        let arr4 = |key: &str| -> [f64; 4] {
            let k = format!("\"{}\":", key);
            let mut v = [0.0; 4];
            if let Some(i) = obj.find(&k) {
                let seg = &obj[i + k.len()..];
                if let Some(l) = seg.find('[') {
                    if let Some(r) = seg[l..].find(']') {
                        for (j, part) in seg[l + 1..l + r].split(',').enumerate().take(4) {
                            v[j] = part.trim().parse::<f64>().unwrap_or(0.0);
                        }
                    }
                }
            }
            v
        };
        let p = arr3("p");
        let v3v = arr3("v");
        let qq = arr4("q");
        let w = arr3("w");
        out.push(Sample {
            t: num("t"),
            p: v3(p[0], p[1], p[2]),
            v: v3(v3v[0], v3v[1], v3v[2]),
            q: quat(qq[0], qq[1], qq[2], qq[3]),
            w: v3(w[0], w[1], w[2]),
            on_ground: num("on_ground") > 0.5,
        });
    }
    out
}

fn write_scenario_multi(
    path: &str,
    bodies: &[(usize, String, &[Sample])],
    fps: f64,
    src_note: &str,
) {
    let mut j = J::new();
    j.raw("{\n");
    j.key("meta");
    j.raw("{");
    j.kv_str("generator", "mocap-sim 0.2.0 multi-body");
    j.raw(",");
    j.kv_str("source", src_note);
    j.raw(",");
    j.kv_str("units", "metres");
    j.raw(",");
    j.kv_str("world", "ENU, Z-up (Blender)");
    j.raw(",");
    j.kv_num("fps", fps);
    j.raw(",");
    j.kv_num("frame_count", bodies.get(0).map(|b| b.2.len()).unwrap_or(0) as f64);
    j.raw(",");
    j.kv_num("duration_s", bodies.get(0).map(|b| b.2.len() as f64 / fps).unwrap_or(0.0));
    j.raw(",");
    j.kv_num("bodies", bodies.len() as f64);
    j.raw(",");
    j.kv_str("position_source", "analytic figure-eight (per body)");
    j.raw(",");
    j.kv_str("attitude_source", "thrust-vector tilt + velocity yaw (per body)");
    j.raw("},\n");

    j.key("bodies");
    j.raw("[\n");
    for (bi, (id, frame, data)) in bodies.iter().enumerate() {
        if bi > 0 {
            j.raw(",\n");
        }
        j.raw("{");
        j.kv_num("body_id", *id as f64);
        j.raw(",");
        j.kv_str("frame", frame);
        j.raw(",");
        j.key("samples");
        j.raw("[\n");
        for (i, s) in data.iter().enumerate() {
            if i > 0 {
                j.raw(",\n");
            }
            j.raw("{");
            j.kv_num("t", s.t);
            j.raw(",");
            j.kv_arr3("p", s.p.arr());
            j.raw(",");
            j.kv_arr3("v", s.v.arr());
            j.raw(",");
            j.key("q");
            j.arr4([s.q.w, s.q.x, s.q.y, s.q.z]);
            j.raw(",");
            j.kv_arr3("w", s.w.arr());
            j.raw(",");
            j.key("on_ground");
            j.raw(if s.on_ground { "true" } else { "false" });
            j.raw("}");
        }
        j.raw("\n]");
        j.raw("}");
    }
    j.raw("\n],\n");

    // legacy view: body 0 as the flat samples list
    j.key("samples");
    j.raw("[\n");
    if let Some((_, _, data)) = bodies.first() {
        for (i, s) in data.iter().enumerate() {
            if i > 0 {
                j.raw(",\n");
            }
            j.raw("{");
            j.kv_num("t", s.t);
            j.raw(",");
            j.kv_arr3("p", s.p.arr());
            j.raw(",");
            j.kv_arr3("v", s.v.arr());
            j.raw(",");
            j.key("q");
            j.arr4([s.q.w, s.q.x, s.q.y, s.q.z]);
            j.raw(",");
            j.kv_arr3("w", s.w.arr());
            j.raw(",");
            j.key("on_ground");
            j.raw(if s.on_ground { "true" } else { "false" });
            j.raw("}");
        }
    }
    j.raw("\n]\n}\n");
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, j.finish()).expect("write scenario");
}

fn write_trajectory_csv_multi(path: &str, bodies: &[(usize, String, &[Sample])]) {
    let mut s = String::from("body_id,t,frame,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,on_ground\n");
    for (bid, _, data) in bodies {
        for (i, x) in data.iter().enumerate() {
            s.push_str(&format!(
                "{},{:.4},{},{:.5},{:.5},{:.5},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4},{}\n",
                bid,
                x.t,
                i,
                x.p.x,
                x.p.y,
                x.p.z,
                x.v.x,
                x.v.y,
                x.v.z,
                x.q.w,
                x.q.x,
                x.q.y,
                x.q.z,
                x.w.x,
                x.w.y,
                x.w.z,
                if x.on_ground { 1 } else { 0 }
            ));
        }
    }
    if let Some(p) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(p).ok();
    }
    std::fs::write(path, s).expect("write trajectory csv");
}

fn write_rig_json_multi(
    dir: &str,
    cams: &[Camera],
    specs: &[rig::BodySpec],
    bodies: &[(usize, String, Vec<Sample>)],
) {
    // single-body writer keeps the legacy schema; then we append the "bodies"
    // array so both consumers work off the same file.
    let frames0 = bodies.get(0).map(|b| b.2.len()).unwrap_or(0);
    let markers0 = specs.get(0).map(|s| s.markers.clone()).unwrap_or_else(rig::crazyflie_markers);
    write_rig_json(dir, cams, &markers0, &bodies.get(0).map(|b| b.2.clone()).unwrap_or_default());
    let path = format!("{dir}/rig_blender.json");
    let mut text = std::fs::read_to_string(&path).expect("read rig json");
    // frames must reflect the shared timeline
    text = re_write_frames(&text, frames0);

    let mut j = String::from("  \"bodies\": [\n");
    for (i, sp) in specs.iter().enumerate() {
        if i > 0 {
            j.push_str(",\n");
        }
        let (base, apex) = sp.signature();
        j.push_str(&format!(
            "    {{\"body_id\":{},\"name\":\"{}\",\"frame\":\"{}\",\"body_radius_m\":{:.3},\"signature\":[{:.4},{:.4}],\"markers\":[",
            sp.id, sp.name, sp.frame, sp.body_radius_m, base, apex
        ));
        for (k, m) in sp.markers.iter().enumerate() {
            if k > 0 {
                j.push(',');
            }
            j.push_str(&format!(
                "{{\"id\":{},\"local\":[{:.4},{:.4},{:.4}],\"local_ned_body\":[{:.4},{:.4},{:.4}],\"radius_m\":{:.4}}}",
                m.id,
                m.local.x, m.local.y, m.local.z,
                rig::marker_in_ned_body(m).x,
                rig::marker_in_ned_body(m).y,
                rig::marker_in_ned_body(m).z,
                m.radius_m
            ));
        }
        j.push_str("]}");
    }
    j.push_str("\n  ],\n");
    // insert before "markers" key
    if let Some(pos) = text.find("  \"markers\"") {
        text.insert_str(pos, &j);
    }
    std::fs::write(&path, text).expect("append bodies to rig json");
}

fn re_write_frames(text: &str, frames: usize) -> String {
    let key = "\"frames\":";
    match text.find(key) {
        Some(i) => {
            let rest = &text[i + key.len()..];
            let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
            format!("{}{}{}", &text[..i + key.len()], frames, &rest[end..])
        }
        None => text.to_string(),
    }
}

fn write_detections_multi(
    dir: &str,
    cams: &[Camera],
    specs: &[rig::BodySpec],
    bodies: &[(usize, String, Vec<Sample>)],
) -> usize {
    let mut det_csv = String::from("frame,body_id,t,cam,cam_name,marker,u_px,v_px,radius_px,intensity\n");
    let mut m3d_csv = String::from("frame,body_id,t,marker,x,y,z,visible_cams\n");
    let mut n_det = 0usize;
    let mut reproj_sum = 0.0;
    let mut reproj_n = 0usize;
    let mut pos_err_sum = 0.0;
    let mut rot_err_sum = 0.0;
    let mut pose_n = 0usize;
    let mut seed = 0x5EED_1234u64;

    let n_frames = bodies.get(0).map(|b| b.2.len()).unwrap_or(0);
    for fi in 0..n_frames {
        for (sp, (_, _, data)) in specs.iter().zip(bodies.iter()) {
            let s = match data.get(fi) {
                Some(s) => s,
                None => continue,
            };
            let cfg = rig::VisionConfig {
                body_radius_m: sp.body_radius_m,
                seed,
                ..rig::VisionConfig::default()
            };
            seed = seed.wrapping_add(0x9E37_79B9);
            let mut rng = Rng::new(seed);
            let obs = rig::observe(fi, s.t, s.p, s.q, cams, &sp.markers, &cfg, &mut rng);
            for d in &obs.detections {
                det_csv.push_str(&format!(
                    "{},{},{:.4},{},{},{},{:.3},{:.3},{:.2},{:.3}\n",
                    fi, sp.id, s.t, d.cam, cams[d.cam].name, d.marker, d.u, d.v, d.radius_px, d.intensity
                ));
                n_det += 1;
            }
            let mut vis_counts = vec![0usize; sp.markers.len()];
            for d in &obs.detections {
                vis_counts[d.marker] += 1;
            }
            for (mi, p) in obs.world.iter().enumerate() {
                m3d_csv.push_str(&format!(
                    "{},{},{:.4},{},{:.5},{:.5},{:.5},{}\n",
                    fi, sp.id, s.t, mi, p.x, p.y, p.z, vis_counts[mi]
                ));
            }
            let mut tri: Vec<Option<V3>> = Vec::with_capacity(sp.markers.len());
            for mi in 0..sp.markers.len() {
                let obs_list: Vec<(usize, f64, f64)> = obs
                    .detections
                    .iter()
                    .filter(|d| d.marker == mi)
                    .map(|d| (d.cam, d.u, d.v))
                    .collect();
                match rig::triangulate(cams, &obs_list) {
                    Some(t) => {
                        reproj_sum += t.reproj_rms_px;
                        reproj_n += 1;
                        tri.push(Some(t.p));
                    }
                    None => tri.push(None),
                }
            }
            let mut fitted = Vec::new();
            let mut truth = Vec::new();
            for mi in 0..sp.markers.len() {
                if let Some(p) = tri[mi] {
                    fitted.push(p);
                    truth.push(obs.world[mi]);
                }
            }
            if fitted.len() >= 3 {
                let locals: Vec<V3> = sp.markers.iter().map(|m| m.local).collect();
                let (q, t) = rig::rigid_fit(&locals, &fitted);
                let dp = t.sub(s.p).norm();
                let dq = q.mul(s.q.conj());
                let dq = if dq.w < 0.0 {
                    Quat { w: -dq.w, x: -dq.x, y: -dq.y, z: -dq.z }
                } else {
                    dq
                };
                let angle = 2.0 * dq.w.clamp(-1.0, 1.0).acos().to_degrees();
                pos_err_sum += dp;
                rot_err_sum += angle;
                pose_n += 1;
            }
        }
    }

    std::fs::write(format!("{dir}/detections_2d.csv"), det_csv).expect("write detections");
    std::fs::write(format!("{dir}/markers_3d.csv"), m3d_csv).expect("write markers");

    let mut report = String::new();
    report.push_str("mocap-sim consistency report (multi-body)\n=========================================\n");
    report.push_str(&format!("frames            : {n_frames}\n"));
    report.push_str(&format!("cameras           : {}\n", cams.len()));
    report.push_str(&format!("bodies            : {}\n", specs.len()));
    for sp in specs {
        report.push_str(&format!(
            "  body {} {:<8} frame {:<12} signature (base, apex) = ({:.3}, {:.3}) m\n",
            sp.id,
            sp.name,
            sp.frame,
            sp.signature().0,
            sp.signature().1
        ));
    }
    report.push_str(&format!("detections        : {n_det}\n"));
    report.push_str(&format!(
        "triangulation rms : {:.4} px (mean over {} marker-frames)\n",
        reproj_sum / reproj_n.max(1) as f64,
        reproj_n
    ));
    report.push_str(&format!("6DOF position err : {:.3} mm mean\n", 1000.0 * pos_err_sum / pose_n.max(1) as f64));
    report.push_str(&format!("6DOF rotation err : {:.3} deg mean\n", rot_err_sum / pose_n.max(1) as f64));
    std::fs::write(format!("{dir}/verify.txt"), report).expect("write verify");
    n_det
}
'''
marker = "\nfn usage() {"
assert marker in src, "usage() marker not found"
src = src.replace(marker, ADD + marker, 1)

# use multi writer in the "all" legacy command path is unchanged.

io.open(SRC, "w", encoding="utf-8", newline="\n").write(src)
print("main.rs patched:", orig_len, "->", len(src), "chars")
