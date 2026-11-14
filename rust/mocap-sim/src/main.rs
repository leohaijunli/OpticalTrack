//! mocap-sim: MAVLink-driven optical motion-capture scene generator.
//!
//! Pipeline
//!   1. `demo`      -> write a MAVLink 2 telemetry log for a Crazyflie flight
//!   2. `ingest`    -> log (file or UDP) -> smoothed trajectory -> scenario.json
//!   3. `simulate`  -> camera rig + 2D marker detections + triangulation check
//!   4. `probe`     -> inspect any MAVLink log
//!
//! All poses are metres, ENU, Z-up (Blender world). The Blender scene consumes
//! `out/scenario.json` and `out/rig/rig_blender.json`.

mod crc;
mod demo;
mod dialect;
mod flight;
mod json;
mod layout;
mod math;
mod mavlink;
mod rig;
mod xml;

use dialect::{Dialect, Value};
use flight::{AttTrack, Sample, Track};
use json::J;
use math::{v3, Quat, Rng, V3};
use mavlink::{field, Stats};
use rig::{Camera, Marker};
use std::collections::BTreeMap;

fn main() {
    // All relative paths (logs/, out/) are defined relative to the repository
    // root, so the Rust tool and blender/mocap_scene.py always agree on `out/`
    // no matter which directory the binary is launched from.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let _ = std::env::set_current_dir(root);

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
        return;
    }
    let cmd = args[0].as_str();
    let (mut opts, mut flags) = (BTreeMap::new(), Vec::new());
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with("--") {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                opts.insert(a.trim_start_matches("--").to_string(), args[i + 1].clone());
                i += 2;
            } else {
                flags.push(a.trim_start_matches("--").to_string());
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    let get = |k: &str, d: &str| opts.get(k).cloned().unwrap_or_else(|| d.to_string());
    let has = |k: &str| flags.iter().any(|f| f == k) || opts.contains_key(k);
    let layout_config: Option<serde_json::Value> = opts.get("layout").map(|path| {
        serde_json::from_slice(&std::fs::read(path).expect("read layout JSON"))
            .expect("parse layout JSON")
    });

    let data_dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
    let dialect = Dialect::load(&[
        &format!("{data_dir}/minimal.xml"),
        &format!("{data_dir}/standard.xml"),
        &format!("{data_dir}/common.xml"),
    ])
    .expect("failed to load MAVLink dialect XML");

    match cmd {
        "demo" => {
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
            assert!(n_bodies <= 3, "the built-in Tarot demo supports at most three bodies");
            let specs = layout_config.as_ref().map(layout::flights).unwrap_or_else(demo::tarot650_specs);
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
                        (i,
                         ["tarot650_x", "tarot650_fwd", "tarot650_rev"][i].to_string(),
                         flight::samples_from(&f.track, None, fps, f.duration))
                    })
                    .collect();
                let refs2: Vec<(usize, String, &[Sample])> =
                    bodies.iter().map(|(i, fr, s)| (*i, fr.clone(), s.as_slice())).collect();
                write_scenario_multi(&scen, &refs2, fps, &out);
                let csv_default = std::path::Path::new(&scen).with_extension("csv");
                let csv = get("trajectory", &csv_default.to_string_lossy());
                write_trajectory_csv_multi(&csv, &refs2);
                println!("scenario: {} bodies x {} frames -> {scen}", bodies.len(), bodies[0].2.len());
                println!("trajectory: {csv}");
            }
        }
        "probe" => {
            let path = get("file", "logs/demo_crazyflie.mavlink");
            let bytes = std::fs::read(&path).expect("cannot read log");
            let mut st = Stats::default();
            let recs = mavlink::read_log(&bytes, &dialect, &mut st);
            println!("file        : {path} ({} bytes)", bytes.len());
            println!(
                "frames      : {} (crc errors {}, unknown ids {}, resync bytes {})",
                st.frames, st.bad_crc, st.unknown_msg, st.resync_bytes
            );
            let mut per = st.per_msg.clone();
            per.sort_by(|a, b| b.1.cmp(&a.1));
            for (n, c) in per.iter().take(24) {
                println!("  {n:<32} {c}");
            }
            if let Some(first) = recs.first() {
                println!("first frame : id {} v{} sys {}", first.msgid, first.version, first.sysid);
            }
        }
        "ingest" => {
            let file = get("file", "logs/demo_crazyflie.mavlink");
            let out = get("out", "out/scenario.json");
            let fps = get("fps", "30").parse::<f64>().unwrap_or(30.0);
            let udp = get("udp", "");
            let cap_secs = get("seconds", "0").parse::<f64>().unwrap_or(0.0);
            let fps_clip = fps;

            let bytes = if !udp.is_empty() || cap_secs > 0.0 {
                capture_udp(&udp, if cap_secs > 0.0 { cap_secs } else { 20.0 }, &get("out-log", "logs/captured.mavlink"))
            } else {
                std::fs::read(&file).expect("cannot read log file")
            };

            let mut st = Stats::default();
            let recs = mavlink::read_log(&bytes, &dialect, &mut st);
            println!(
                "ingested {} frames ({} message types), crc errors {}",
                recs.len(),
                st.per_msg.len(),
                st.bad_crc
            );
            let (traj, att_src, pos_src) = build_trajectory(&dialect, &recs);

            println!("position source: {pos_src}, attitude source: {att_src}");
            if traj.keys.keys.is_empty() {
                eprintln!("no usable pose data in log");
                return;
            }
            let duration = traj.keys.t_end() - traj.keys.t_start();
            let dur = if opts.contains_key("duration") {
                get("duration", "0").parse::<f64>().unwrap_or(duration)
            } else {
                duration
            };
            let data = flight::samples_from(&traj.keys, traj.att.as_ref(), fps_clip, dur);
            write_scenario(&out, &data, fps_clip, &file, &st, &att_src, &pos_src);
            println!(
                "scenario: {} frames @ {:.0} fps ({:.2} s) -> {out}",
                data.len(),
                fps_clip,
                data.len() as f64 / fps_clip
            );
            let csv_default = std::path::Path::new(&out).with_extension("csv");
            let csv = get("trajectory", &csv_default.to_string_lossy());
            write_trajectory_csv(&csv, &data);
            println!("trajectory: {csv}");
        }
        "simulate" => {
            let scen = get("scenario", "out/scenario.json");
            let dir = get("out", "out/rig");
            let cams = layout_config.as_ref().map(layout::cameras).unwrap_or_else(rig::default_rig);
            std::fs::create_dir_all(&dir).expect("create out dir");
            let bodies = read_scenario_bodies(&scen);
            if bodies.len() > 1 {
                let registry = rig::tarot650_bodies();
                let specs: Vec<_> = bodies.iter().map(|(id, frame, _)| {
                    let mut body = registry.iter().find(|body| body.frame == frame)
                        .unwrap_or_else(|| panic!("unknown airframe {frame}")).clone();
                    body.id = *id;
                    body
                }).collect();
                let n_det = write_detections_multi(&dir, &cams, &specs, &bodies);
                write_rig_json_multi(&dir, &cams, &specs, &bodies);
                if let Some(ref cfg) = layout_config {
                    layout::apply_metadata(&dir, cfg);
                }
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
                if let Some(ref cfg) = layout_config {
                    layout::apply_metadata(&dir, cfg);
                }
                let n_det = write_detections(&dir, data, &cams, &markers);
                println!(
                    "rig: {} cameras, {} markers, {} detections -> {dir}",
                    cams.len(),
                    markers.len(),
                    n_det
                );
            }
        }
        "all" => {
            let fl = demo::build();
            println!("flight: {}", demo::summary(&fl));
            demo::write_log(&dialect, "logs/demo_crazyflie.mavlink", &fl).expect("write log");
            println!("wrote logs/demo_crazyflie.mavlink");
        }
        "messages" => {
            let filter = get("filter", "");
            println!("{:>6}  {:<34} {:>5} {:>6}", "id", "name", "crc", "bytes");
            for m in &dialect.messages {
                if !filter.is_empty() && !m.name.to_lowercase().contains(&filter.to_lowercase()) {
                    continue;
                }
                println!(
                    "{:>6}  {:<34} {:>5} {:>6}",
                    m.id, m.name, m.crc_extra, m.payload_len
                );
                if !filter.is_empty() {
                    for &i in &m.wire {
                        let f = &m.fields[i];
                        println!(
                            "         {:<10} {:<28} x{}{}",
                            f.elem,
                            f.name,
                            f.count,
                            if f.ext { "  (ext)" } else { "" }
                        );
                    }
                }
            }
            println!("{} messages in dialect", dialect.messages.len());
        }
        _ => usage(),
    }
}


/// Multi-body scenario reader.  Returns (body_id, frame key, samples) per
/// body; falls back to the legacy single-body "samples" list as body 0.
fn read_scenario_bodies(path: &str) -> Vec<(usize, String, Vec<Sample>)> {
    let text = std::fs::read_to_string(path).expect("read scenario");
    if let Some(bpos) = text.find("\"bodies\"") {
        let tail = &text[bpos..];
        let region = match tail.find("\n  ],") {
            Some(i) => &tail[..i],
            None => tail,
        };
        let mut marks: Vec<usize> = Vec::new();
        let mut from = 0usize;
        while let Some(rel) = region[from..].find("\"body_id\"") {
            marks.push(from + rel);
            from += rel + 9;
        }
        let mut out: Vec<(usize, String, Vec<Sample>)> = Vec::new();
        for (k, &m) in marks.iter().enumerate() {
            let seg_end = marks.get(k + 1).copied().unwrap_or(region.len());
            let chunk = &region[m..seg_end];
            let id = grab_num(chunk, "body_id") as usize;
            let frame = grab_str(chunk, "frame").unwrap_or_else(|| "crazyflie".to_string());
            let samples = parse_samples_object(chunk);
            if !samples.is_empty() {
                out.push((id, frame, samples));
            }
        }
        if !out.is_empty() {
            return out;
        }
    }
    vec![(0usize, "crazyflie".to_string(), read_scenario(path))]
}

fn after_key<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let k = format!("\"{}\":", key);
    let i = s.find(&k)?;
    s.get(i + k.len()..)
}

fn grab_num(s: &str, key: &str) -> f64 {
    after_key(s, key)
        .and_then(|r| {
            r.trim_start()
                .split(|c: char| c == ',' || c == '}')
                .next()
        })
        .and_then(|t| t.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

fn grab_str(s: &str, key: &str) -> Option<String> {
    let r = after_key(s, key)?;
    let t = r.trim_start();
    if t.starts_with('"') {
        let e = t[1..].find('"')? + 1;
        Some(t[1..e].to_string())
    } else {
        None
    }
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
            q: crate::math::quat(qq[0], qq[1], qq[2], qq[3]),
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
        let (base, apex) = sp.reference_signature();
        j.push_str(&format!(
            "    {{\"body_id\":{},\"name\":\"{}\",\"frame\":\"{}\",\"body_radius_m\":{:.3},\"apex_marker_id\":{},\"base_marker_ids\":[{},{}],\"signature\":[{:.4},{:.4}],\"edge_lengths\":[{}],\"markers\":[",
            sp.id, sp.name, sp.frame, sp.body_radius_m, sp.apex_marker_id,
            sp.base_marker_ids[0], sp.base_marker_ids[1], base, apex,
            sp.edge_lengths().iter().map(|e| format!("{e:.4}")).collect::<Vec<_>>().join(",")
        ));
        for (k, m) in sp.markers.iter().enumerate() {
            if k > 0 {
                j.push(',');
            }
            j.push_str(&format!(
                "{{\"id\":{},\"local\":[{:.4},{:.4},{:.4}],\"mount\":[{:.4},{:.4},{:.4}],\"local_ned_body\":[{:.4},{:.4},{:.4}],\"radius_m\":{:.4}}}",
                m.id,
                m.local.x, m.local.y, m.local.z,
                m.mount_local.x, m.mount_local.y, m.mount_local.z,
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
    if let Some(pos) = text.find("\"markers\"") {
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
        let (base, apex) = sp.reference_signature();
        report.push_str(&format!(
            "  body {} {:<8} frame {:<12} ref(base,apex)=({:.3},{:.3}) m markers={} extras={:?}\n",
            sp.id,
            sp.name,
            sp.frame,
            base,
            apex,
            sp.markers.len(),
            sp.extra_marker_ids()
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

fn usage() {
    println!(
        "mocap-sim\n\
         \n  demo      [--out logs/demo_crazyflie.mavlink]\n\
         \n  probe     [--file logs/demo_crazyflie.mavlink]\n\
         \n  ingest    [--file ...] [--udp 0.0.0.0:14550 --seconds 20] [--fps 30] [--duration s]\n\
         \n  simulate  [--scenario out/scenario.json] [--out out/rig]"
    );
}

// ---------------------------------------------------------------- telemetry

struct Traj {
    keys: Track,
    att: Option<AttTrack>,
    att_src: String,
    pos_src: String,
}

/// Unwrap 32-bit millisecond / microsecond clocks.
struct Clock {
    last: f64,
    wraps: f64,
}
impl Clock {
    fn new() -> Self {
        Clock { last: -1.0, wraps: 0.0 }
    }
    fn unwrap(&mut self, raw: f64, modulo: f64) -> f64 {
        if self.last >= 0.0 && raw + self.wraps * modulo < self.last - modulo * 0.5 {
            self.wraps += 1.0;
        }
        let v = raw + self.wraps * modulo;
        self.last = self.last.max(v);
        v
    }
}

fn build_trajectory(dl: &Dialect, recs: &[mavlink::Record]) -> (Traj, String, String) {
    let mut clk_lpn = Clock::new();
    let mut clk_mocap = Clock::new();
    let mut clk_odom = Clock::new();
    let mut clk_att = Clock::new();

    let mut lpn: Vec<(f64, V3, V3)> = Vec::new();
    let mut mocap: Vec<(f64, V3, Quat)> = Vec::new();
    let mut odom: Vec<(f64, V3, Quat, V3)> = Vec::new();
    let mut att: Vec<(f64, Quat)> = Vec::new();

    for r in recs {
        let Some(m) = dl.by_id(r.msgid) else { continue };
        let dec = m.decode(&r.payload);
        match m.name.as_str() {
            "LOCAL_POSITION_NED" => {
                let t = clk_lpn.unwrap(field(&dec, "time_boot_ms").map(|v| v.f()).unwrap_or(0.0), 4294967296.0) * 1e-3;
                let p = v3(
                    field(&dec, "x").map(|v| v.f()).unwrap_or(0.0),
                    field(&dec, "y").map(|v| v.f()).unwrap_or(0.0),
                    field(&dec, "z").map(|v| v.f()).unwrap_or(0.0),
                );
                let v = v3(
                    field(&dec, "vx").map(|v| v.f()).unwrap_or(0.0),
                    field(&dec, "vy").map(|v| v.f()).unwrap_or(0.0),
                    field(&dec, "vz").map(|v| v.f()).unwrap_or(0.0),
                );
                lpn.push((t, flight::ned_to_enu(p), flight::ned_to_enu(v)));
            }
            "ATT_POS_MOCAP" => {
                let t = clk_mocap.unwrap(field(&dec, "time_usec").map(|v| v.f()).unwrap_or(0.0), 1.8446744e19) * 1e-6;
                let p = v3(
                    field(&dec, "x").map(|v| v.f()).unwrap_or(0.0),
                    field(&dec, "y").map(|v| v.f()).unwrap_or(0.0),
                    field(&dec, "z").map(|v| v.f()).unwrap_or(0.0),
                );
                let q = quat_from_field(&dec, "q");
                mocap.push((t, flight::ned_to_enu(p), flight::attitude_to_blender(q)));
            }
            "ODOMETRY" => {
                let t = clk_odom.unwrap(field(&dec, "time_usec").map(|v| v.f()).unwrap_or(0.0), 1.8446744e19) * 1e-6;
                let p = v3(
                    field(&dec, "x").map(|v| v.f()).unwrap_or(0.0),
                    field(&dec, "y").map(|v| v.f()).unwrap_or(0.0),
                    field(&dec, "z").map(|v| v.f()).unwrap_or(0.0),
                );
                let v = v3(
                    field(&dec, "vx").map(|v| v.f()).unwrap_or(0.0),
                    field(&dec, "vy").map(|v| v.f()).unwrap_or(0.0),
                    field(&dec, "vz").map(|v| v.f()).unwrap_or(0.0),
                );
                let q = quat_from_field(&dec, "q");
                odom.push((t, flight::ned_to_enu(p), flight::attitude_to_blender(q), flight::ned_to_enu(v)));
            }
            "ATTITUDE" => {
                let t = clk_att.unwrap(field(&dec, "time_boot_ms").map(|v| v.f()).unwrap_or(0.0), 4294967296.0) * 1e-3;
                let roll = field(&dec, "roll").map(|v| v.f()).unwrap_or(0.0) as f64;
                let pitch = field(&dec, "pitch").map(|v| v.f()).unwrap_or(0.0) as f64;
                let yaw = field(&dec, "yaw").map(|v| v.f()).unwrap_or(0.0) as f64;
                let q = Quat::from_rpy(roll, pitch, yaw);
                att.push((t, flight::attitude_to_blender(q)));
            }
            _ => {}
        }
    }

    let (mut pos_src, keys) = if !lpn.is_empty() {
        let mut v = lpn.clone();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        ("LOCAL_POSITION_NED", Track::from_pos_vel(&v))
    } else if !mocap.is_empty() {
        let mut v = mocap.clone();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let pts: Vec<(f64, V3)> = v.iter().map(|(t, p, _)| (*t, *p)).collect();
        ("ATT_POS_MOCAP", Track::from_positions(&pts))
    } else if !odom.is_empty() {
        let mut v = odom.clone();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let pts: Vec<(f64, V3, V3)> = v.iter().map(|(t, p, _, vel)| (*t, *p, *vel)).collect();
        ("ODOMETRY", Track::from_pos_vel(&pts))
    } else {
        ("none", Track::default())
    };

    // Normalise the clock so the trajectory starts at t = 0.
    let t0 = keys.t_start();
    let keys = Track {
        keys: keys
            .keys
            .iter()
            .map(|k| flight::Key { t: k.t - t0, ..*k })
            .collect(),
    };

    let mut att_keys: Vec<(f64, Quat)> = if !att.is_empty() {
        att.iter().map(|(t, q)| (*t - t0, *q)).collect()
    } else if !mocap.is_empty() {
        mocap.iter().map(|(t, _, q)| (*t - t0, *q)).collect()
    } else if !odom.is_empty() {
        odom.iter().map(|(t, _, q, _)| (*t - t0, *q)).collect()
    } else {
        Vec::new()
    };
    att_keys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let att_track = if att_keys.is_empty() {
        None
    } else {
        Some(AttTrack::sorted(att_keys))
    };
    let pos_src = pos_src.to_string();
    let att_src = if att.is_empty() {
        if mocap.is_empty() { "derived" } else { "ATT_POS_MOCAP" }
    } else {
        "ATTITUDE"
    };
    let traj = Traj {
        keys,
        att: att_track,
        att_src: att_src.to_string(),
        pos_src: pos_src.clone(),
    };
    (traj, att_src.to_string(), pos_src)
}

fn quat_from_field(dec: &[(String, Value)], name: &str) -> Quat {
    let a = field(dec, name).map(|v| v.arr()).unwrap_or_default();
    if a.len() >= 4 {
        // MAVLink convention: w, x, y, z
        math::quat(a[0], a[1], a[2], a[3])
    } else {
        Quat::default()
    }
}

fn capture_udp(bind: &str, seconds: f64, out_log: &str) -> Vec<u8> {
    use std::net::UdpSocket;
    let addr = if bind.is_empty() { "0.0.0.0:14550" } else { bind };
    let sock = UdpSocket::bind(addr).expect("cannot bind UDP");
    sock.set_read_timeout(Some(std::time::Duration::from_millis(500))).ok();
    println!("listening on {addr} for {seconds:.1}s (MAVLink UDP)...");
    let start = std::time::Instant::now();
    let mut buf = vec![0u8; 65535];
    let mut all: Vec<u8> = Vec::new();
    while start.elapsed().as_secs_f64() < seconds {
        match sock.recv(&mut buf) {
            Ok(n) => all.extend_from_slice(&buf[..n]),
            Err(_) => {}
        }
    }
    let _ = std::fs::create_dir_all("logs");
    std::fs::write(out_log, &all).ok();
    println!("captured {} bytes -> {out_log}", all.len());
    all
}

// ------------------------------------------------------------------ writers

fn write_scenario(path: &str, data: &[Sample], fps: f64, src: &str, st: &Stats, att: &str, pos: &str) {
    let mut j = J::new();
    j.raw("{\n");
    j.key("meta");
    j.raw("{");
    j.kv_str("generator", "mocap-sim 0.1.0");
    j.raw(",");
    j.kv_str("source", src);
    j.raw(",");
    j.kv_str("units", "metres");
    j.raw(",");
    j.kv_str("world", "ENU, Z-up (Blender)");
    j.raw(",");
    j.kv_num("fps", fps);
    j.raw(",");
    j.kv_num("frame_count", data.len() as f64);
    j.raw(",");
    j.kv_num("duration_s", data.len() as f64 / fps);
    j.raw(",");
    j.kv_str("position_source", pos);
    j.raw(",");
    j.kv_str("attitude_source", att);
    j.raw(",");
    j.kv_num("mavlink_frames", st.frames as f64);
    j.raw(",");
    j.key("messages");
    j.raw("[");
    let mut per = st.per_msg.clone();
    per.sort_by(|a, b| b.1.cmp(&a.1));
    for (i, (n, c)) in per.iter().enumerate() {
        if i > 0 {
            j.raw(",");
        }
        j.raw("[");
        j.str(n);
        j.raw(",");
        j.num(*c as f64);
        j.raw("]");
    }
    j.raw("]");
    j.raw("},\n");

    j.key("samples");
    j.raw("[\n");
    for (i, s) in data.iter().enumerate() {
        if i > 0 {
            j.raw(",\n");
        }
        j.raw("{");
        j.kv_num("t", s.t);
        j.raw(",");
        let p = s.p.arr();
        j.kv_arr3("p", p);
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
    j.raw("\n]\n}\n");
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, j.finish()).expect("write scenario");
}

fn read_scenario(path: &str) -> Vec<Sample> {
    // The scenario is produced by this tool, so a purpose-built reader is
    // enough: collect each `{...}` sample object.
    let src = std::fs::read_to_string(path).expect("read scenario.json");
    let start = src.find("\"samples\"").expect("no samples in scenario");
    let body = &src[start..];
    let mut out = Vec::new();
    for chunk in body.split("{").skip(1) {
        let end = match chunk.find('}') {
            Some(e) => e,
            None => continue,
        };
        let obj = &chunk[..end];
        if !obj.contains("\"p\"") {
            continue;
        }
        let f = |k: &str, idx: usize| -> f64 {
            let pat = format!("\"{k}\":[");
            if let Some(p) = obj.find(&pat) {
                let rest = &obj[p + pat.len()..];
                let arr_end = rest.find(']').unwrap_or(rest.len());
                let nums: Vec<f64> = rest[..arr_end]
                    .split(',')
                    .filter_map(|x| x.trim().parse::<f64>().ok())
                    .collect();
                nums.get(idx).copied().unwrap_or(0.0)
            } else {
                let pat1 = format!("\"{k}\":");
                obj.find(&pat1)
                    .and_then(|p| {
                        let rest = &obj[p + pat1.len()..];
                        let seg: String = rest
                            .chars()
                            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == '+')
                            .collect();
                        seg.parse::<f64>().ok()
                    })
                    .unwrap_or(0.0)
            }
        };
        let q = [
            f("q", 0),
            f("q", 1),
            f("q", 2),
            f("q", 3),
        ];
        out.push(Sample {
            t: f("t", 0),
            p: v3(f("p", 0), f("p", 1), f("p", 2)),
            v: v3(f("v", 0), f("v", 1), f("v", 2)),
            q: math::quat(q[0], q[1], q[2], q[3]),
            w: v3(f("w", 0), f("w", 1), f("w", 2)),
            on_ground: obj.contains("\"on_ground\":true"),
        });
    }
    out
}

fn write_trajectory_csv(path: &str, data: &[Sample]) {
    let mut s = String::from("t,frame,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,on_ground\n");
    for (i, x) in data.iter().enumerate() {
        s.push_str(&format!(
            "{:.4},{},{:.5},{:.5},{:.5},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4},{}\n",
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
    if let Some(p) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(p).ok();
    }
    std::fs::write(path, s).expect("write trajectory.csv");
}

fn write_rig_json(dir: &str, cams: &[Camera], markers: &[Marker], data: &[Sample]) {
    let mut j = J::new();
    j.raw("{\n");
    j.kv_str("units", "metres");
    j.raw(",");
    j.kv_str("world", "ENU, Z-up (Blender)");
    j.raw(",");
    j.key("resolution");
    j.raw(&format!("[{},{}]", rig::CANON_W, rig::CANON_H));
    j.raw(",\n");
    j.key("sensor");
    j.raw(&format!(
        "{{\"model\":\"{}\",\"shutter\":\"global\",\"color\":\"mono\",\"pixel_pitch_um\":{},\"active_px\":[{},{}],\"lens\":{{\"model\":\"M27280M07S\",\"mount\":\"M12\",\"focal_mm\":{},\"fnumber\":{},\"quoted_hfov_deg\":75}}}}",
        rig::SENSOR_MODEL,
        rig::PIXEL_PITCH_UM,
        rig::CANON_W,
        rig::CANON_H,
        rig::LENS_FOCAL_MM,
        rig::LENS_FNUMBER
    ));
    j.raw(",\n");
    j.kv_num("capture_fps", 120.0);
    j.raw(",");
    j.kv_num("render_fps", 30.0);
    j.raw(",\n");
    j.key("volume");
    j.raw(&format!(
        "{{\"center\":[{},{},{}],\"size\":[{},{},{}]}}",
        rig::VOLUME_CENTER.x, rig::VOLUME_CENTER.y, rig::VOLUME_CENTER.z,
        rig::VOLUME_SIZE.x, rig::VOLUME_SIZE.y, rig::VOLUME_SIZE.z
    ));
    j.raw(",\n");

    j.key("cameras");
    j.raw("[\n");
    for (i, c) in cams.iter().enumerate() {
        if i > 0 {
            j.raw(",\n");
        }
        let b = c.blender_rotation();
        let q = flight::m3_to_quat(b);
        let cv = c.cv_rotation();
        let tcv = math::m3_mul_v(cv, c.pos).mul(-1.0);
        j.raw("{");
        j.kv_num("id", c.id as f64);
        j.raw(",");
        j.kv_str("name", &c.name);
        j.raw(",");
        j.kv_str("band", c.band);
        j.raw(",");
        j.kv_str("mount", "ring");
        j.raw(",");
        j.kv_arr3("pos", c.pos.arr());
        j.raw(",");
        j.kv_arr3("target", c.target.arr());
        j.raw(",");
        j.key("quat");
        j.arr4([q.w, q.x, q.y, q.z]);
        j.raw(",");
        j.kv_num("lens_mm", c.blender_lens_mm());
        j.raw(",");
        j.kv_num("sensor_width_mm", c.sensor_width_mm());
        j.raw(",");
        j.kv_num("sensor_height_mm", c.sensor_height_mm());
        j.raw(",");
        j.kv_num("hfov_deg", c.hfov_deg);
        j.raw(",");
        j.kv_num("mount_height_m", c.mount_height);
        j.raw(",");
        j.key("intrinsics");
        j.raw(&format!(
            "{{\"width\":{},\"height\":{},\"fx\":{:.6},\"fy\":{:.6},\"cx\":{:.6},\"cy\":{:.6},\"k1\":{},\"k2\":{},\"p1\":0,\"p2\":0,\"model\":\"pinhole\"}}",
            rig::CANON_W,
            rig::CANON_H,
            c.fx,
            c.fy,
            c.cx,
            c.cy,
            c.k1,
            c.k2
        ));
        j.raw(",");
        j.key("extrinsics_cv");
        j.raw("{");
        j.key("R");
        j.raw("[");
        for (r, row) in cv.iter().enumerate() {
            if r > 0 {
                j.raw(",");
            }
            j.arr3(*row);
        }
        j.raw("],");
        j.key("t");
        j.arr3(tcv.arr());
        j.raw("}");
        j.raw("}");
    }
    j.raw("\n],\n");

    j.key("markers");
    j.raw("[");
    for (i, m) in markers.iter().enumerate() {
        if i > 0 {
            j.raw(",");
        }
        j.raw("{");
        j.kv_num("id", m.id as f64);
        j.raw(",");
        j.kv_arr3("local", m.local.arr());
        j.raw(",");
        j.kv_arr3("mount", m.mount_local.arr());
        j.raw(",");
        j.kv_arr3("local_ned_body", rig::marker_in_ned_body(m).arr());
        j.raw(",");
        j.kv_num("radius_m", m.radius_m);
        j.raw("}");
    }
    j.raw("],\n");
    j.key("frames");
    j.raw(&format!("{}\n", data.len()));
    j.raw("}\n");

    let path = format!("{dir}/rig_blender.json");
    std::fs::write(&path, j.finish()).expect("write rig json");

    // Camera-only JSON for other tooling (no Blender specifics).
    let mut k = J::new();
    k.raw("{\"cameras\":[");
    for (i, c) in cams.iter().enumerate() {
        if i > 0 {
            k.raw(",");
        }
        k.raw("{");
        k.kv_num("id", c.id as f64);
        k.raw(",");
        k.kv_str("name", &c.name);
        k.raw(",");
        k.kv_arr3("pos_enu", c.pos.arr());
        k.raw(",");
        k.kv_arr3("fwd", c.fwd.arr());
        k.raw(",");
        k.kv_arr3("up", c.up.arr());
        k.raw(",");
        k.kv_arr3("right", c.right.arr());
        k.raw("}");
    }
    k.raw("]}");
    std::fs::write(format!("{dir}/cameras_pose.json"), k.finish()).expect("write cameras pose");

    write_calib(dir, cams, data.len());
}

/// Per-camera calibration in the layout OpenCV / most trackers expect:
/// K (3x3), dist [k1,k2,p1,p2,k3], and R,t mapping world (ENU) into the
/// camera frame (x right, y down, z forward).
fn write_calib(dir: &str, cams: &[Camera], frames: usize) {
    let fps = 30.0;
    let mut j = J::new();
    j.raw("{\n");
    j.kv_str("units", "metres");
    j.raw(",\n");
    j.kv_str("world", "ENU, Z-up (Blender)");
    j.raw(",\n");
    j.key("sensor");
    j.raw(&format!(
        "{{\"model\":\"{}\",\"shutter\":\"global\",\"color\":\"mono\",\"width\":{},\"height\":{},\"pixel_pitch_um\":{},\"lens\":{{\"model\":\"M27280M07S\",\"mount\":\"M12\",\"focal_mm\":{},\"fnumber\":{}}},\"distortion\":\"none (ideal pinhole; real lens barrel distortion is not modelled)\"}}",
        rig::SENSOR_MODEL,
        rig::CANON_W,
        rig::CANON_H,
        rig::PIXEL_PITCH_UM,
        rig::LENS_FOCAL_MM,
        rig::LENS_FNUMBER
    ));
    j.raw(",\n");
    j.key("time");
    j.raw(&format!(
        "{{\"fps\":{fps},\"frame_count\":{frames},\"first_frame\":0,\"t\":\"frame/fps\",\"shutter\":\"global, simultaneous on all cameras\",\"note\":\"image f####.png, detections_2d.csv and markers_3d.csv all use the same 0-based frame index\"}}"
    ));
    j.raw(",\n");
    j.key("sync");
    j.raw(&format!(
        "{{\"mode\":\"one Blender scene, shared body trajectories, six cameras rendered at the same frame index\",\"cameras\":[{}],\"frames\":[0,{}],\"max_skew_s\":0}}",
        cams
            .iter()
            .map(|c| format!("\"{}\"", c.name))
            .collect::<Vec<_>>()
            .join(","),
        frames.saturating_sub(1)
    ));
    j.raw(",\n");
    j.key("cameras");
    j.raw("[\n");
    for (i, c) in cams.iter().enumerate() {
        if i > 0 {
            j.raw(",\n");
        }
        let cv = c.cv_rotation();
        let tcv = math::m3_mul_v(cv, c.pos).mul(-1.0);
        j.raw("{");
        j.kv_str("name", &c.name);
        j.raw(",");
        j.kv_num("fx", c.fx);
        j.raw(",");
        j.kv_num("fy", c.fy);
        j.raw(",");
        j.kv_num("cx", c.cx);
        j.raw(",");
        j.kv_num("cy", c.cy);
        j.raw(",");
        j.key("K");
        j.raw(&format!(
            "[[{:.6},0,{:.6}],[0,{:.6},{:.6}],[0,0,1]]",
            c.fx, c.cx, c.fy, c.cy
        ));
        j.raw(",");
        j.key("dist");
        j.raw(&format!("[{},{},0,0,0]", c.k1, c.k2));
        j.raw(",");
        j.key("R");
        j.raw("[");
        for (r, row) in cv.iter().enumerate() {
            if r > 0 {
                j.raw(",");
            }
            j.arr3(*row);
        }
        j.raw("],");
        j.key("t");
        j.arr3(tcv.arr());
        j.raw(",");
        j.kv_arr3("pos_enu", c.pos.arr());
        j.raw("}");
    }
    j.raw("\n]\n}\n");
    std::fs::write(format!("{dir}/cameras_calib.json"), j.finish())
        .expect("write camera calibration");
}

fn write_detections(dir: &str, data: &[Sample], cams: &[Camera], markers: &[Marker]) -> usize {
    let cfg = rig::VisionConfig::default();
    let mut rng = Rng::new(cfg.seed);
    let mut det_csv =
        String::from("frame,t,cam,cam_name,marker,u_px,v_px,radius_px,intensity\n");
    let mut m3d_csv = String::from("frame,t,marker,x,y,z,visible_cams\n");

    let mut per_cam = vec![0usize; cams.len()];
    let mut per_marker = vec![0usize; markers.len()];
    let mut n_det = 0usize;
    let mut reproj_sum = 0.0;
    let mut reproj_n = 0usize;
    let mut pos_err_sum = 0.0;
    let mut rot_err_sum = 0.0;
    let mut pose_n = 0usize;
    let mut missing_marker_frames = 0usize;
    let mut frames_with_lt3 = 0usize;

    
    for (fi, s) in data.iter().enumerate() {
        let obs = rig::observe(fi, s.t, s.p, s.q, cams, markers, &cfg, &mut rng);
        for d in &obs.detections {
            det_csv.push_str(&format!(
                "{},{:.4},{},{},{},{:.3},{:.3},{:.2},{:.3}\n",
                fi,
                s.t,
                d.cam,
                cams[d.cam].name,
                d.marker,
                d.u,
                d.v,
                d.radius_px,
                d.intensity
            ));
            per_cam[d.cam] += 1;
            per_marker[d.marker] += 1;
            n_det += 1;
        }

        let mut vis_counts = vec![0usize; markers.len()];
        for d in &obs.detections {
            vis_counts[d.marker] += 1;
        }
        if vis_counts.iter().any(|&c| c < 2) {
            missing_marker_frames += 1;
        }
        if vis_counts.iter().filter(|&&c| c >= 2).count() < 3 {
            frames_with_lt3 += 1;
        }
        for (mi, p) in obs.world.iter().enumerate() {
            m3d_csv.push_str(&format!(
                "{},{:.4},{},{:.5},{:.5},{:.5},{}\n",
                fi, s.t, mi, p.x, p.y, p.z, vis_counts[mi]
            ));
        }

        // Triangulate every marker with >= 2 views and compare with truth.
        let mut tri: Vec<Option<V3>> = Vec::with_capacity(markers.len());
        for mi in 0..markers.len() {
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
        for mi in 0..markers.len() {
            if let Some(p) = tri[mi] {
                fitted.push(p);
                truth.push(obs.world[mi]);
            }
        }
        if fitted.len() >= 3 {
            // Full 6DOF rigid-body solve: fit the body-local marker layout onto
            // the triangulated world points, then compare pose with truth.
            let locals: Vec<V3> = markers.iter().map(|m| m.local).collect();
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

    std::fs::write(format!("{dir}/detections_2d.csv"), det_csv).expect("write detections");
    std::fs::write(format!("{dir}/markers_3d.csv"), m3d_csv).expect("write markers");

    let total_frames = data.len();
    let mut report = String::new();
    report.push_str("mocap-sim consistency report\n===========================\n");
    report.push_str(&format!("frames            : {total_frames}\n"));
    report.push_str(&format!("cameras           : {}\n", cams.len()));
    report.push_str(&format!("markers per body  : {}\n", markers.len()));
    report.push_str(&format!("detections        : {n_det}\n"));
    report.push_str(&format!(
        "detections/frame  : {:.2}\n",
        n_det as f64 / total_frames.max(1) as f64
    ));
    report.push_str(&format!(
        "triangulation rms : {:.4} px (mean over {} marker-frames)\n",
        reproj_sum / reproj_n.max(1) as f64,
        reproj_n
    ));
    report.push_str(&format!(
        "6DOF position err : {:.2} mm mean\n",
        1000.0 * pos_err_sum / pose_n.max(1) as f64
    ));
    report.push_str(&format!(
        "6DOF rotation err : {:.3} deg mean\n",
        rot_err_sum / pose_n.max(1) as f64
    ));
    report.push_str(&format!(
        "frames w/ <2 views for some marker: {missing_marker_frames}\n"
    ));
    report.push_str(&format!(
        "frames w/ <3 solvable markers      : {frames_with_lt3}\n"
    ));
    report.push_str("\nper camera detections:\n");
    for (i, c) in cams.iter().enumerate() {
        report.push_str(&format!(
            "  {:<6} {:>8.2}% of frames have this marker set visible? {}\n",
            c.name,
            100.0 * per_cam[i] as f64 / (total_frames as f64 * markers.len() as f64),
            per_cam[i]
        ));
    }
    report.push_str("\nper marker detections:\n");
    for (i, m) in markers.iter().enumerate() {
        report.push_str(&format!(
            "  marker {}  {} ({:.1}%)\n",
            m.id,
            per_marker[i],
            100.0 * per_marker[i] as f64 / (total_frames as f64 * cams.len() as f64)
        ));
    }
    std::fs::write(format!("{dir}/verify.txt"), &report).expect("write verify");
    print!("{report}");
    n_det
}
