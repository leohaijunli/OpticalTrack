"""Splice the multi-body MAVLink writer into demo.rs (replaces write_log)."""
import io, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "rust", "mocap-sim", "src", "demo.rs")

NEW = r'''struct BodyEmitter<'a> {
    sysid: u8,
    seq: u8,
    counters: [f64; 8],
    fl: &'a DemoFlight,
}

/// Write one telemetry stream shared by every body: each drone is its own
/// MAVLink system (sysid = 1..N) interleaved on the link at the same rates,
/// exactly like a real multi-vehicle mocap bench.  The message mix and the
/// per-message rates are unchanged from the single-body log.
pub fn write_log_multi(
    dl: &Dialect,
    path: &str,
    flights: &[(u8, &DemoFlight)],
) -> std::io::Result<usize> {
    let mut buf: Vec<u8> = Vec::new();
    let boot_ms0 = 812_345.0;
    let us0 = 4_120_000_000.0; // time_usec near the 32-bit ms wrap, wraps once
    let home = (47.397742_f64, 8.545594_f64, 488.0_f64);

    let get = |name: &str| dl.by_name(name).unwrap_or_else(|| panic!("dialect missing {name}"));

    let att = get("ATTITUDE");
    let lpn = get("LOCAL_POSITION_NED");
    let gpi = get("GLOBAL_POSITION_INT");
    let hb = get("HEARTBEAT");
    let hb_imu = get("HIGHRES_IMU");
    let odom = get("ODOMETRY");
    let mocap = get("ATT_POS_MOCAP");
    let sp = get("SET_POSITION_TARGET_LOCAL_NED");

    let mut bodies: Vec<BodyEmitter> = flights
        .iter()
        .map(|(s, f)| BodyEmitter { sysid: *s, seq: 0, counters: [0.0; 8], fl: f })
        .collect();

    let dt = 0.001_f64; // 1 kHz base tick
    let total = flights
        .iter()
        .map(|(_, f)| f.duration)
        .fold(0.0_f64, |a, b| a.max(b));
    let steps = (total / dt).round() as usize;
    let rates = [1.0, 50.0, 50.0, 10.0, 100.0, 50.0, 120.0, 20.0];

    for i in 0..steps {
        let t = i as f64 * dt;
        for b in bodies.iter_mut() {
            if t > b.fl.duration {
                continue;
            }
            let (p_enu, v_enu) = b.fl.track.eval(t);
            let q_mesh = mesh_attitude(&b.fl.track, t);
            // mesh/ENU -> FRD/NED for the wire format
            let q_frd = flight::blender_to_attitude_frd(q_mesh);
            let (roll, pitch, yaw) = q_frd.to_rpy();
            let p_ned = flight::enu_to_ned(p_enu);
            let v_ned = flight::enu_to_ned(v_enu);

            let (_, v2) = b.fl.track.eval(t + 0.002);
            let acc_enu = v2.sub(v_enu).mul(1.0 / 0.002);
            let acc_ned = flight::enu_to_ned(acc_enu);
            let gyro = {
                let q1 = mesh_attitude(&b.fl.track, t - 0.004);
                let q2 = mesh_attitude(&b.fl.track, t + 0.004);
                let d = q2.mul(q1.conj());
                let s = if d.w < 0.0 { -1.0 } else { 1.0 };
                let ang = 2.0 * d.w.clamp(-1.0, 1.0).acos() * s;
                let sn = (1.0 - d.w * d.w).sqrt().max(1e-9);
                let axis = v3(d.x / sn, d.y / sn, d.z / sn);
                let w_enu = axis.mul(ang / 0.008);
                flight::enu_to_ned(w_enu)
            };

            let t_ms = boot_ms0 + t * 1000.0;
            let t_us = us0 + t * 1e6;

            let write = |buf: &mut Vec<u8>, seq: &mut u8, m: &crate::dialect::MessageDef, vals: &[(&str, Value)]| {
                let frame = encode_v2(m, vals, *seq, b.sysid, 1);
                buf.extend_from_slice(&frame);
                *seq = seq.wrapping_add(1);
            };

            // 0 @1Hz heartbeat
            b.counters[0] += rates[0] * dt;
            if b.counters[0] >= 1.0 {
                b.counters[0] -= 1.0;
                write(
                    &mut buf,
                    &mut b.seq,
                    hb,
                    &[
                        ("type", Value::Num(vec![2.0])),      // MAV_TYPE_QUADROTOR
                        ("autopilot", Value::Num(vec![12.0])), // PX4
                        ("base_mode", Value::Num(vec![217.0])),
                        ("custom_mode", Value::Num(vec![0.0])),
                        ("system_status", Value::Num(vec![4.0])),
                        ("mavlink_version", Value::Num(vec![3.0])),
                    ],
                );
            }

            // 1 @50Hz ATTITUDE
            b.counters[1] += rates[1] * dt;
            if b.counters[1] >= 1.0 {
                b.counters[1] -= 1.0;
                write(
                    &mut buf,
                    &mut b.seq,
                    att,
                    &[
                        ("time_boot_ms", n(t_ms)),
                        ("roll", n(roll)),
                        ("pitch", n(pitch)),
                        ("yaw", n(yaw)),
                        ("rollspeed", n(gyro.x)),
                        ("pitchspeed", n(gyro.y)),
                        ("yawspeed", n(gyro.z)),
                    ],
                );
            }

            // 2 @50Hz LOCAL_POSITION_NED
            b.counters[2] += rates[2] * dt;
            if b.counters[2] >= 1.0 {
                b.counters[2] -= 1.0;
                write(
                    &mut buf,
                    &mut b.seq,
                    lpn,
                    &[
                        ("time_boot_ms", n(t_ms)),
                        ("x", n(p_ned.x)),
                        ("y", n(p_ned.y)),
                        ("z", n(p_ned.z)),
                        ("vx", n(v_ned.x)),
                        ("vy", n(v_ned.y)),
                        ("vz", n(v_ned.z)),
                    ],
                );
            }

            // 3 @10Hz GLOBAL_POSITION_INT
            b.counters[3] += rates[3] * dt;
            if b.counters[3] >= 1.0 {
                b.counters[3] -= 1.0;
                let dlat = p_enu.y / 111_320.0;
                let dlon = p_enu.x / (111_320.0 * home.1.to_radians().cos());
                write(
                    &mut buf,
                    &mut b.seq,
                    gpi,
                    &[
                        ("time_boot_ms", n(t_ms)),
                        ("lat", n((home.0 + dlat) * 1e7)),
                        ("lon", n((home.1 + dlon) * 1e7)),
                        ("alt", n((home.2 - p_ned.z) * 1000.0)),
                        ("relative_alt", n(-p_ned.z * 1000.0)),
                        ("vx", n(v_ned.x * 100.0)),
                        ("vy", n(v_ned.y * 100.0)),
                        ("vz", n(v_ned.z * 100.0)),
                        ("hdg", n((yaw.to_degrees()).rem_euclid(360.0) * 100.0)),
                    ],
                );
            }

            // 4 @100Hz HIGHRES_IMU
            b.counters[4] += rates[4] * dt;
            if b.counters[4] >= 1.0 {
                b.counters[4] -= 1.0;
                let thrust = acc_ned.add(v3(0.0, 0.0, 9.81));
                write(
                    &mut buf,
                    &mut b.seq,
                    hb_imu,
                    &[
                        ("time_usec", n(t_us)),
                        ("xacc", n(thrust.x)),
                        ("yacc", n(thrust.y)),
                        ("zacc", n(thrust.z)),
                        ("xgyro", n(gyro.x)),
                        ("ygyro", n(gyro.y)),
                        ("zgyro", n(gyro.z)),
                        ("xmag", n(0.21)),
                        ("ymag", n(0.02)),
                        ("zmag", n(0.44)),
                        ("abs_pressure", n(1013.25)),
                        ("diff_pressure", n(0.02 * v_enu.norm())),
                        ("pressure_alt", n(-p_enu.z)),
                        ("temperature", n(31.5)),
                        ("fields_updated", Value::Num(vec![8191.0])),
                    ],
                );
            }

            // 5 @50Hz ODOMETRY
            b.counters[5] += rates[5] * dt;
            if b.counters[5] >= 1.0 {
                b.counters[5] -= 1.0;
                write(
                    &mut buf,
                    &mut b.seq,
                    odom,
                    &[
                        ("time_usec", n(t_us)),
                        ("frame_id", Value::Num(vec![1.0])), // MAV_FRAME_LOCAL_FRD
                        ("child_frame_id", Value::Num(vec![9.0])),
                        ("x", n(p_ned.x)),
                        ("y", n(p_ned.y)),
                        ("z", n(p_ned.z)),
                        ("q", arr(&[q_frd.w, q_frd.x, q_frd.y, q_frd.z])),
                        ("vx", n(v_ned.x)),
                        ("vy", n(v_ned.y)),
                        ("vz", n(v_ned.z)),
                        ("rollspeed", n(gyro.x)),
                        ("pitchspeed", n(gyro.y)),
                        ("yawspeed", n(gyro.z)),
                        ("covariance", Value::Num(vec![0.0; 21])),
                    ],
                );
            }

            // 6 @120Hz ATT_POS_MOCAP: the tracking system feeding the vehicle
            b.counters[6] += rates[6] * dt;
            if b.counters[6] >= 1.0 {
                b.counters[6] -= 1.0;
                write(
                    &mut buf,
                    &mut b.seq,
                    mocap,
                    &[
                        ("time_usec", n(t_us)),
                        ("x", n(p_ned.x)),
                        ("y", n(p_ned.y)),
                        ("z", n(p_ned.z)),
                        ("q", arr(&[q_frd.w, q_frd.x, q_frd.y, q_frd.z])),
                    ],
                );
            }

            // 7 @20Hz SET_POSITION_TARGET_LOCAL_NED (offboard velocity + yaw)
            b.counters[7] += rates[7] * dt;
            if b.counters[7] >= 1.0 {
                b.counters[7] -= 1.0;
                write(
                    &mut buf,
                    &mut b.seq,
                    sp,
                    &[
                        ("time_boot_ms", n(t_ms)),
                        ("target_system", Value::Num(vec![b.sysid as f64])),
                        ("target_component", Value::Num(vec![1.0])),
                        ("coordinate_frame", Value::Num(vec![1.0])), // LOCAL_NED
                        ("type_mask", Value::Num(vec![0x0FC7 as f64])), // vel + yaw_rate
                        ("x", n(0.0)),
                        ("y", n(0.0)),
                        ("z", n(0.0)),
                        ("vx", n(v_ned.x)),
                        ("vy", n(v_ned.y)),
                        ("vz", n(v_ned.z)),
                        ("afx", n(0.0)),
                        ("afy", n(0.0)),
                        ("afz", n(0.0)),
                        ("yaw", n(yaw)),
                        ("yaw_rate", n(gyro.z)),
                    ],
                );
            }
        }
    }

    std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap_or(std::path::Path::new(".")))?;
    std::fs::write(path, &buf)?;
    Ok(buf.len())
}

/// Single-body convenience wrapper (system id 1).
pub fn write_log(dl: &Dialect, path: &str, fl: &DemoFlight) -> std::io::Result<usize> {
    write_log_multi(dl, path, &[(1, fl)])
}

'''

src = io.open(SRC, encoding="utf-8").read()
start = src.index("pub fn write_log(")
end = src.index("/// Convert a mesh-space orientation", start)
patched = src[:start] + NEW + src[end:]
io.open(SRC, "w", encoding="utf-8", newline="\n").write(patched)
print("spliced write_log_multi:", len(NEW), "chars over", end - start, "chars")
