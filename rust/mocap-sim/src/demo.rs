//! Synthetic Crazyflie flight + MAVLink 2 telemetry log generator.
//!
//! The log is a realistic mixed-rate stream (heartbeat, attitude, local
//! position, odometry, high-res IMU, mocap pose, offboard setpoints) so the
//! same ingest path that consumes a real `.tlog` from PX4 / Crazyflie can be
//! exercised without hardware.

use crate::dialect::{Dialect, Value};
use crate::flight::{self, Track};
use crate::math::{v3, Quat, V3};
use crate::mavlink::encode_v2;

pub struct DemoFlight {
    pub track: Track,
    pub duration: f64,
}

/// Figure-eight demo: 1.5 s vertical take-off, 17 s of a clean Lissajous "8"
/// (x = A sin(wt), y = B sin(2wt)) with altitude modulation, then 1.5 s of
/// landing. 20 s at 30 fps = 600 frames per camera. The path is analytic, so
/// the velocity written into the MAVLink stream is exact rather than
/// differentiated.
pub const T_RAMP: f64 = 1.5;
pub const T_FIG: f64 = 17.0;
pub const T_LAND: f64 = 1.5;
pub const A_X: f64 = 1.90;
pub const B_Y: f64 = 1.00;
/// 17 s / 4.25 s per Lissajous period = exactly four figure-eights.
pub const OMEGA: f64 = std::f64::consts::TAU / 4.25;

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

pub fn demo_pos(t: f64) -> V3 {
    let pad = v3(0.0, -0.70, 0.020);
    let center = v3(0.0, 0.0, 1.35);

    let up = smoothstep(t / T_RAMP);
    let down = 1.0 - smoothstep((t - (T_RAMP + T_FIG)) / T_LAND);
    let h = up.min(down);

    let ramp_in = smoothstep((t - T_RAMP * 0.30) / (T_RAMP * 0.95));
    let ramp_out = 1.0 - smoothstep((t - (T_RAMP + T_FIG - 1.15)) / 1.15);
    let env = ramp_in.min(ramp_out);

    let tau = t - T_RAMP;
    let x = A_X * (OMEGA * tau).sin();
    let y = B_Y * (2.0 * OMEGA * tau).sin();
    let z = 0.26 * (2.0 * OMEGA * tau).sin();

    let base = pad.lerp(center, h);
    base.add(v3(env * x, env * y, env * z))
}

pub fn build() -> DemoFlight {
    let total = T_RAMP + T_FIG + T_LAND;
    let hz = 200.0;
    let n = (total * hz).round() as usize;
    let dt = 1.0 / hz;
    let mut keys: Vec<(f64, V3, V3)> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = i as f64 * dt;
        let p = demo_pos(t);
        let v = demo_pos(t + 0.0005).sub(demo_pos(t - 0.0005)).mul(1000.0);
        keys.push((t, p, v));
    }
    // The smoothstep landing profile already brings the position *and* the
    // velocity to zero at t = total, so no extra tail damping is needed.
    DemoFlight {
        track: Track::from_pos_vel(&keys),
        duration: total,
    }
}

// ------------------------------------------------------------------ specs
//
// A FlightSpec turns the analytic figure-eight into a parameterised flight:
// pad / cruise centre / amplitude scale / Lissajous phase.  The original
// Crazyflie demo is exactly spec #0, so legacy behaviour is preserved bit for
// bit while multi-body scenes become plain data.

#[derive(Clone, Copy, Debug)]
pub struct FlightSpec {
    /// Take-off / landing point (on the ground, z ~ 0.02).
    pub pad: V3,
    /// Cruise volume centre for this body.
    pub center: V3,
    /// Amplitude scale applied to the Lissajous and its altitude bob.
    pub scale: f64,
    /// Phase offset (rad) applied to both the x- and the y-Lissajous.
    pub phase: f64,
    /// Frequency multiplier; 0.5 gives a slower 8.5 s figure-eight.
    pub frequency_scale: f64,
}

impl Default for FlightSpec {
    fn default() -> Self {
        FlightSpec { pad: v3(0.0, -0.70, 0.020), center: v3(0.0, 0.0, 1.35), scale: 1.0, phase: 0.0, frequency_scale: 1.0 }
    }
}

/// The original single-Crazyflie flight, expressed as a spec.
pub fn default_spec() -> FlightSpec {
    FlightSpec::default()
}

/// Three Tarot Ironman 650 flights: same 20 s choreography, staggered in
/// phase and altitude so the bodies never collide while still crossing the
/// capture volume from opposite sides.
pub fn tarot650_specs() -> Vec<FlightSpec> {
    vec![
        FlightSpec { pad: v3(0.0, -0.70, 0.020), center: v3(0.0, 0.0, 1.35), scale: 1.00, phase: 0.0, frequency_scale: 1.0 },
        FlightSpec { pad: v3(-1.50, 0.60, 0.020), center: v3(0.0, 0.0, 1.98), scale: 0.75, phase: 2.1, frequency_scale: 1.0 },
        FlightSpec { pad: v3(1.50, 0.60, 0.020), center: v3(0.0, 0.0, 0.74), scale: 0.75, phase: 4.2, frequency_scale: 1.0 },
    ]
}

pub fn spec_pos(t: f64, sp: &FlightSpec) -> V3 {
    let up = smoothstep(t / T_RAMP);
    let down = 1.0 - smoothstep((t - (T_RAMP + T_FIG)) / T_LAND);
    let h = up.min(down);

    let ramp_in = smoothstep((t - T_RAMP * 0.30) / (T_RAMP * 0.95));
    let ramp_out = 1.0 - smoothstep((t - (T_RAMP + T_FIG - 1.15)) / 1.15);
    let env = ramp_in.min(ramp_out);

    let tau = t - T_RAMP;
    let ph = OMEGA * sp.frequency_scale * tau + sp.phase;
    let x = A_X * sp.scale * ph.sin();
    let y = B_Y * sp.scale * (2.0 * ph).sin();
    let z = 0.26 * sp.scale * (2.0 * ph).sin();

    sp.pad.lerp(sp.center, h).add(v3(env * x, env * y, env * z))
}

pub fn build_spec(sp: &FlightSpec) -> DemoFlight {
    let total = T_RAMP + T_FIG + T_LAND;
    let hz = 200.0;
    let n = (total * hz).round() as usize;
    let dt = 1.0 / hz;
    let mut keys: Vec<(f64, V3, V3)> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = i as f64 * dt;
        let p = spec_pos(t, sp);
        let v = spec_pos(t + 0.0005, sp).sub(spec_pos(t - 0.0005, sp)).mul(1000.0);
        keys.push((t, p, v));
    }
    DemoFlight {
        track: Track::from_pos_vel(&keys),
        duration: total,
    }
}

/// Build every body of a multi-target scene.
pub fn build_multi(specs: &[FlightSpec]) -> Vec<DemoFlight> {
    specs.iter().map(build_spec).collect()
}

fn n(v: f64) -> Value {
    Value::Num(vec![v])
}
fn arr(v: &[f64]) -> Value {
    Value::Num(v.to_vec())
}

/// Body attitude in mesh-Blender space from the trajectory dynamics.
pub fn mesh_attitude(track: &Track, t: f64) -> Quat {
    let (_, v) = track.eval(t);
    let (_, v2) = track.eval(t + 0.004);
    let acc = v2.sub(v).mul(1.0 / 0.004);
    flight::attitude_from_dynamics(v, acc)
}

struct BodyEmitter<'a> {
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

/// Convert a mesh-space orientation back to a FRD/NED attitude quaternion.
pub fn mesh_to_frd(q_mesh: Quat) -> Quat {
    flight::blender_to_attitude_frd(q_mesh)
}

/// Human-readable summary of the flight profile.
pub fn summary(fl: &DemoFlight) -> String {
    let mut vmax: f64 = 0.0;
    let mut tmax = 0.0;
    let mut amax: f64 = 0.0;
    let mut amax_t = 0.0;
    let steps = (fl.duration * 200.0) as usize;
    for i in 0..steps {
        let t = i as f64 / 200.0;
        let (_, v) = fl.track.eval(t);
        let (_, v2) = fl.track.eval(t + 0.005);
        let a = v2.sub(v).mul(1.0 / 0.005).norm();
        if v.norm() > vmax {
            vmax = v.norm();
            tmax = t;
        }
        if a > amax {
            amax = a;
            amax_t = t;
        }
    }
    format!(
        "duration {:.2} s, peak speed {:.2} m/s @ {:.2}s, peak accel {:.2} m/s^2 @ {:.2}s",
        fl.duration, vmax, tmax, amax, amax_t
    )
}

// convenience for callers that only have the ENU sample position
pub fn enu_to_ned_public(v: V3) -> V3 {
    flight::enu_to_ned(v)
}
