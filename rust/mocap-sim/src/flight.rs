//! Telemetry ingest -> smooth flight trajectory -> per-frame render samples.
//!
//! Coordinate systems
//! ------------------
//! * MAVLink / PX4 / Crazyflie: `NED` world, `FRD` body (x fwd, y right, z down).
//! * Blender: `ENU` world (x east, y north, z up), body authored with
//!   x fwd, y left, z up.
//!
//! Position:    p_enu = (n.e, n.n, -n.d)
//! Orientation: R_enu_body = M * R_ned_body * M^T  with M = 180 deg about (1,1,0)/sqrt(2),
//!              then a final 180 deg roll to convert FRD body axes into the
//!              Blender mesh body axes (x fwd, y left, z up).

use crate::math::{quat, v3, Quat, V3};

/// Rotation taking NED coordinates to ENU coordinates.
pub fn ned_to_enu(v: V3) -> V3 {
    v3(v.y, v.x, -v.z)
}

/// Rotation taking ENU coordinates to NED coordinates (self-inverse).
pub fn enu_to_ned(v: V3) -> V3 {
    v3(v.y, v.x, -v.z)
}

/// Quaternion `q_M` for the NED->ENU basis change.
pub fn q_ned_to_enu() -> Quat {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    quat(0.0, s, s, 0.0)
}

/// Convert an attitude expressed as body-FRD -> NED into the Blender mesh
/// orientation (body: x fwd, y left, z up; world: ENU).
pub fn attitude_to_blender(q_ned: Quat) -> Quat {
    // q_mesh = M * q_frd_ned * X180
    q_ned_to_enu()
        .mul(q_ned)
        .mul(quat(0.0, 1.0, 0.0, 0.0))
}

/// Inverse of [`attitude_to_blender`].
pub fn blender_to_attitude_frd(q_mesh: Quat) -> Quat {
    q_ned_to_enu()
        .conj()
        .mul(q_mesh)
        .mul(quat(0.0, -1.0, 0.0, 0.0))
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub t: f64,
    /// Position, ENU metres (Blender world).
    pub p: V3,
    /// Velocity, ENU m/s.
    pub v: V3,
    /// Blender-space body orientation.
    pub q: Quat,
    /// Blender-space body angular rate, rad/s (approx, for prop animation).
    pub w: V3,
    pub on_ground: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Key {
    pub t: f64,
    pub p: V3,
    pub v: V3,
}

/// Monotone-ish cubic Hermite through keys, falling back to linear when a
/// segment has no velocity information.
#[derive(Clone, Debug, Default)]
pub struct Track {
    pub keys: Vec<Key>,
}

impl Track {
    pub fn from_positions(pts: &[(f64, V3)]) -> Self {
        let mut keys: Vec<Key> = Vec::with_capacity(pts.len());
        for i in 0..pts.len() {
            let (t, p) = pts[i];
            let v = if i == 0 || i + 1 == pts.len() {
                v3(0.0, 0.0, 0.0)
            } else {
                let (t0, p0) = pts[i - 1];
                let (t1, p1) = pts[i + 1];
                if t1 - t0 > 1e-9 {
                    p1.sub(p0).mul(1.0 / (t1 - t0))
                } else {
                    v3(0.0, 0.0, 0.0)
                }
            };
            keys.push(Key { t, p, v });
        }
        Track { keys }
    }

    pub fn from_pos_vel(samples: &[(f64, V3, V3)]) -> Self {
        let mut keys: Vec<Key> = Vec::with_capacity(samples.len());
        for &(t, p, v) in samples {
            // Guard against noisy differentiated positions: clamp absurd speeds.
            let speed = v.norm();
            let v = if speed > 12.0 { v.mul(12.0 / speed) } else { v };
            keys.push(Key { t, p, v });
        }
        keys.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        Track { keys }
    }

    pub fn t_start(&self) -> f64 {
        self.keys.first().map(|k| k.t).unwrap_or(0.0)
    }
    pub fn t_end(&self) -> f64 {
        self.keys.last().map(|k| k.t).unwrap_or(0.0)
    }

    pub fn eval(&self, t: f64) -> (V3, V3) {
        let n = self.keys.len();
        if n == 0 {
            return (v3(0.0, 0.0, 0.0), v3(0.0, 0.0, 0.0));
        }
        if n == 1 || t <= self.keys[0].t {
            return (self.keys[0].p, self.keys[0].v);
        }
        if t >= self.keys[n - 1].t {
            return (self.keys[n - 1].p, self.keys[n - 1].v);
        }
        let mut i = 0;
        while i + 1 < n && self.keys[i + 1].t < t {
            i += 1;
        }
        let a = self.keys[i];
        let b = self.keys[i + 1];
        let h = (b.t - a.t).max(1e-9);
        let s = ((t - a.t) / h).clamp(0.0, 1.0);
        let s2 = s * s;
        let s3 = s2 * s;
        // Hermite basis
        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;
        let p = a
            .p
            .mul(h00)
            .add(a.v.mul(h10 * h))
            .add(b.p.mul(h01))
            .add(b.v.mul(h11 * h));
        let d00 = 6.0 * s2 - 6.0 * s;
        let d10 = 3.0 * s2 - 4.0 * s + 1.0;
        let d01 = -6.0 * s2 + 6.0 * s;
        let d11 = 3.0 * s2 - 2.0 * s;
        let v = a
            .p
            .mul(d00 / h)
            .add(a.v.mul(d10))
            .add(b.p.mul(d01 / h))
            .add(b.v.mul(d11));
        (p, v)
    }
}

/// Attitude track: quaternion keys with slerp.
#[derive(Clone, Debug, Default)]
pub struct AttTrack {
    pub keys: Vec<(f64, Quat)>,
}

impl AttTrack {
    pub fn sorted(mut keys: Vec<(f64, Quat)>) -> Self {
        keys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        AttTrack { keys }
    }
    pub fn eval(&self, t: f64) -> Option<Quat> {
        let n = self.keys.len();
        if n == 0 {
            return None;
        }
        if t <= self.keys[0].0 {
            return Some(self.keys[0].1);
        }
        if t >= self.keys[n - 1].0 {
            return Some(self.keys[n - 1].1);
        }
        let mut i = 0;
        while i + 1 < n && self.keys[i + 1].0 < t {
            i += 1;
        }
        let (t0, q0) = self.keys[i];
        let (t1, q1) = self.keys[i + 1];
        let s = ((t - t0) / (t1 - t0).max(1e-9)).clamp(0.0, 1.0);
        Some(q0.slerp(q1, s))
    }
}

/// Build per-frame samples at `fps` for a `duration` window, optionally
/// synthesising attitude from acceleration when no ATTITUDE stream exists.
pub fn samples_from(
    pos: &Track,
    att: Option<&AttTrack>,
    fps: f64,
    duration: f64,
) -> Vec<Sample> {
    let n = (duration * fps).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(n);
    let t0 = pos.t_start();
    for i in 0..n {
        let t = t0 + i as f64 / fps;
        let (p, v) = pos.eval(t);
        let (p2, v2) = pos.eval(t + 1.0 / 240.0);
        let acc = v2.sub(v).mul(240.0);
        let q = match att.and_then(|a| a.eval(t)) {
            Some(q) => q,
            None => attitude_from_dynamics(v, acc),
        };
        let prev = out.last().map(|s: &Sample| s.q).unwrap_or(q);
        let w = ang_rate(prev, q, 1.0 / fps);
        out.push(Sample {
            t: t - t0,
            p,
            v,
            q,
            w,
            on_ground: p.z < 0.012 && v.norm() < 0.15,
        });
    }
    out
}

/// Yaw follows the velocity vector; tilt follows the thrust vector
/// (a = thrust - g, so the body z axis points along `a + g`).
pub fn attitude_from_dynamics(v: V3, acc: V3) -> Quat {
    let g = v3(0.0, 0.0, 9.81);
    let thrust_dir = acc.add(g).unit();
    if thrust_dir.norm() < 1e-6 {
        return Quat::default();
    }
    // Body z (up) must equal thrust_dir; body x follows the horizontal velocity.
    let zb = thrust_dir;
    let horiz = v3(v.x, v.y, 0.0);
    let xh = if horiz.norm() > 0.25 {
        horiz.unit()
    } else {
        v3(1.0, 0.0, 0.0)
    };
    let yb = zb.cross(xh).unit();
    if yb.norm() < 1e-6 {
        return Quat::default();
    }
    let xb = yb.cross(zb).unit();
    // rotation matrix columns xb, yb, zb -> quaternion
    m3_to_quat([
        [xb.x, yb.x, zb.x],
        [xb.y, yb.y, zb.y],
        [xb.z, yb.z, zb.z],
    ])
}

pub fn m3_to_quat(m: [[f64; 3]; 3]) -> Quat {
    let tr = m[0][0] + m[1][1] + m[2][2];
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        quat(
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        quat(
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        quat(
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        quat(
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        )
    }
}

fn ang_rate(a: Quat, b: Quat, dt: f64) -> V3 {
    let d = b.mul(a.conj());
    let mut s = if d.w < 0.0 { -1.0 } else { 1.0 };
    let angle = 2.0 * d.w.clamp(-1.0, 1.0).acos();
    let sn = (1.0 - d.w * d.w).sqrt();
    if sn < 1e-8 {
        s *= 0.0;
    }
    let axis = if sn < 1e-6 {
        v3(0.0, 0.0, 0.0)
    } else {
        v3(d.x / sn, d.y / sn, d.z / sn)
    };
    axis.mul(angle * s / dt.max(1e-9))
}
