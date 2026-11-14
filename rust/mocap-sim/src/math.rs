//! Small linear-algebra kit: vectors, quaternions, 3x3/4x4 helpers, solving.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub const fn v3(x: f64, y: f64, z: f64) -> V3 {
    V3 { x, y, z }
}

impl V3 {
    pub fn add(self, o: V3) -> V3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn sub(self, o: V3) -> V3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    pub fn mul(self, s: f64) -> V3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
    pub fn dot(self, o: V3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: V3) -> V3 {
        v3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn unit(self) -> V3 {
        let n = self.norm();
        if n > 1e-12 {
            self.mul(1.0 / n)
        } else {
            v3(0.0, 0.0, 0.0)
        }
    }
    pub fn lerp(self, o: V3, t: f64) -> V3 {
        self.mul(1.0 - t).add(o.mul(t))
    }
    pub fn arr(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for Quat {
    fn default() -> Self {
        Quat {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

pub fn quat(w: f64, x: f64, y: f64, z: f64) -> Quat {
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if n < 1e-12 {
        Quat::default()
    } else {
        Quat {
            w: w / n,
            x: x / n,
            y: y / n,
            z: z / n,
        }
    }
}

impl Quat {
    pub fn mul(self, o: Quat) -> Quat {
        Quat {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }
    pub fn conj(self) -> Quat {
        Quat {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
    pub fn rotate(self, v: V3) -> V3 {
        let u = v3(self.x, self.y, self.z);
        let t = u.cross(v).mul(2.0);
        v.add(t.mul(self.w)).add(u.cross(t))
    }
    pub fn slerp(self, mut o: Quat, t: f64) -> Quat {
        let mut d = self.w * o.w + self.x * o.x + self.y * o.y + self.z * o.z;
        if d < 0.0 {
            o = Quat {
                w: -o.w,
                x: -o.x,
                y: -o.y,
                z: -o.z,
            };
            d = -d;
        }
        if d > 0.9995 {
            return quat(
                self.w + (o.w - self.w) * t,
                self.x + (o.x - self.x) * t,
                self.y + (o.y - self.y) * t,
                self.z + (o.z - self.z) * t,
            );
        }
        let theta = d.clamp(-1.0, 1.0).acos();
        let s = theta.sin();
        let a = ((1.0 - t) * theta).sin() / s;
        let b = (t * theta).sin() / s;
        quat(
            self.w * a + o.w * b,
            self.x * a + o.x * b,
            self.y * a + o.y * b,
            self.z * a + o.z * b,
        )
    }
    pub fn from_rpy(roll: f64, pitch: f64, yaw: f64) -> Quat {
        // aerospace Z-Y-X intrinsic (yaw-pitch-roll), body FRD -> NED
        let (sr, cr) = (roll * 0.5).sin_cos();
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sy, cy) = (yaw * 0.5).sin_cos();
        quat(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    }
    pub fn to_rpy(self) -> (f64, f64, f64) {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        (roll, pitch, yaw)
    }
}

/// 3x3 matrix, row-major.
pub type M3 = [[f64; 3]; 3];

pub fn m3_mul(a: M3, b: M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

pub fn m3_transpose(a: M3) -> M3 {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

pub fn m3_mul_v(a: M3, v: V3) -> V3 {
    v3(
        a[0][0] * v.x + a[0][1] * v.y + a[0][2] * v.z,
        a[1][0] * v.x + a[1][1] * v.y + a[1][2] * v.z,
        a[2][0] * v.x + a[2][1] * v.y + a[2][2] * v.z,
    )
}

pub fn quat_to_m3(q: Quat) -> M3 {
    let (w, x, y, z) = (q.w, q.x, q.y, q.z);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Solve `A x = b` for a small dense system (Gaussian elimination with
/// partial pivoting).  Returns None if singular.
pub fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        x.swap(col, piv);
        for r in col + 1..n {
            let f = m[r][col] / m[col][col];
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                m[r][c] -= f * m[col][c];
            }
            x[r] -= f * x[col];
        }
    }
    let mut out = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = x[r];
        for c in r + 1..n {
            s -= m[r][c] * out[c];
        }
        out[r] = s / m[r][r];
    }
    Some(out)
}

/// Deterministic PRNG (SplitMix64) so every render reproduces the same noise.
#[derive(Clone, Copy)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0,1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.unit()
    }
    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
