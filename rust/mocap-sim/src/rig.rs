//! Camera rig, rigid-body marker layout, synthetic 2D detections and a
//! triangulation consistency check.

use crate::flight::ned_to_enu;
use crate::math::{m3_mul_v, m3_transpose, quat_to_m3, solve, v3, Quat, Rng, V3, M3};

// ------------------------------------------------------------------ sensor
// Arducam OV9281 (monochrome, global shutter) behind the stock M27280M07S M12
// lens. The numbers below are the datasheet values, and the intrinsics are
// derived from them instead of being picked by hand:
//
//   active array 1280 x 800, pixel pitch 3.0 um, 1/4"
//   lens EFL 2.8 mm, F/2.8, quoted HFOV 75 deg (that figure includes the
//   barrel distortion of the real lens)
//
// We render a distortion-free pinhole with the same EFL, so
//   fx = fy = 2.8 mm / 3.0 um = 933.33 px
//   1/4" active area = 3.84 x 2.40 mm -> HFOV 68.9 deg, VFOV 46.4 deg
// Keeping k1 = k2 = 0 means the rendered pixels agree exactly with the
// ground-truth projection; the residual corner offset is the same one a
// calibration would absorb into k1.
pub const SENSOR_MODEL: &str = "OV9281";
pub const CANON_W: u32 = 1280;
pub const CANON_H: u32 = 800;
pub const PIXEL_PITCH_UM: f64 = 3.0;
pub const LENS_FOCAL_MM: f64 = 2.8;
pub const LENS_FNUMBER: f64 = 2.8;
pub const SENSOR_W_MM: f64 = CANON_W as f64 * PIXEL_PITCH_UM / 1000.0;
pub const SENSOR_H_MM: f64 = CANON_H as f64 * PIXEL_PITCH_UM / 1000.0;

/// Calibrated capture volume, ENU metres (Blender world).
pub const VOLUME_CENTER: V3 = V3 { x: 0.0, y: 0.0, z: 1.20 };
pub const VOLUME_SIZE: V3 = V3 { x: 4.60, y: 2.80, z: 2.40 };

#[derive(Clone, Debug)]
pub struct Camera {
    pub id: usize,
    pub name: String,
    pub pos: V3,
    pub target: V3,
    pub right: V3,
    pub up: V3,
    pub fwd: V3,
    pub hfov_deg: f64,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// Lens distortion of the synthetic sensor (0 by default so the rendered
    /// pixels agree with the ground-truth projection exactly).
    pub k1: f64,
    pub k2: f64,
    pub band: &'static str,
    pub mount_height: f64,
}

impl Camera {
    pub fn new(id: usize, pos: V3, target: V3) -> Self {
        let fwd = target.sub(pos).unit();
        let up_hint = v3(0.0, 0.0, 1.0);
        let mut right = fwd.cross(up_hint).unit();
        if right.norm() < 1e-6 {
            right = v3(1.0, 0.0, 0.0);
        }
        let up = right.cross(fwd).unit();
        // Pinhole focal length in pixels from the physical lens focal length.
        let fx = LENS_FOCAL_MM * 1000.0 / PIXEL_PITCH_UM;
        let fy = fx;
        let hfov_deg = 2.0 * (CANON_W as f64 * 0.5 / fx).atan().to_degrees();
        Camera {
            id,
            name: format!("CAM{:02}", id + 1),
            pos,
            target,
            right,
            up,
            fwd,
            hfov_deg,
            fx,
            fy,
            // OpenCV (and the renderer) put pixel centres on integer
            // coordinates, so a centred sensor has its principal point at
            // (W-1)/2, not W/2.  Using W/2 shifted every projected dot by
            // half a pixel in both axes; with integer blob centroids that
            // showed up as a 1 px systematic error.
            cx: (CANON_W as f64 - 1.0) * 0.5,
            cy: (CANON_H as f64 - 1.0) * 0.5,
            k1: 0.0,
            k2: 0.0,
            band: "MONO",
            mount_height: pos.z,
        }
    }

    /// Blender camera rotation matrix (columns: local x,y,z in world).
    pub fn blender_rotation(&self) -> M3 {
        [
            [self.right.x, self.up.x, -self.fwd.x],
            [self.right.y, self.up.y, -self.fwd.y],
            [self.right.z, self.up.z, -self.fwd.z],
        ]
    }

    /// OpenCV-style world->camera rotation (x right, y down, z forward).
    pub fn cv_rotation(&self) -> M3 {
        [
            [self.right.x, self.right.y, self.right.z],
            [-self.up.x, -self.up.y, -self.up.z],
            [self.fwd.x, self.fwd.y, self.fwd.z],
        ]
    }

    pub fn blender_lens_mm(&self) -> f64 {
        LENS_FOCAL_MM
    }

    pub fn sensor_width_mm(&self) -> f64 {
        SENSOR_W_MM
    }

    pub fn sensor_height_mm(&self) -> f64 {
        SENSOR_H_MM
    }

    pub fn project(&self, p: V3) -> Option<Projection> {
        let d = p.sub(self.pos);
        let z = d.dot(self.fwd);
        if z <= 0.15 {
            return None;
        }
        let x = d.dot(self.right);
        let y = d.dot(self.up);
        let mut u = self.cx + self.fx * x / z;
        let mut v = self.cy - self.fy * y / z;
        if self.k1 != 0.0 || self.k2 != 0.0 {
            let xn = (u - self.cx) / self.fx;
            let yn = (v - self.cy) / self.fy;
            let r2 = xn * xn + yn * yn;
            let rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
            u = self.cx + self.fx * xn * rad;
            v = self.cy + self.fy * yn * rad;
        }
        Some(Projection { u, v, depth: z })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Projection {
    pub u: f64,
    pub v: f64,
    pub depth: f64,
}

/// Six cameras hung from the hangar roof structure around the edge of the
/// capture volume (no floor stands). 60 deg apart, 5.0 m radius, 6.4 m up,
/// looking down at the middle of the volume. With the OV9281 + 2.8 mm lens
/// (HFOV 68.9 deg) each camera covers a 6.9 m wide swath at the centre; the
/// shared hexagon comfortably contains the 3.8 x 2.0 m figure-eight.
pub fn default_rig() -> Vec<Camera> {
    ring_rig(5.0, 6.40, VOLUME_CENTER)
}

pub fn ring_rig(ring_r: f64, ring_h: f64, center: V3) -> Vec<Camera> {
    assert!(ring_r.is_finite() && ring_r > 0.0, "invalid camera ring radius");
    assert!(ring_h.is_finite() && ring_h > center.z, "camera height must exceed aim height");
    let ring_az: [f64; 6] = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0];
    let mut cams = Vec::new();
    for (i, a) in ring_az.iter().enumerate() {
        let (s, c) = a.to_radians().sin_cos();
        let pos = v3(ring_r * c, ring_r * s, ring_h);
        // Slight per-camera aim offset keeps the views non-degenerate for
        // single-body pose solvers (no two cameras share a principal axis).
        let tgt = center.add(v3(0.20 * s, -0.18 * c, 0.0));
        cams.push(Camera::new(i, pos, tgt));
    }
    cams
}

/// Rigid-body marker layout, mesh body frame (x forward, y left, z up), metres.
#[derive(Clone, Debug)]
pub struct Marker {
    pub id: usize,
    pub local: V3,
    pub radius_m: f64,
    pub normal_local: V3,
    /// Where the marker's mast attaches to the airframe (body frame, metres).
    /// Blender draws a rod from here to `local`; a marker that is not bolted
    /// to the frame is a modelling error, not a shortcut.
    pub mount_local: V3,
}

pub fn crazyflie_markers() -> Vec<Marker> {
    vec![
        Marker {
            id: 0,
            local: v3(0.030, -0.0235, 0.0135),
            radius_m: 0.010,
            normal_local: v3(0.0, 0.0, 1.0),
            mount_local: v3(0.030, -0.0235, 0.0065),
        },
        Marker {
            id: 1,
            local: v3(0.030, 0.0235, 0.0135),
            radius_m: 0.010,
            normal_local: v3(0.0, 0.0, 1.0),
            mount_local: v3(0.030, 0.0235, 0.0065),
        },
        Marker {
            id: 2,
            local: v3(-0.0355, 0.0, 0.0155),
            radius_m: 0.010,
            normal_local: v3(0.0, 0.0, 1.0),
            mount_local: v3(-0.0355, 0.0, 0.0085),
        },
    ]
}

// ------------------------------------------------------------------ bodies
//
// The dataset is body-spec driven: every tracked rigid body carries a frame
// type (which selects both the rendered model and the marker layout) and a
// marker signature -- (base, apex_base) triangle edge lengths -- that the
// multi-target classifier in the consumer pipeline uses to tell drones apart.
// Adding a new airframe means adding one builder here plus one mesh builder in
// blender/mocap_scene.py; nothing else in the chain changes.

#[derive(Clone, Debug)]
pub struct BodySpec {
    pub id: usize,
    pub name: String,
    /// Mesh/model key shared with blender/mocap_scene.py.
    pub frame: &'static str,
    pub markers: Vec<Marker>,
    /// Marker used as the apex of the reference triangle.  The consumer's
    /// apex/base search is declared here instead of being inferred, so a
    /// four-marker layout can never be mistaken for a different reference
    /// triangle by an "equal incident edges" heuristic.
    pub apex_marker_id: usize,
    /// Base pair of the reference triangle, in marker-id order.
    pub base_marker_ids: [usize; 2],
    /// Occlusion radius of the airframe around the body origin (metres).
    pub body_radius_m: f64,
}

impl BodySpec {
    fn marker(&self, id: usize) -> &Marker {
        self.markers
            .iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("body {} has no marker id {}", self.id, id))
    }

    /// (base, apex-base) edge lengths of the declared reference triangle,
    /// metres.  This is the pair the upstream apex/base search consumes.
    pub fn reference_signature(&self) -> (f64, f64) {
        let a = self.marker(self.apex_marker_id).local;
        let b0 = self.marker(self.base_marker_ids[0]).local;
        let b1 = self.marker(self.base_marker_ids[1]).local;
        (b0.sub(b1).norm(), 0.5 * (a.sub(b0).norm() + a.sub(b1).norm()))
    }

    /// All pairwise marker distances (i<j order), sorted ascending.  The
    /// consumer uses the full set for identity/rigid-body verification.
    pub fn edge_lengths(&self) -> Vec<f64> {
        let mut edges = Vec::new();
        for i in 0..self.markers.len() {
            for j in (i + 1)..self.markers.len() {
                edges.push(self.markers[i].local.sub(self.markers[j].local).norm());
            }
        }
        edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
        edges
    }

    /// Markers outside the reference triangle (the identity/rigidity extras).
    pub fn extra_marker_ids(&self) -> Vec<usize> {
        self.markers
            .iter()
            .map(|m| m.id)
            .filter(|id| *id != self.apex_marker_id && !self.base_marker_ids.contains(id))
            .collect()
    }
}

fn quad(
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    c: (f64, f64, f64),
    d: (f64, f64, f64),
    r: f64,
    mounts: [V3; 4],
) -> Vec<Marker> {
    let p = [a, b, c, d];
    (0..4)
        .map(|i| Marker {
            id: i,
            local: v3(p[i].0, p[i].1, p[i].2),
            radius_m: r,
            normal_local: v3(0.0, 0.0, 1.0),
            mount_local: mounts[i],
        })
        .collect()
}

/// Snap a marker's mast foot onto real airframe structure.
///
/// The Ironman 650 is a 176 mm centre-plate stack with four 8 mm arm tubes
/// along the +-45 deg diagonals out to the 230 mm motor mounts.  A mast may
/// only stand on one of those two things; anything else is a floating prop.
/// Points that are neither on an arm (inboard of the motor) nor over the
/// plate are pulled radially onto the plate rim, and the foot is sunk 2 mm
/// into the surface so no gap is visible.
fn tarot_mount(x: f64, y: f64) -> V3 {
    const PLATE_R: f64 = 0.078;   // usable radius of the 88 mm centre plate
    const PLATE_TOP: f64 = 0.061; // 3 mm below the plate face -> embedded
    const ARM_TOP: f64 = 0.028;   // top of the 8 mm arm tube
    const ARM_HALF_W: f64 = 0.008;
    const ARM_ALONG_MAX: f64 = 0.300; // stay clear of the motor/prop hub

    let inv = 1.0 / (2.0_f64).sqrt();
    for (sx, sy) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
        let (ux, uy) = (sx * inv, sy * inv);
        let along = x * ux + y * uy;
        let perp = (x * uy - y * ux).abs();
        if perp <= ARM_HALF_W && along > 0.030 && along <= ARM_ALONG_MAX {
            return v3(ux * along, uy * along, ARM_TOP);
        }
    }

    let r = (x * x + y * y).sqrt();
    if r <= PLATE_R {
        v3(x, y, PLATE_TOP)
    } else {
        let s = PLATE_R / r;
        v3(x * s, y * s, PLATE_TOP)
    }
}

/// Tarot Ironman 650 (650 mm wheelbase, motors at +-0.230 m on the X arms).
///
/// Each airframe carries four markers: the first three form the reference
/// triangle the upstream apex/base search consumes (unchanged from the
/// three-marker dataset), and the fourth is a non-coplanar identity/rigidity
/// marker.  The four-marker layouts have distinct pairwise-distance sets, so
/// a body can be recognised by geometry alone; the fourth marker also
/// disambiguates the reference triangle's base order (which the three-point
/// matcher cannot resolve) and makes roll/pitch observable with redundancy.
pub fn tarot650_markers(variant: usize) -> Vec<Marker> {
    match variant {
        // wide X: two front-motor masts + a tail mast, extra mast on the plate
        0 => quad(
            (0.230, -0.230, 0.165),
            (0.230, 0.230, 0.165),
            (-0.290, 0.0, 0.185),
            (0.045, -0.030, 0.265),
            0.012,
            [
                tarot_mount(0.230, -0.230),
                tarot_mount(0.230, 0.230),
                tarot_mount(-0.290, 0.0),
                tarot_mount(0.045, -0.030),
            ],
        ),
        // compact forward plate rig, extra mast off the symmetry plane
        1 => quad(
            // strongly isosceles: base 0.28 m, apex-edge 0.340 m (ratio 1.21).
            // A near-equilateral triangle (as the original B was, base 0.28 /
            // apex-edge 0.278) is ambiguous at the 25 mm match tolerance: the
            // matcher may pick either base vertex as the apex slash and the
            // returned base-midpoint jumps by ~base/2 every frame.
            (0.050, -0.140, 0.205),
            (0.050, 0.140, 0.205),
            (-0.260, 0.0, 0.205),
            (-0.060, -0.040, 0.290),
            0.012,
            [
                tarot_mount(0.050, -0.140),
                tarot_mount(0.050, 0.140),
                tarot_mount(-0.260, 0.0),
                tarot_mount(-0.060, -0.040),
            ],
        ),
        // compact reverse rig (apex on the nose), extra mast on the plate
        2 => quad(
            (-0.05, -0.17, 0.205),
            (-0.05, 0.17, 0.205),
            (0.360, 0.0, 0.205),
            (0.020, 0.030, 0.280),
            0.012,
            [
                tarot_mount(-0.05, -0.17),
                tarot_mount(-0.05, 0.17),
                tarot_mount(0.360, 0.0),
                tarot_mount(0.020, 0.030),
            ],
        ),
        _ => panic!("tarot650 variant {} undefined", variant),
    }
}

/// The default multi-target scene: three Tarot Ironman 650 quads.
pub fn tarot650_bodies() -> Vec<BodySpec> {
    let frames = ["tarot650_x", "tarot650_fwd", "tarot650_rev"];
    (0..3)
        .map(|i| BodySpec {
            id: i,
            name: format!("T650-{}", ['A', 'B', 'C'][i]),
            frame: frames[i],
            markers: tarot650_markers(i),
            apex_marker_id: 2,
            base_marker_ids: [0, 1],
            body_radius_m: if i == 0 { 0.24 } else { 0.17 },
        })
        .collect()
}

/// Legacy single-body command: the original Crazyflie.
pub fn crazyflie_body() -> BodySpec {
    BodySpec {
        id: 0,
        name: "CF2.1".into(),
        frame: "crazyflie",
        markers: crazyflie_markers(),
        apex_marker_id: 2,
        base_marker_ids: [0, 1],
        body_radius_m: 0.036,
    }
}

/// Convert the mesh-local layout to NED body axes (for downstream tools that
/// expect OptiTrack-style body frames).
pub fn marker_in_ned_body(m: &Marker) -> V3 {
    // mesh (x fwd, y left, z up) -> FRD (x fwd, y right, z down)
    v3(m.local.x, -m.local.y, -m.local.z)
}

#[derive(Clone, Copy, Debug)]
pub struct Detection {
    pub cam: usize,
    pub marker: usize,
    pub u: f64,
    pub v: f64,
    pub radius_px: f64,
    pub intensity: f64,
}

pub struct FrameObs {
    pub frame: usize,
    pub t: f64,
    /// ENU world positions of each marker, truth.
    pub world: Vec<V3>,
    pub detections: Vec<Detection>,
}

pub struct VisionConfig {
    pub noise_px: f64,
    pub dropout: f64,
    pub seed: u64,
    pub body_radius_m: f64,
}

impl Default for VisionConfig {
    fn default() -> Self {
        VisionConfig {
            noise_px: 0.045,
            dropout: 0.03,
            seed: 0x5EED_1234,
            body_radius_m: 0.036,
        }
    }
}

/// Project one frame: which markers are visible from which cameras, with
/// sub-pixel noise, photometric intensity and random dropouts.
pub fn observe(
    frame: usize,
    t: f64,
    p: V3,
    q: Quat,
    cams: &[Camera],
    markers: &[Marker],
    cfg: &VisionConfig,
    rng: &mut Rng,
) -> FrameObs {
    let world: Vec<V3> = markers.iter().map(|m| p.add(q.rotate(m.local))).collect();
    let mut detections = Vec::new();

    for cam in cams {
        for (mi, m) in markers.iter().enumerate() {
            let wp = world[mi];
            let n = q.rotate(m.normal_local);
            let to_cam = cam.pos.sub(wp);
            let dist = to_cam.norm();
            let dir = to_cam.mul(1.0 / dist);

            // Emitter faces away from the camera.
            if n.dot(dir) < 0.05 {
                continue;
            }
            // Body occlusion: closest approach of the marker->camera ray to the
            // airframe centre must be outside the body radius.
            let t_closest = p.sub(wp).dot(dir);
            if t_closest > 0.0 && t_closest < dist {
                let closest = wp.add(dir.mul(t_closest));
                if closest.sub(p).norm() < cfg.body_radius_m {
                    continue;
                }
            }
            let Some(proj) = cam.project(wp) else { continue };
            if proj.u < 6.0
                || proj.v < 6.0
                || proj.u > CANON_W as f64 - 6.0
                || proj.v > CANON_H as f64 - 6.0
            {
                continue;
            }
            if rng.unit() < cfg.dropout {
                continue;
            }
            let u = proj.u + rng.normal() * cfg.noise_px;
            let v = proj.v + rng.normal() * cfg.noise_px;
            // Marker image radius from the pinhole model, clamped to a sane
            // detection blob size.
            let r = (cam.fx * m.radius_m / proj.depth).clamp(0.7, 6.0);
            let intensity = (1.0 / (1.0 + dist / 3.0)).clamp(0.15, 1.0);
            detections.push(Detection {
                cam: cam.id,
                marker: mi,
                u,
                v,
                radius_px: r,
                intensity,
            });
        }
    }

    FrameObs {
        frame,
        t,
        world,
        detections,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Triangulation {
    pub p: V3,
    pub used: usize,
    pub reproj_rms_px: f64,
}

/// Linear multi-view triangulation through the normal equations of the DLT
/// system (2 equations per view).
pub fn triangulate(cams: &[Camera], obs: &[(usize, f64, f64)]) -> Option<Triangulation> {
    if obs.len() < 2 {
        return None;
    }
    // A = 2n x 4, solve A^T A x = A^T b via 4x4 normal equations on the
    // homogeneous system; simpler: parametrise directly in 3D with the two
    // projection equations per view (u,v are linear in (x,y,z)/depth).
    let mut ata = vec![vec![0.0; 3]; 3];
    let mut atb = vec![0.0; 3];
    for &(ci, u, v) in obs {
        let c = &cams[ci];
        let o = c.pos;
        // u = cx + fx * (X-o)·r / (X-o)·f
        // => ((u-cx)/fx) * (X-o)·f - (X-o)·r = 0
        let a1 = c.fwd.mul((u - c.cx) / c.fx).sub(c.right);
        let b1 = a1.dot(o);
        let a2 = c.fwd.mul(-(v - c.cy) / c.fy).sub(c.up);
        let b2 = a2.dot(o);
        for (a, b) in [(a1, b1), (a2, b2)] {
            let av = [a.x, a.y, a.z];
            for i in 0..3 {
                for j in 0..3 {
                    ata[i][j] += av[i] * av[j];
                }
                atb[i] += av[i] * b;
            }
        }
    }
    let x = solve(&ata, &atb)?;
    let p = v3(x[0], x[1], x[2]);
    let mut se = 0.0;
    for &(ci, u, v) in obs {
        if let Some(pr) = cams[ci].project(p) {
            se += (pr.u - u).powi(2) + (pr.v - v).powi(2);
        }
    }
    Some(Triangulation {
        p,
        used: obs.len(),
        reproj_rms_px: (se / (2.0 * obs.len() as f64)).sqrt(),
    })
}

/// Rigid transform between the true marker set and a triangulated set
/// (3-point Kabsch), used to report 6-DOF pose error.
pub fn rigid_fit(from: &[V3], to: &[V3]) -> (Quat, V3) {
    let n = from.len().min(to.len());
    if n == 0 {
        return (Quat::default(), v3(0.0, 0.0, 0.0));
    }
    let cf = from.iter().take(n).fold(v3(0.0, 0.0, 0.0), |a, b| a.add(*b)).mul(1.0 / n as f64);
    let ct = to.iter().take(n).fold(v3(0.0, 0.0, 0.0), |a, b| a.add(*b)).mul(1.0 / n as f64);
    // Covariance
    let mut h = [[0.0f64; 3]; 3];
    for i in 0..n {
        let a = from[i].sub(cf);
        let b = to[i].sub(ct);
        let av = [a.x, a.y, a.z];
        let bv = [b.x, b.y, b.z];
        for r in 0..3 {
            for c in 0..3 {
                h[r][c] += av[r] * bv[c];
            }
        }
    }
    let q = quat_from_cov(&h);
    (q, ct.sub(q.rotate(cf)))
}

fn quat_from_cov(h: &M3) -> Quat {
    // Horn's quaternion method (largest eigenvector via power iteration).
    let (sxx, sxy, sxz) = (h[0][0], h[0][1], h[0][2]);
    let (syx, syy, syz) = (h[1][0], h[1][1], h[1][2]);
    let (szx, szy, szz) = (h[2][0], h[2][1], h[2][2]);
    let n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let mut q = [0.5, 0.5, 0.5, 0.5];
    // Shift the spectrum so power iteration targets the algebraically largest
    // eigenvalue (Horn's optimum) rather than the largest magnitude, which may
    // be a spurious negative eigenvalue.
    let shift: f64 = n
        .iter()
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let mut ns = n;
    for i in 0..4 {
        ns[i][i] += shift;
    }
    for _ in 0..300 {
        let mut nq = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                nq[i] += ns[i][j] * q[j];
            }
        }
        let norm = nq.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        for i in 0..4 {
            q[i] = nq[i] / norm;
        }
    }
    Quat {
        w: q[0],
        x: q[1],
        y: q[2],
        z: q[3],
    }
}

#[cfg(test)]
mod tests_fit {
    use super::*;

    #[test]
    fn rigid_fit_recovers_rotation() {
        let from = vec![
            v3(0.0, 0.0, 0.0),
            v3(1.0, 0.0, 0.0),
            v3(0.0, 1.0, 0.0),
        ];
        let q_true = crate::math::quat(0.0, 0.0, 0.0, 1.0); // 180 deg about z
        let t_true = v3(2.0, -1.0, 0.5);
        let to: Vec<V3> = from.iter().map(|p| q_true.rotate(*p).add(t_true)).collect();
        let (q, t) = rigid_fit(&from, &to);
        let dq = q.mul(q_true.conj());
        let ang = 2.0 * dq.w.abs().clamp(-1.0, 1.0).acos().to_degrees();
        println!("ang err {ang:.3} deg, t err {:.4} m", t.sub(t_true).norm());
        assert!(ang < 1.0, "angle error {ang}");
        assert!(t.sub(t_true).norm() < 1e-6);
    }
}
