"""Multi-target rigid-body localisation for the Low-Cost-Mocap adapter.

This adapts upstream ``locate_objects`` (api/helpers.orig.py), retaining its
apex search, base-midpoint position and heading convention, with triangle
parameters loaded from a file instead of one hard-coded triangle
to N triangles:

    upstream                          here
    ------------------------------    ------------------------------------
    dist1/dist2 from two env vars     one reference triangle per body spec
                                      (declared apex/base marker ids)
    apex = point with two dist1       identical (the search *is* the apex
    neighbours                        test, no fit for the search itself)
    pos   = midpoint of base pair     identical
    heading = base-pair direction,    identical
              wrapped to +-pi/2
    droneIndex = side of the third    which template matched (the bodies
    light (2 drones, one rig)         carry distinct marker layouts)
    greedy consumption of points      identical, and shared across
                                      templates so no marker serves
                                      two drones

Each body may carry markers beyond its reference triangle.  Those extras are
matched against the pose predicted by the reference triangle, consumed so two
bodies cannot share them, and included in a Kabsch rigid fit.  This adds:
  * identity evidence (an extra that lands where the rigid body says it must),
  * a unique base-pair order (the three-point search alone cannot distinguish
    the two label orders of an isosceles base),
  * a full 6-DOF pose and its fit residual.
Extras are optional: when occluded, the body is still reported from the
reference triangle (``pose_mode == "triangle3"``). A recent accepted rotation
selects the nearest of its two base-order branches; the output records whether
extra markers, a temporal prior, or neither resolved that ambiguity.

The optional motion gate uses capture timestamps in seconds.  It is a local
extension and is evaluated separately from the original repository.  No
ground truth is consulted.

The application selects ``position_reference="body_origin"``: after resolving
the marker order, the rigid-fit translation is used by BOTH the motion gate
and Kalman filter. The measured base midpoint remains a diagnostic field.
The low-level default retains the upstream midpoint for legacy comparisons.
"""
from __future__ import annotations

import itertools
import json
import os

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

TOL_DEFAULT = 0.025          # metres, per-edge budget (upstream default)
EXTRA_TOL_DEFAULT = 0.050    # metres, rigid prediction budget for extra markers
REFERENCE_ISOSCELES_TOL = 0.003
POSITION_REFERENCES = ("body_origin", "base_midpoint")
POSITION_REFERENCE_LABELS = {
    "body_origin": "body-frame origin (0, 0, 0), geometric reference",
    "base_midpoint": "declared base-marker midpoint",
}


# --------------------------------------------------------------- templates
def _load_legacy_reference(lm, marker_ids, bid):
    """Infer the reference triangle for a three-marker legacy body spec."""
    pairs = list(itertools.combinations(range(len(lm)), 2))
    edges = [float(np.linalg.norm(lm[i] - lm[j])) for i, j in pairs]
    incident = [[e for pair, e in zip(pairs, edges) if v in pair]
                for v in range(len(lm))]
    deltas = [abs(e[0] - e[1]) for e in incident]
    apex = int(np.argmin(deltas))
    if sorted(deltas)[1] - min(deltas) < 1e-7:
        raise ValueError(f"body {bid}: ambiguous template apex; specify apex_marker_id")
    if deltas[apex] > 1e-5:
        raise ValueError(f"body {bid} is not isosceles; specify a reference triangle")
    base_pair = next(pair for pair in pairs if apex not in pair)
    return apex, base_pair


def load_body_templates(path: str) -> list[dict]:
    """Read every rigid body's marker layout and reference triangle.

    A body with three markers keeps the original implicit isosceles layout.
    A body with four or more markers must declare ``apex_marker_id`` and
    ``base_marker_ids`` (the Rust writer emits both), because inferring them
    by "most equal incident edges" is ambiguous once extra markers exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        rig = json.load(f)

    bodies = rig.get("bodies") or []
    if not bodies:
        if not rig.get("markers"):
            raise ValueError(f"{path} has neither 'bodies' nor 'markers'")
        bodies = [{"body_id": 0, "name": "body0", "markers": rig["markers"]}]

    out = []
    seen_ids = set()
    for b in bodies:
        lm = np.array([m["local"] for m in b["markers"]], dtype=float)
        k = len(lm)
        bid = int(b.get("body_id", 0))
        if bid in seen_ids:
            raise ValueError(f"duplicate body_id {bid}")
        seen_ids.add(bid)
        if k < 3 or lm.shape != (k, 3) or not np.all(np.isfinite(lm)):
            raise ValueError(f"body {bid} needs at least three finite 3-D markers")
        marker_ids = [int(m["id"]) for m in b["markers"]]
        if len(set(marker_ids)) != k:
            raise ValueError(f"body {bid} has duplicate marker ids")

        pairs = list(itertools.combinations(range(k), 2))
        edges = [float(np.linalg.norm(lm[i] - lm[j])) for i, j in pairs]

        declared = "apex_marker_id" in b or "base_marker_ids" in b
        if declared:
            if "apex_marker_id" not in b or "base_marker_ids" not in b:
                raise ValueError(f"body {bid} must declare apex_marker_id and base_marker_ids together")
            if b["apex_marker_id"] not in marker_ids:
                raise ValueError(f"body {bid} has an unknown apex_marker_id")
            base_ids = [int(value) for value in b["base_marker_ids"]]
            if len(base_ids) != 2 or any(value not in marker_ids for value in base_ids):
                raise ValueError(f"body {bid} has invalid base_marker_ids")
            apex = marker_ids.index(int(b["apex_marker_id"]))
            base_pair = tuple(marker_ids.index(value) for value in base_ids)
            if apex in base_pair or base_pair[0] == base_pair[1]:
                raise ValueError(f"body {bid} reference triangle repeats a marker")
            a = lm[apex]
            d0 = float(np.linalg.norm(a - lm[base_pair[0]]))
            d1 = float(np.linalg.norm(a - lm[base_pair[1]]))
            if abs(d0 - d1) > REFERENCE_ISOSCELES_TOL:
                raise ValueError(f"body {bid} reference triangle is not isosceles")
            dist1, dist2 = 0.5 * (d0 + d1), float(
                np.linalg.norm(lm[base_pair[0]] - lm[base_pair[1]]))
        else:
            if k != 3:
                raise ValueError(f"body {bid} has {k} markers; declare apex_marker_id and base_marker_ids")
            apex, base_pair = _load_legacy_reference(lm, marker_ids, bid)
            dist1 = float(np.mean([edges[pairs.index(tuple(sorted((apex, v))))]
                                   for v in base_pair]))
            dist2 = float(np.linalg.norm(lm[base_pair[0]] - lm[base_pair[1]]))

        extras = [i for i in range(k) if i != apex and i not in base_pair]
        if extras:
            plane_normal = np.cross(lm[base_pair[0]] - lm[apex],
                                    lm[base_pair[1]] - lm[apex])
            norm = np.linalg.norm(plane_normal)
            if norm < 1e-9:
                raise ValueError(f"body {bid} has a degenerate reference triangle")
            off_plane = [abs(float((lm[e] - lm[apex]) @ (plane_normal / norm)))
                         for e in extras]
            rigidity_margin = max(off_plane)
        else:
            rigidity_margin = 0.0

        out.append({
            "body_id": bid,
            "name": b.get("name", f"body{b.get('body_id', 0)}"),
            "frame": b.get("frame", "unknown"),
            "marker_ids": marker_ids,
            "locals": lm,
            "pairs": pairs,
            "edges": edges,
            "edge_lengths": sorted(edges),
            "apex": apex,
            "base_pair": base_pair,
            "extras": extras,
            "dist1": dist1,
            "dist2": dist2,
            "edge_separation_m": abs(dist1 - dist2),
            "base_mid_local": 0.5 * (lm[base_pair[0]] + lm[base_pair[1]]),
            "rigidity_margin_m": rigidity_margin,
        })
    out.sort(key=lambda t: t["body_id"])
    return out



def rigid_pose(local, world):
    """Least-squares rigid transform local -> world (proper rotation, no scale).

    Returns ``(R, t, rmse_m)``.  Kabsch via SVD; the reflection guard keeps
    det(R) = +1 so a mirrored point set is reported as a large residual
    instead of being silently accepted as a rotation.
    """
    local = np.asarray(local, dtype=float)
    world = np.asarray(world, dtype=float)
    if local.shape != world.shape or local.ndim != 2 or local.shape[1] != 3:
        raise ValueError("rigid_pose expects matching (N,3) arrays")
    if len(local) < 3:
        raise ValueError("rigid_pose needs at least three correspondences")
    lc, wc = local.mean(axis=0), world.mean(axis=0)
    cov = (local - lc).T @ (world - wc)
    # The matrix is always 3x3. OpenCV avoids general NumPy/LAPACK dispatch
    # overhead while solving the same double-precision Kabsch SVD.
    _, u, vt = cv.SVDecomp(cov)
    sign = 1.0 if cv.determinant(vt.T @ u.T) >= 0 else -1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    translation = wc - rotation @ lc
    residual = world - (rotation @ local.T).T - translation
    rmse = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    return rotation, translation, rmse


def _pose_fields(rotation, translation, rmse, body_id, marker_ids, mode, tolerance):
    euler = Rotation.from_matrix(rotation).as_euler("ZYX", degrees=True)
    body_up = rotation @ np.array([0.0, 0.0, 1.0])
    return {
        "pose_r": rotation.tolist(),
        "pose_t": [float(v) for v in translation],
        "pose_euler_zyx_deg": [float(v) for v in euler],  # yaw, pitch, roll
        "pose_tilt_deg": float(np.degrees(np.arccos(np.clip(body_up[2], -1.0, 1.0)))),
        "pose_rmse_m": float(rmse),
        "pose_mode": mode,
        "markers_used": int(len(marker_ids)),
        "marker_ids": [int(v) for v in marker_ids],
        "pose_ambiguous": mode != "rigid",
        "pose_disambiguation": "extra_marker" if mode == "rigid" else "unresolved",
        "pose_tolerance_m": float(tolerance),
        "droneIndex": body_id,
    }


def _rigid_assignment(points, tpl, apex_i, base_order, blocked, extra_tol):
    """Fit the reference triangle, then find its declared extra markers.

    ``base_order`` fixes which world point is the first/second base marker.
    The two orders are geometrically indistinguishable from three points of
    an isosceles triangle, so the caller evaluates both and keeps the one the
    extras support.
    """
    apex, (b0, b1) = tpl["apex"], tpl["base_pair"]
    local3 = tpl["locals"][[apex, b0, b1]]
    world3 = points[[apex_i, base_order[0], base_order[1]]]
    rotation, translation, rmse = rigid_pose(local3, world3)

    used = {int(apex_i), int(base_order[0]), int(base_order[1])}
    local_idx = [apex, b0, b1]
    world_idx = [int(apex_i), int(base_order[0]), int(base_order[1])]
    extra_distance = []
    for marker in tpl.get("extras", ()):
        predicted = rotation @ tpl["locals"][marker] + translation
        distances = np.linalg.norm(points - predicted, axis=1)
        distances[list(used.union(blocked))] = np.inf
        best = int(np.argmin(distances))
        if distances[best] <= extra_tol:
            used.add(best)
            local_idx.append(marker)
            world_idx.append(best)
            extra_distance.append(float(distances[best]))

    if extra_distance:
        rotation, translation, rmse = rigid_pose(tpl["locals"][local_idx], points[world_idx])
    ids = [tpl["marker_ids"][i] for i in local_idx]
    mode = "rigid" if extra_distance else "triangle3"
    # Euler conversion and JSON-facing fields are needed only for the winning
    # branch. Keep numeric evidence here while the two base orders compete.
    return {"pose_r": rotation, "pose_t": translation, "pose_rmse_m": rmse,
            "pose_mode": mode, "marker_ids": ids,
            "extra_residuals_m": extra_distance, "_used_points": tuple(world_idx)}


def _assign_geometry(points, tpl, apex_i, base, blocked, extra_tol,
                     rotation_prior=None):
    """Pick the base order and extras that the rigid body supports best."""
    if "locals" not in tpl or not tpl.get("extras"):
        # Three-point vectors carry no declared identity or base order, so the
        # pose fields are omitted rather than invented from an arbitrary label
        # order.  Callers still get the upstream pos/heading/error.
        if not all(key in tpl for key in ("marker_ids", "apex", "base_pair")):
            return {"extra_residuals_m": []}
        return {"pose_mode": "triangle3", "markers_used": 3,
                "marker_ids": [tpl["marker_ids"][tpl["apex"]],
                               tpl["marker_ids"][tpl["base_pair"][0]],
                               tpl["marker_ids"][tpl["base_pair"][1]]],
                "pose_ambiguous": True, "pose_disambiguation": "unresolved",
                "_used_points": (),
                "extra_residuals_m": []}
    first = _rigid_assignment(points, tpl, apex_i, base, blocked, extra_tol)
    second = _rigid_assignment(points, tpl, apex_i, (base[1], base[0]), blocked, extra_tol)
    # More extras matched wins, then lower rigid-fit residual. Without extra
    # evidence, a fresh prior chooses the orientation branch. The upstream
    # base-midpoint/undirected-heading measurement is unchanged by this choice.
    first_key = (len(first["_used_points"]), -first["pose_rmse_m"])
    second_key = (len(second["_used_points"]), -second["pose_rmse_m"])
    use_prior = (first["pose_mode"] == "triangle3" and second["pose_mode"] == "triangle3"
                 and rotation_prior is not None)
    if use_prior:
        # An isosceles reference triangle is invariant under the 180 deg
        # rotation that swaps its base markers, so three points cannot say
        # which order is the physical one.  The two branches differ by that
        # half-turn about the apex-bisector axis, which may read as a yaw or a
        # roll flip depending on the mounting; comparing the full rotation
        # (not just the body-up axis) catches both.  Continuity with the last
        # accepted fix assumes motion stayed closer to that branch than to
        # its half-turn alternative during the capture interval.
        first_r = np.asarray(first["pose_r"], dtype=float)
        second_r = np.asarray(second["pose_r"], dtype=float)
        selected = (second if _rotation_distance(second_r, rotation_prior) <
                     _rotation_distance(first_r, rotation_prior) else first)
    else:
        selected = second if second_key > first_key else first
    fields = _pose_fields(selected["pose_r"], selected["pose_t"], selected["pose_rmse_m"],
                          tpl["body_id"], selected["marker_ids"], selected["pose_mode"], extra_tol)
    fields["extra_residuals_m"] = selected["extra_residuals_m"]
    fields["_used_points"] = selected["_used_points"]
    if use_prior:
        fields["pose_disambiguation"] = "temporal_prior"
    return fields


# ------------------------------------------------------------- the matcher
def _locate_one(points, errors, distance_matrix, dist1, dist2, tol,
                drone_index, already_matched):
    """Upstream apex/base search with parameters and shared point ownership.

    Returns candidates in the original repository's traversal order, so the caller can
    try the next geometry-consistent triple when an earlier candidate fails a
    motion/error gate instead of dropping the body for that frame.
    """
    n = points.shape[0]
    candidates = []
    available = np.ones(n, dtype=bool)
    available[list(already_matched)] = False
    apex_edges = (np.abs(distance_matrix - dist1) < tol)
    apex_edges &= available[:, None] & available[None, :]
    base_edges = np.abs(distance_matrix - dist2) <= tol
    np.fill_diagonal(apex_edges, False)
    np.fill_diagonal(base_edges, False)
    # nonzero traverses C order: apex ascending, then first base, then second
    # base. This is the same order as the original nested product loop. Chunk
    # the apex dimension instead of allocating an unbounded N^3 mask.
    for start in range(0, n, 16):
        adjacent = apex_edges[start:start + 16]
        a, b0, b1 = (adjacent[:, :, None] & adjacent[:, None, :] & base_edges[None]).nonzero()
        a += start
        positions = (points[b0] + points[b1]) / 2
        triangle_errors = np.stack((errors[a], errors[b0], errors[b1]), axis=1).mean(axis=1)
        directions = points[b0] - points[b1]
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        headings = np.arctan2(directions[:, 1], directions[:, 0])
        headings = np.where(headings > np.pi / 2, headings - np.pi, headings)
        headings = np.where(headings < -np.pi / 2, headings + np.pi, headings)
        for row in range(len(a)):
            candidates.append({"pos": positions[row], "heading": -float(headings[row]),
                               "error": float(triangle_errors[row]), "droneIndex": drone_index,
                               "apex_i": int(a[row]), "base": (int(b0[row]), int(b1[row])),
                               "_edge_ok": True})
    # Keep the upstream vertex-ascending / product order (do NOT re-rank by
    # error): re-ranking can flip the base-pair order and turn the heading by
    # pi.  Gates try candidates in this order and stop at the first accept.
    return candidates


def locate_bodies(object_points, errors, templates, tracker=None,
                  tol=TOL_DEFAULT, frame=0, dt=1.0 / 30.0, max_err=None,
                  timestamp=None, extra_tol=EXTRA_TOL_DEFAULT,
                  position_reference="base_midpoint",
                  **_ignored):
    """Run the upstream matcher once per body signature.

    ``tracker`` is an optional per-body motion gate (see BodyTracker); it only
    rejects physically impossible single-frame jumps and never consults truth.
    ``max_err`` limits mean reprojection MSE (px^2) of the reference triangle.
    Low reprojection error alone does not establish that a bright point is a
    tracking marker.
    ``timestamp`` is the actual capture time in seconds; frame*dt is only a
    compatibility fallback for fixed-rate offline callers.
    ``extra_tol`` is the rigid prediction budget for non-reference markers.
    ``position_reference`` selects the point consumed by the motion gate and
    output as ``pos``. Body-origin tracking requires a resolved rigid pose;
    an unresolved three-point pose cannot silently change the reference point.

    Candidate selection stays in upstream traversal order.  The only exception
    is a candidate whose extra markers confirm the rigid body: such a candidate
    wins over an earlier gate-passing triple that has no extra support.  That
    is positive marker evidence, not a ground-truth score, and it cannot flip
    the heading because the extras determine the base order.
    """
    if position_reference not in POSITION_REFERENCES:
        raise ValueError(f"unknown position_reference: {position_reference}")
    points = np.asarray(object_points, dtype=float)
    errors = np.asarray(errors, dtype=float).ravel()
    if points.size == 0:
        points = points.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] != 3 or len(errors) != len(points):
        raise ValueError("expected points (N,3) and one error per point")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(errors)):
        raise ValueError("non-finite input point or reprojection error")
    if not np.isfinite(extra_tol) or extra_tol <= 0:
        raise ValueError("extra_tol must be positive")
    capture_time = float(timestamp) if timestamp is not None else float(frame) * float(dt)
    if not np.isfinite(capture_time):
        raise ValueError("capture timestamp must be finite")
    if tracker is not None and not tracker.start_frame(capture_time):
        return []  # a repeated capture cannot update state or missing counts
    n = points.shape[0]
    if n < 3:
        if tracker is not None:
            tracker.finish_frame(set())
        return []

    distance_matrix = np.sqrt(np.sum((points[:, None] - points[None, :]) ** 2, axis=2))

    already_matched: list[int] = []
    objects = []
    for tpl in templates:
        candidates = _locate_one(points, errors, distance_matrix,
                                 tpl["dist1"], tpl["dist2"], tol,
                                 tpl["body_id"], already_matched)
        rotation_prior = (tracker.rotation_prior(tpl["body_id"])
                          if tracker is not None else None)
        fallback = None
        chosen = None
        for obj in candidates:
            if max_err is not None and obj["error"] > max_err:
                continue
            if (position_reference == "base_midpoint" and tracker is not None
                    and not tracker.would_accept(obj, tpl["body_id"])):
                continue
            geometry = _assign_geometry(points, tpl, obj["apex_i"], obj["base"],
                                        already_matched, extra_tol, rotation_prior)
            geometry["body_id"] = tpl["body_id"]
            obj["base_midpoint_measured"] = obj["pos"].copy()
            obj["position_reference"] = position_reference
            if position_reference == "body_origin":
                if (geometry.get("pose_t") is None or
                        geometry.get("pose_disambiguation") not in ("extra_marker", "temporal_prior")):
                    continue
                obj["pos"] = np.asarray(geometry["pose_t"], dtype=float).copy()
                if tracker is not None and not tracker.would_accept(obj, tpl["body_id"]):
                    continue
            if fallback is None:
                fallback = (obj, geometry)
            if geometry.get("pose_mode") == "rigid":
                chosen = (obj, geometry)
                break
        if chosen is None:
            chosen = fallback
        if chosen is None:
            continue
        obj, geometry = chosen
        used_points = geometry.pop("_used_points", ())
        obj.update(geometry)
        # Commit the fix only after the pose is attached: the tracker keeps the
        # rotation to disambiguate the next frame's three-point solution.
        if tracker is not None and not tracker.accept(obj, tpl["body_id"]):
            continue
        obj["reference_ambiguous"] = tpl["edge_separation_m"] <= 2 * tol
        obj["name"] = tpl["name"]
        obj["timestamp"] = capture_time
        consumed = {obj["apex_i"], *obj["base"], *used_points}
        already_matched.extend(p for p in consumed if p not in already_matched)
        objects.append(obj)
    if tracker is not None:
        tracker.finish_frame({obj["droneIndex"] for obj in objects})
    return objects


def _body_rotation(obj):
    """World rotation of the body, if the pose carries one."""
    rotation = obj.get("pose_r")
    if rotation is None:
        return None
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    return matrix.copy()


def _rotation_distance(a, b):
    """Geodesic angle (rad) between two rotation matrices."""
    value = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) * 0.5
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


# --------------------------------------------------- optional temporal gate
class BodyTracker:
    """Timestamped constant-velocity innovation gate; units are m, m/s, s.

    Candidate rejection never changes the motion state or the gate width.
    ``finish_frame`` counts a loss at most once per capture, including empty
    observations. After reacquire_gap_s without a fix the velocity prediction
    expires; the next geometric match initializes a new track with zero
    velocity instead of extrapolating stale motion indefinitely.
    """

    def __init__(self, templates, gate=0.5, alpha=0.6, reacquire_gap_s=0.5):
        if not np.isfinite(gate) or gate <= 0 or not 0 < alpha <= 1:
            raise ValueError("gate must be positive and alpha in (0,1]")
        if not np.isfinite(reacquire_gap_s) or reacquire_gap_s <= 0:
            raise ValueError("reacquire_gap_s must be positive")
        self.templates = templates
        self.gate, self.alpha = float(gate), float(alpha)
        self.reacquire_gap_s = float(reacquire_gap_s)
        self.state = {}
        self.current_time = None
        self.last_frame_time = None

    def reset(self):
        self.state.clear()
        self.current_time = None
        self.last_frame_time = None

    def start_frame(self, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.last_frame_time is not None:
            if timestamp == self.last_frame_time:
                return False
            if timestamp < self.last_frame_time:
                self.reset()  # a replay seek/loop starts a new time epoch
        self.current_time = timestamp
        return True

    def predict(self, body_id):
        state = self.state.get(body_id)
        if state is None or self.current_time - state["last_t"] > self.reacquire_gap_s:
            return None
        elapsed_s = self.current_time - state["last_t"]
        return state["pos"] + state["vel"] * elapsed_s

    def rotation_prior(self, body_id):
        """World rotation from the last accepted fix, or None when stale.

        Only used to pick between the two orientation branches of a symmetric
        three-marker triangle; it never moves or gates the position.
        """
        state = self.state.get(body_id)
        if state is None or self.current_time is None:
            return None
        if self.current_time - state["last_t"] > self.reacquire_gap_s:
            return None
        rotation = state.get("rot")
        return None if rotation is None else np.asarray(rotation, dtype=float).copy()

    def would_accept(self, obj, body_id):
        """Gate check without mutating state, so candidates can be compared.

        ``accept`` performs the same check and then commits the update.
        """
        if self.current_time is None:
            raise RuntimeError("start_frame(timestamp) must precede candidate evaluation")
        pos = np.asarray(obj["pos"], dtype=float)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            return False
        predicted = self.predict(body_id)
        if predicted is None:
            return True
        elapsed_s = self.current_time - self.state[body_id]["last_t"]
        return elapsed_s > 0 and float(np.linalg.norm(pos - predicted)) <= self.gate

    def status(self, body_id):
        """Per-body lock state for the operator UI (no truth involved)."""
        state = self.state.get(body_id)
        if state is None or self.current_time is None:
            return {"state": "lost", "misses": 0, "age_s": None, "locked": False}
        age = self.current_time - state["last_t"]
        if age > self.reacquire_gap_s:
            label = "lost"
        elif state["misses"] >= 3:
            label = "coasting"
        else:
            label = "locked"
        return {"state": label, "misses": int(state["misses"]),
                "age_s": float(age), "locked": label == "locked"}

    def accept(self, obj, body_id):
        if self.current_time is None:
            raise RuntimeError("start_frame(timestamp) must precede candidate evaluation")
        pos = np.asarray(obj["pos"], dtype=float)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            return False
        state = self.state.get(body_id)
        predicted = self.predict(body_id)
        if predicted is None:
            self.state[body_id] = {"pos": pos.copy(), "vel": np.zeros(3),
                                  "last_t": self.current_time, "misses": 0,
                                  "rot": _body_rotation(obj)}
            return True
        elapsed_s = self.current_time - state["last_t"]
        if elapsed_s <= 0 or np.linalg.norm(pos - predicted) > self.gate:
            return False
        measured_velocity = (pos - state["pos"]) / elapsed_s
        state["vel"] = (1.0 - self.alpha) * state["vel"] + self.alpha * measured_velocity
        state["pos"] = pos.copy()
        state["last_t"] = self.current_time
        state["misses"] = 0
        rotation = _body_rotation(obj)
        if rotation is not None:
            state["rot"] = rotation
        return True

    def finish_frame(self, found_ids):
        if self.current_time is None or self.current_time == self.last_frame_time:
            return
        for body_id, state in self.state.items():
            if body_id not in found_ids:
                state["misses"] += 1
        self.last_frame_time = self.current_time


def default_rig_path() -> str:
    root = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    return os.path.join(root, "out", "rig3", "rig_blender.json")
