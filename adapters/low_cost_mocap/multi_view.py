"""Batched implementation of the repository's ordered multiview search.

The 5-pixel epipolar gate, root creation, candidate order, normal-equation DLT
and float32 OpenCV reprojection score are retained. Geometry is cached by value,
so modifying calibration arrays in place cannot silently reuse old matrices.
Only independent numeric work is batched; temporal state remains in the caller.
"""
from __future__ import annotations

from functools import lru_cache
import itertools

import cv2 as cv
import numpy as np


@lru_cache(maxsize=8)
def _geometry(snapshots, fundamental):
    intrinsics, rotations, translations, projections = [], [], [], []
    for kb, rb, tb in snapshots:
        K = np.frombuffer(kb, dtype=np.float64).reshape(3, 3)
        R = np.frombuffer(rb, dtype=np.float64).reshape(3, 3)
        t = np.frombuffer(tb, dtype=np.float64).reshape(3, 1)
        intrinsics.append(K)
        rotations.append(R)
        translations.append(t)
        projections.append(K @ np.concatenate((R, t), axis=1))
    P = np.asarray(projections)
    matrices = {(i, j): fundamental(P[i], P[j])
                for j in range(len(P)) for i in range(j)}
    return intrinsics, rotations, translations, P, matrices


def _choices(points, fundamentals, frames, drawlines):
    """Keep root and candidate traversal order without deepcopy/product growth."""
    # Roots from each source camera occupy one contiguous range; keep their
    # pixel indices in an array instead of repeatedly scanning Python tuples.
    source_roots = [(0, np.arange(len(points[0]), dtype=np.intp))]
    choices = [[(index,)] for index in range(len(points[0]))]
    for camera in range(1, len(points)):
        current = points[camera]
        if not len(current) and frames is None:
            for options in choices:
                options.append((-1,))
            continue
        lines = np.empty((len(choices), 3), dtype=np.float64)
        start = 0
        for source, indices in source_roots:
            if not indices.size:
                continue
            pixels = points[source][indices].astype(np.float32)
            lines[start:start + len(indices)] = cv.computeCorrespondEpilines(
                pixels, 1, fundamentals[source, camera]).reshape(-1, 3)
            start += len(indices)
        if frames is not None:
            for line in lines:
                frames[camera] = drawlines(frames[camera], line.reshape(1, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            distances = np.abs(lines[:, None, 0] * current[None, :, 0]
                               + lines[:, None, 1] * current[None, :, 1] + lines[:, None, 2])
            distances /= np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2)[:, None]
        nearest = set()
        for row, options in enumerate(choices):
            matches = (distances[row] < 5).nonzero()[0]
            if matches.size:
                # Match the original default argsort, including its tie order.
                matches = matches[distances[row, matches].argsort()]
                options.append(tuple(int(index) for index in matches))
                # The original removes all equal-coordinate rows, not just
                # one index, when deciding which observations become roots.
                nearest.add(tuple(current[matches[0]]))
            else:
                options.append((-1,))
        new_roots = []
        for index, point in enumerate(current):
            if tuple(point) not in nearest:
                new_roots.append(index)
                choices.append([(-1,)] * camera + [(index,)])
        source_roots.append((camera, np.asarray(new_roots, dtype=np.intp)))
    return choices


def _candidate_rows(choices):
    for root, options in enumerate(choices):
        if sum(option != (-1,) for option in options) < 2:
            continue
        # In upstream, each newly added camera is the OUTER loop. Reverse the
        # product dimensions to retain the first-minimum tie-breaking order.
        for reverse_indices in itertools.product(*reversed(options)):
            yield root, reverse_indices[::-1]


def _solve(indices, points, geometry):
    K, R, t, P, _ = geometry
    present = indices >= 0
    observations = np.zeros((*indices.shape, 2), dtype=np.float64)
    for camera, pixels in enumerate(points):
        rows = present[:, camera]
        observations[rows, camera] = pixels[indices[rows, camera]]
    # Zero rows stand in for absent views, preserving their calibrated slots.
    # As in upstream: A^T A -> smallest right singular vector -> homogeneous
    # division. Batch SVD may differ in the last floating-point bits.
    A = np.empty((*indices.shape, 2, 4), dtype=np.float64)
    A[:, :, 0] = observations[:, :, 1, None] * P[None, :, 2] - P[None, :, 1]
    A[:, :, 1] = P[None, :, 0] - observations[:, :, 0, None] * P[None, :, 2]
    A *= present[:, :, None, None]
    A = A.reshape(len(indices), -1, 4)
    gram = A.swapaxes(1, 2) @ A
    _, _, vh = np.linalg.svd(gram, full_matrices=False)
    homogeneous = vh[:, -1]
    valid = np.abs(homogeneous[:, 3]) >= 1e-12
    world = np.full((len(indices), 3), np.nan)
    world[valid] = homogeneous[valid, :3] / homogeneous[valid, 3, None]
    valid &= np.isfinite(world).all(axis=1)
    errors = np.full(len(indices), np.inf)
    if not valid.any():
        return errors, world
    rows = valid.nonzero()[0]
    # Preserve OpenCV's float32 input/output and its handling of R. Replacing
    # this with double-precision P multiplication changes the legacy score.
    world32 = world[rows].astype(np.float32)
    predicted = np.empty((len(rows), len(points), 2), dtype=np.float32)
    for camera in range(len(points)):
        predicted[:, camera] = cv.projectPoints(world32, R[camera], t[camera], K[camera],
                                               np.array([]))[0].reshape(-1, 2)
    squared = (observations[rows] - predicted) ** 2
    sums = np.where(present[rows, :, None], squared, 0).sum(axis=(1, 2))
    scores = sums / (2 * present[rows].sum(axis=1))
    errors[rows] = np.where(np.isfinite(scores), scores, np.inf)
    return errors, world


def fuse_points(image_points, camera_poses, intrinsics, fundamental, frames=None,
                drawlines=None, batch_size=2048):
    if (len(image_points) != len(camera_poses) or len(intrinsics) != len(camera_poses)
            or len(camera_poses) < 2):
        raise ValueError("one observation list is required per calibrated camera (at least two)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    points = [np.asarray([p for p in camera if len(p) == 2 and
                          all(x is not None and np.isfinite(x) for x in p)], dtype=np.float64).reshape(-1, 2)
              for camera in image_points]
    if not any(len(camera) for camera in points):
        return np.empty(0), np.empty((0, 3)), frames
    snapshots = tuple((np.asarray(K, dtype=np.float64).reshape(3, 3).tobytes(),
                       np.asarray(pose["R"], dtype=np.float64).reshape(3, 3).tobytes(),
                       np.asarray(pose["t"], dtype=np.float64).reshape(3, 1).tobytes())
                      for K, pose in zip(intrinsics, camera_poses))
    geometry = _geometry(snapshots, fundamental)
    choices = _choices(points, geometry[-1], frames, drawlines)
    best_error = np.full(len(choices), np.inf)
    best_points = np.empty((len(choices), 3), dtype=np.float64)
    candidates = _candidate_rows(choices)
    while batch := list(itertools.islice(candidates, batch_size)):
        roots = np.asarray([root for root, _ in batch])
        indices = np.asarray([row for _, row in batch], dtype=np.intp)
        errors, world = _solve(indices, points, geometry)
        boundaries = [0, *((roots[1:] != roots[:-1]).nonzero()[0] + 1), len(roots)]
        for start, end in zip(boundaries, boundaries[1:]):
            root = roots[start]
            winner = start + np.argmin(errors[start:end])
            # Strict < keeps the earliest candidate when scores tie, including
            # ties crossing a batch boundary. No candidate is pruned or capped.
            if errors[winner] < best_error[root]:
                best_error[root] = errors[winner]
                best_points[root] = world[winner]
    valid = np.isfinite(best_error)
    return best_error[valid], best_points[valid], frames
