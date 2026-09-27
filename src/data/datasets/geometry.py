"""Pure geometric helpers for MM-OR scene-graph construction.

Oriented-bounding-box distance utilities (point->OBB, OBB<->OBB surface) and
world-space coordinate jitter. Pure numpy — no graph/dataset state.
"""

from __future__ import annotations

import numpy as np


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _obb_sample_points(center, dims, quat):
    R = quaternion_to_rotation_matrix(quat)
    half = dims / 2.0
    signs = np.array(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
        dtype=np.float64,
    )
    corners = (R @ (signs * half).T).T + center
    faces = []
    for axis in range(3):
        for sign in [-1, 1]:
            pt = np.zeros(3)
            pt[axis] = sign * half[axis]
            faces.append((R @ pt) + center)
    return np.vstack([corners, np.array(faces)])


def _point_to_obb_distance(point, center, dims, quat):
    R = quaternion_to_rotation_matrix(quat)
    local = R.T @ (point - center)
    half = dims / 2.0
    clamped = np.clip(local, -half, half)
    if np.all(np.abs(local) <= half):
        return 0.0
    return np.linalg.norm(local - clamped)


def _obb_surface_distance(
    center_a: np.ndarray,
    dims_a: np.ndarray,
    quat_a: np.ndarray,
    center_b: np.ndarray,
    dims_b: np.ndarray,
    quat_b: np.ndarray,
    check_threshold: float = float("inf"),
) -> float:
    radius_a = np.linalg.norm(dims_a) / 2.0
    radius_b = np.linalg.norm(dims_b) / 2.0
    dist_centers = np.linalg.norm(center_a - center_b)
    min_sphere_dist = dist_centers - (radius_a + radius_b)
    if min_sphere_dist > check_threshold:
        return min_sphere_dist

    pts_a = _obb_sample_points(center_a, dims_a, quat_a)
    pts_b = _obb_sample_points(center_b, dims_b, quat_b)

    min_dist = float("inf")
    for pt in pts_a:
        d = _point_to_obb_distance(pt, center_b, dims_b, quat_b)
        if d < min_dist:
            min_dist = d
    for pt in pts_b:
        d = _point_to_obb_distance(pt, center_a, dims_a, quat_a)
        if d < min_dist:
            min_dist = d
    return min_dist


def _jitter_world_coords(coords: np.ndarray, noise_std_mm: float) -> np.ndarray:
    """Perturb world-space mm coordinates before graph construction, so every derived
    feature is computed from the same jittered positions."""
    if noise_std_mm <= 0:
        return coords
    noise = np.random.randn(*coords.shape).astype(coords.dtype) * noise_std_mm
    return coords + noise
