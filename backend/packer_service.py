"""
packer_service.py
==================

Pure packing logic shared by the HTTP layer (server.py). Wraps GridPacker
and image_boundary so packing can be tested directly, without going through
FastAPI or HTTP.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon

from grid_packer import GridPacker

Point = Tuple[float, float]

DEFAULT_STEPS = 10
ROTATE_STEP = 15


def _rotate_angles(cell_width: float, cell_height: float) -> Tuple[float, ...]:
    """Grid angles worth testing, in degrees.

    Translation needs no direction sweep: the grid is periodic, so the offsets
    dx in [0, cw) / dy in [0, ch) that optimize() already scans cover every
    possible shift in every direction (shifting left by k is the same grid as
    shifting right by cw - k).

    Rotation is different. Turning a cw x ch grid by 180 deg reproduces it, so
    the search space is [0, 180). Only when cells are SQUARE does 90 deg also
    reproduce it -- for rectangular cells a quarter turn swaps cw and ch, which
    is a genuinely different packing and can be the better one.

    RETAINED AS THE BASELINE, not used to serve requests. This fixed ladder plus
    the uniform offset sweep is what the packer used to do, and the paper's
    ablation measures the new solver against it, so it has to stay runnable and
    unchanged. `_solve` is what requests go through.
    """
    period = 90 if cell_width == cell_height else 180
    return tuple(range(0, period, ROTATE_STEP))


def _solve(packer: GridPacker, rotate: bool):
    """Place the grid, and read back what the placement certifies.

    Translation is solved either way rather than sampled, and on BOTH axes: the
    offsets are not enumerated at all, they are read off the deepest overlap of
    the region eroded by the cell and folded onto the grid period. So there is
    no `steps` resolution to pick, no sharp optimum to step over, and no
    boundary shape the answer is only approximate for. With `rotate` the angle
    comes from the partial-cell fringe's own orientation vote rather than a
    fixed 15 degree ladder.

    Returns (best, certificate). The best placement is re-evaluated once with
    the taxonomy on so the certificate and the vote diagnostics can be read off
    it; the search itself never pays for classification.
    """
    if rotate:
        best, _ = packer.optimize_guided()
    else:
        best, _ = packer.optimize_erosion()

    classified = packer.evaluate(best.dx, best.dy, best.angle, classify=True)
    classified.rotation_vote = best.rotation_vote
    return classified, packer.certificate(classified)


def _polygon_coords(poly: Polygon) -> List[Point]:
    return [(float(x), float(y)) for x, y in poly.exterior.coords[:-1]]


def _placement_to_result(packer: GridPacker, best, certificate=None) -> dict:
    """Serialise a placement for the API.

    The response SHAPE is unchanged -- same keys, same geometry -- and the new
    diagnostics are additive fields inside `stats`, so existing clients keep
    working while the paper's figures read the certificate straight off the
    endpoint.
    """
    stats = {
        "complete": best.complete,
        "partial": best.partial,
        "coverage": best.coverage,
        "dx": best.dx,
        "dy": best.dy,
        "angle": best.angle,
    }

    vote = best.rotation_vote
    if vote is not None:
        # How confident the fringe was about the angle, and what it cost. R
        # below the gate means the grid deliberately stayed put.
        stats["resultant"] = vote.resultant
        stats["rotated"] = vote.confident()
        stats["evaluations"] = vote.evaluations

    if certificate is not None:
        # How far from optimal this result could possibly be. `optimality_gap`
        # of 0 means no placement of this grid on this region has fewer
        # partials; `certified` False means the floor's assumption did not hold
        # on this instance and the gap should not be quoted.
        stats["irreducible"] = certificate.irreducible
        stats["partial_floor"] = certificate.floor
        stats["optimality_gap"] = certificate.gap
        stats["certified"] = certificate.certified
        stats["recoverable_area"] = certificate.recoverable_area

    return {
        "shape": _polygon_coords(packer.shape),
        "obstacles": [_polygon_coords(o) for o in packer.obstacles],
        "complete_cells": [_polygon_coords(c) for c in best.complete_cells],
        "partial_cells": [_polygon_coords(c) for c in best.partial_cells],
        "stats": stats,
    }


def _to_polygon(points: Sequence[Point], label: str) -> Polygon:
    if len(points) < 3:
        raise ValueError(f"{label} must have at least 3 points")
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        raise ValueError(f"{label} polygon is invalid or self-intersecting")
    return poly


def run_packing(
    shape_points: Sequence[Point],
    obstacle_points: Sequence[Sequence[Point]],
    cell_width: float,
    cell_height: float,
    rotate: bool,
) -> dict:
    """Pack a manually specified polygon. Raises ValueError on bad input."""
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("cell dimensions must be positive")

    shape = _to_polygon(shape_points, "shape")
    obstacles = [_to_polygon(pts, "obstacle") for pts in obstacle_points]

    packer = GridPacker(shape, obstacles, cell_width=cell_width, cell_height=cell_height)
    best, certificate = _solve(packer, rotate)
    return _placement_to_result(packer, best, certificate)


def run_packing_from_image(
    image_bytes: bytes,
    cell_width: float,
    cell_height: float,
    rotate: bool,
) -> dict:
    """Pack the boundary detected in raw image bytes. Raises ValueError on bad input."""
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("cell dimensions must be positive")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not read image: unsupported or corrupt file")

    packer = GridPacker.from_image(img, cell_width=cell_width, cell_height=cell_height)
    best, certificate = _solve(packer, rotate)
    return _placement_to_result(packer, best, certificate)
