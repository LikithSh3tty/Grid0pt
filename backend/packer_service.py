"""
packer_service.py
==================

Pure packing logic shared by the HTTP layer (server.py). Wraps GridPacker
and image_boundary so packing can be tested directly, without going through
FastAPI or HTTP.
"""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict, namedtuple
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon

from exporters import to_csv, to_dxf
from grid_packer import GridPacker

Point = Tuple[float, float]

DEFAULT_STEPS = 10
ROTATE_STEP = 15

#: Answers kept for repeat requests. Packing is deterministic and entirely CPU,
#: so the same outline asked twice is the same work done twice -- and with
#: `certify` that is tens of seconds, not one.
#:
#: 64 because the entries are results rather than references: each holds every
#: cell polygon of a placement, which for a fine grid on a large plan is the
#: largest object this service produces. A few dozen bounds the memory at
#: something a server can hold while still covering the case this exists for,
#: which is one client adjusting one shape.
CACHE_SIZE = 64


def _cache_key(shape_points, obstacle_points, cell_width, cell_height,
               rotate, certify) -> str:
    """A stable digest of everything that changes the answer.

    Coordinates are rounded before hashing, so a client that rebuilds its list
    and lands a float one ulp away still hits. The rounding is deliberately
    coarser than `_GEOM_TOL`: two outlines differing by less than that produce
    the same placement anyway, so treating them as one request is not an
    approximation.

    A ring is normalised by rotating it to start at its smallest vertex, since
    the same polygon listed from a different starting point is the same
    polygon. Direction is left alone -- Shapely normalises winding itself, and
    reversing here would merge a ring with its mirror, which is not the same
    shape.
    """
    def ring(points) -> Tuple[Tuple[float, float], ...]:
        rounded = tuple((round(float(x), 6), round(float(y), 6))
                        for x, y in points)
        if not rounded:
            return rounded
        start = min(range(len(rounded)), key=lambda i: rounded[i])
        return rounded[start:] + rounded[:start]

    payload = repr((
        ring(shape_points),
        tuple(sorted(ring(o) for o in obstacle_points)),
        round(float(cell_width), 9), round(float(cell_height), 9),
        bool(rotate), bool(certify),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_CacheInfo = namedtuple("_CacheInfo", "hits misses maxsize currsize")
_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_hits = 0
_cache_misses = 0


def _cached_result(key: str, compute):
    """Least-recently-used, keyed on the digest ALONE.

    Not `functools.lru_cache`, which keys on the whole argument tuple: the
    geometry would be part of the key, Shapely hashes a polygon by its
    coordinates, and the same outline listed from a different starting vertex
    would hash differently -- defeating the normalisation `_cache_key` does and
    leaving the cache useless to any client that rebuilds its coordinate list.
    """
    global _cache_hits, _cache_misses

    if key in _cache:
        _cache_hits += 1
        _cache.move_to_end(key)
        return _cache[key]

    _cache_misses += 1
    result = compute()
    _cache[key] = result
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return result


def _cache_clear() -> None:
    global _cache_hits, _cache_misses
    _cache.clear()
    _cache_hits = _cache_misses = 0


def _cache_info() -> _CacheInfo:
    return _CacheInfo(_cache_hits, _cache_misses, CACHE_SIZE, len(_cache))


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


def _solve(packer: GridPacker, rotate: bool, certify: bool = False):
    """Place the grid, and read back what the placement certifies.

    Translation is solved either way rather than sampled, and on BOTH axes: the
    offsets are not enumerated at all, they are read off the deepest overlap of
    the region eroded by the cell and folded onto the grid period. So there is
    no `steps` resolution to pick, no sharp optimum to step over, and no
    boundary shape the answer is only approximate for. With `rotate` the angle
    comes from the partial-cell fringe's own orientation vote rather than a
    fixed 15 degree ladder.

    `certify` turns the vote's answer into a proof: branch and bound over the
    angle establishes that no placement at ANY angle does better, or reports the
    gap it could not close. It is off by default because proving the angle costs
    orders of magnitude more than finding it -- tens of seconds against under
    one -- and most callers want a placement rather than a theorem. It only
    applies with `rotate`; without it there is no angle in question, and
    translation is already exact.

    Returns (best, certificate, rotation_certificate). The best placement is
    re-evaluated once with the taxonomy on so the certificate and the vote
    diagnostics can be read off it; the search itself never pays for
    classification.
    """
    rotation_certificate = None
    if rotate and certify:
        # The guided pipeline runs inside this, as the incumbent the proof is
        # built around -- so certifying cannot return a worse placement than
        # not certifying, only the same one with a proof attached.
        best, rotation_certificate = packer.certify_rotation()
    elif rotate:
        best, _ = packer.optimize_guided()
    else:
        best, _ = packer.optimize_erosion()

    classified = packer.evaluate(best.dx, best.dy, best.angle, classify=True)
    classified.rotation_vote = best.rotation_vote
    # The computed partial floor rides on the same flag: it is cheap next to
    # the rotation proof and pointless without one, since a caller who did not
    # want bounds does not want this one either.
    certificate = packer.certificate(classified, exact_floor=certify)
    return classified, certificate, rotation_certificate


def _polygon_coords(poly: Polygon) -> List[Point]:
    return [(float(x), float(y)) for x, y in poly.exterior.coords[:-1]]


def _placement_to_result(packer: GridPacker, best, certificate=None,
                         rotation_certificate=None) -> dict:
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
        if certificate.angle_floor is not None:
            # Computed rather than argued, and true only at this angle --
            # named apart from partial_floor for exactly that reason.
            stats["angle_partial_floor"] = certificate.angle_floor
            stats["angle_partial_gap"] = certificate.angle_gap

    if rotation_certificate is not None:
        # The strongest statement the solver can make: not "the best this
        # found" but "nothing at any angle does better". Absent rather than
        # false when the proof was not requested -- a client that did not ask
        # for it should not be handed a claim it cannot interpret.
        stats["rotation_bound"] = rotation_certificate.bound
        stats["rotation_gap"] = rotation_certificate.gap
        stats["rotation_optimal"] = rotation_certificate.optimal
        stats["rotation_exhausted"] = rotation_certificate.exhausted
        stats["rotation_nodes"] = rotation_certificate.nodes

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
    certify: bool = False,
) -> dict:
    """Pack a manually specified polygon. Raises ValueError on bad input."""
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("cell dimensions must be positive")

    # Validated BEFORE the cache is consulted, so a bad request raises the same
    # way every time instead of being remembered as an answer.
    shape = _to_polygon(shape_points, "shape")
    obstacles = [_to_polygon(pts, "obstacle") for pts in obstacle_points]

    key = _cache_key(shape_points, obstacle_points, cell_width, cell_height,
                     rotate, certify)
    def compute():
        packer = GridPacker(shape, obstacles, cell_width=cell_width,
                            cell_height=cell_height)
        best, certificate, rotation = _solve(packer, rotate, certify)
        return _placement_to_result(packer, best, certificate, rotation)

    result = _cached_result(key, compute)
    # Handed out as a copy: the caller gets a plain dict and may do what it
    # likes with it, and a mutation reaching the cache would poison every later
    # request for that shape.
    return copy.deepcopy(result)


#: Export formats, mapped to the writer, the media type and the extension.
#: Declared in one place so the endpoint validating a format and the exporter
#: producing it cannot disagree about which are supported.
EXPORT_FORMATS = {
    "csv": (to_csv, "text/csv", "csv"),
    "dxf": (to_dxf, "application/dxf", "dxf"),
}


def run_export(
    shape_points: Sequence[Point],
    obstacle_points: Sequence[Sequence[Point]],
    cell_width: float,
    cell_height: float,
    rotate: bool,
    fmt: str,
    certify: bool = False,
) -> Tuple[str, str, str]:
    """Pack, then serialise the result. Returns (text, media type, filename).

    Packs through `run_packing` rather than reimplementing it, so an export
    describes exactly the placement the pack endpoint returns -- and, because
    that is cached, asking for a drawing of something already on screen costs
    the serialisation only.
    """
    if fmt not in EXPORT_FORMATS:
        raise ValueError(
            f"unknown export format {fmt!r}: expected one of "
            f"{', '.join(sorted(EXPORT_FORMATS))}")

    writer, media_type, extension = EXPORT_FORMATS[fmt]
    result = run_packing(shape_points, obstacle_points, cell_width,
                         cell_height, rotate, certify)
    return writer(result), media_type, f"grid-layout.{extension}"


def run_packing_from_image(
    image_bytes: bytes,
    cell_width: float,
    cell_height: float,
    rotate: bool,
    certify: bool = False,
) -> dict:
    """Pack the boundary detected in raw image bytes. Raises ValueError on bad input."""
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("cell dimensions must be positive")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not read image: unsupported or corrupt file")

    packer = GridPacker.from_image(img, cell_width=cell_width, cell_height=cell_height)
    best, certificate, rotation = _solve(packer, rotate, certify)
    return _placement_to_result(packer, best, certificate, rotation)
