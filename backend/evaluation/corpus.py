"""The evaluation corpus (design note section 11).

The note asks for "synthetic rectangles / L-shapes / rotated shapes with known
answers; a set of real floor plans through image_boundary.py; vary cell aspect
ratio and cell size", and to "move beyond the three toy shapes the code
currently ships with".

Everything here is generated from code, seeded where randomness is involved. The
corpus is therefore released by releasing this module -- no data files to lose,
and a reader reproduces the exact instances by importing it.

KNOWN ANSWERS. Some families come with a proven optimum rather than a measured
one, which is what makes them worth having: a method's result can be compared
against the truth instead of against another method. Two constructions give one.

  Exact tilings. A w x h room whose sides are integer multiples of the cell
  tiles perfectly, so the optimum is exactly (w/cw) * (h/ch) complete cells and
  zero partials -- attained by the placement that flushes the grid to a corner.

  Tilted exact tilings. Rotating that room rigidly does not change what a grid
  can do to it: turn the grid by the same angle and the tiling is recovered. So
  a room tilted by theta has the SAME known optimum, and any method that fails
  to reach it has failed to find the rotation, which is precisely the claim
  section 8 makes. This is the corpus's sharpest instrument.

Instances without a known answer (curved boundaries, random plans, traced
images) still carry the certificate floor, which bounds them from the other
side.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon

from grid_packer import GridPacker


@dataclass(frozen=True)
class Instance:
    """One packing problem, plus whatever is known about its answer.

    name          : unique, stable, and descriptive enough to read in a results
                    table without consulting this file.
    family        : the group it belongs to, for per-family aggregation.
    known_optimum : complete cells at the true optimum, where it is PROVEN by
                    construction (see the module docstring). None otherwise --
                    never a guess, because the point of the column is that it
                    can be trusted.
    """

    name: str
    family: str
    shape: Polygon
    obstacles: Tuple[Polygon, ...]
    cell_width: float
    cell_height: float
    known_optimum: Optional[int] = None
    notes: str = ""

    def packer(self) -> GridPacker:
        """A fresh packer. Fresh matters: the evaluation counter is per-packer,
        so reusing one across methods would attribute one method's cost to the
        next."""
        return GridPacker(self.shape, list(self.obstacles),
                          cell_width=self.cell_width,
                          cell_height=self.cell_height)

    @property
    def cell_count_bound(self) -> int:
        """Cells that would fit if area alone decided -- the loosest upper
        bound there is, useful for sanity-checking a result table."""
        area = self.shape.area - sum(o.area for o in self.obstacles)
        return int(area // (self.cell_width * self.cell_height))


# --------------------------------------------------------------------------- #
# shape generators
# --------------------------------------------------------------------------- #

def rectangle(width: float, height: float, origin=(0.0, 0.0)) -> Polygon:
    ox, oy = origin
    return Polygon([(ox, oy), (ox + width, oy),
                    (ox + width, oy + height), (ox, oy + height)])


def l_shape(width: float, height: float, cut_w: float, cut_h: float) -> Polygon:
    """A rectangle with its top-right corner removed."""
    return Polygon([(0, 0), (width, 0), (width, height - cut_h),
                    (width - cut_w, height - cut_h), (width - cut_w, height),
                    (0, height)])


def u_shape(width: float, height: float, notch_w: float, notch_h: float) -> Polygon:
    """A rectangle with a notch cut into the top edge -- two reflex corners, so
    two C2 cells that translation must snap simultaneously."""
    x0 = (width - notch_w) / 2.0
    return Polygon([(0, 0), (width, 0), (width, height),
                    (x0 + notch_w, height), (x0 + notch_w, height - notch_h),
                    (x0, height - notch_h), (x0, height), (0, height)])


def plus_shape(arm: float, thickness: float) -> Polygon:
    """A cross: four reflex corners and no dominant wall family longer than the
    others, which is where the vote has the least to work with."""
    a, t = arm, thickness / 2.0
    return Polygon([(-t, -a), (t, -a), (t, -t), (a, -t), (a, t), (t, t),
                    (t, a), (-t, a), (-t, t), (-a, t), (-a, -t), (-t, -t)])


def random_rectilinear(seed: int, cells_x: int = 7, cells_y: int = 5,
                       unit: float = 3.0, fill: float = 0.62) -> Polygon:
    """A seeded rectilinear plan: a connected blob of unit squares.

    Grown by a random walk from the centre so the result is always connected,
    then unioned. The boundary is axis-aligned and full of reflex corners --
    the regime-T workload -- and unlike the hand-made toys there are many of
    them, which is the point of section 11's "move beyond the three toy shapes".
    """
    rng = random.Random(seed)
    target = max(4, int(cells_x * cells_y * fill))
    cx, cy = cells_x // 2, cells_y // 2
    taken = {(cx, cy)}
    x, y = cx, cy
    while len(taken) < target:
        x = min(cells_x - 1, max(0, x + rng.choice((-1, 0, 1))))
        y = min(cells_y - 1, max(0, y + rng.choice((-1, 0, 1))))
        taken.add((x, y))
        if rng.random() < 0.15:                 # occasionally teleport back
            x, y = cx, cy

    from shapely.ops import unary_union
    blob = unary_union([rectangle(unit, unit, (i * unit, j * unit))
                        for i, j in taken])
    # A random walk can pinch to a point; buffer(0) repairs that into a valid
    # polygon, and taking the largest part keeps it single-bodied.
    blob = blob.buffer(0)
    if blob.geom_type == "MultiPolygon":
        blob = max(blob.geoms, key=lambda g: g.area)
    # Only the outer wall is kept. A walk that encircles a gap would otherwise
    # produce a hole, and a hole here would be indistinguishable from the
    # obstacle family, which is measured separately and on purpose.
    return Polygon(blob.exterior)


def traced(shape: Polygon, *, scale: float = 0.1, margin: int = 20,
           obstacles: Sequence[Polygon] = ()) -> Tuple[Polygon, List[Polygon]]:
    """Push a polygon through the image pipeline and read back what comes out.

    Rasterise, then recover the boundary with `image_boundary` exactly as the
    /pack-image endpoint does. What returns is not the polygon that went in: it
    is quantised to the pixel grid, closed by the morphological step and
    decimated by Douglas-Peucker, so its walls carry the degree or two of
    orientation scatter that a real traced floor plan carries.

    That is the point. The note asks for "real floor plans through
    image_boundary.py"; generating them this way keeps the corpus reproducible
    with no binary assets, while still exercising the noise that the vote's bin
    width and the certificate's alignment tolerance exist to absorb.
    """
    minx, miny, maxx, maxy = shape.bounds
    w = int((maxx - minx) / scale) + 2 * margin
    h = int((maxy - miny) / scale) + 2 * margin
    img = np.zeros((h, w), np.uint8)

    def to_px(poly: Polygon) -> np.ndarray:
        pts = [((x - minx) / scale + margin, (y - miny) / scale + margin)
               for x, y in poly.exterior.coords]
        # Image rows run downward; flip y so the traced result comes back the
        # same way up as it went in.
        return np.array([[int(round(px)), int(round(h - 1 - py))]
                         for px, py in pts], np.int32)

    cv2.fillPoly(img, [to_px(shape)], 255)
    for obstacle in obstacles:
        cv2.fillPoly(img, [to_px(obstacle)], 0)

    from image_boundary import polygons_from_image
    return polygons_from_image(img, simplify_tol=2.0, min_area=64.0, scale=scale)


# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #

TILTS: Tuple[float, ...] = (7.0, 12.0, 23.0, 31.0, 44.0)

#: Cell geometries the corpus sweeps. Aspect matters because the rotation period
#: is 90 degrees for square cells and 180 for rectangular ones (note section 1),
#: and size matters because it drives the regime-X floor: halve the cell and a
#: feature that was sub-cell stops being sub-cell.
CELLS: Tuple[Tuple[float, float], ...] = ((3.0, 3.0), (2.0, 3.0), (1.5, 1.5))


def _exact_tiling_optimum(width: float, height: float,
                          cw: float, ch: float) -> Optional[int]:
    """Complete cells at the optimum, when the room tiles exactly -- else None."""
    nx, ny = width / cw, height / ch
    if abs(nx - round(nx)) > 1e-9 or abs(ny - round(ny)) > 1e-9:
        return None
    return int(round(nx)) * int(round(ny))


def build(quick: bool = True) -> List[Instance]:
    """The corpus. `quick` keeps the families but trims the sweeps.

    Quick is the default everywhere because the dense reference method is
    quadratic in the offset resolution and cubic once angles are added; the full
    corpus is for the paper run, not for a change you want to check.
    """
    tilts = (12.0, 23.0) if quick else TILTS
    cells = CELLS[:1] if quick else CELLS
    seeds = range(2) if quick else range(6)

    out: List[Instance] = []

    for cw, ch in cells:
        tag = f"{cw:g}x{ch:g}"

        # --- rectangles, exact tilings: the known-answer backbone ----------- #
        room_w, room_h = 12 * cw, 9 * ch
        out.append(Instance(
            name=f"room-aligned-{tag}", family="rectangle",
            shape=rectangle(room_w, room_h), obstacles=(),
            cell_width=cw, cell_height=ch,
            known_optimum=_exact_tiling_optimum(room_w, room_h, cw, ch),
            notes="tiles exactly; optimum is a flush corner"))

        # Offset from the origin so a grid pinned at (0,0) is wrong: the
        # translation problem in its purest form.
        out.append(Instance(
            name=f"room-offset-{tag}", family="rectangle",
            shape=rectangle(room_w, room_h, (0.37 * cw, 0.61 * ch)),
            obstacles=(), cell_width=cw, cell_height=ch,
            known_optimum=_exact_tiling_optimum(room_w, room_h, cw, ch),
            notes="same room, shifted off the lattice"))

        for tilt in tilts:
            out.append(Instance(
                name=f"room-tilt{tilt:g}-{tag}", family="rotated",
                shape=shp_rotate(rectangle(room_w, room_h), tilt),
                obstacles=(), cell_width=cw, cell_height=ch,
                known_optimum=_exact_tiling_optimum(room_w, room_h, cw, ch),
                notes="rigid rotation of an exact tiling: same optimum"))

        # --- rectilinear shapes: reflex corners, still regime T ------------- #
        out.append(Instance(
            name=f"l-shape-{tag}", family="l_shape",
            shape=l_shape(12 * cw, 9 * ch, 5 * cw, 4 * ch), obstacles=(),
            cell_width=cw, cell_height=ch,
            known_optimum=(12 * 9 - 5 * 4),
            notes="exact tiling with a corner removed"))

        out.append(Instance(
            name=f"u-shape-{tag}", family="l_shape",
            shape=u_shape(12 * cw, 9 * ch, 4 * cw, 3 * ch), obstacles=(),
            cell_width=cw, cell_height=ch,
            known_optimum=(12 * 9 - 4 * 3),
            notes="two reflex corners to snap at once"))

        out.append(Instance(
            name=f"plus-{tag}", family="l_shape",
            shape=plus_shape(6 * cw, 4 * ch), obstacles=(),
            cell_width=cw, cell_height=ch,
            notes="four reflex corners, no dominant wall family"))

        for tilt in tilts[:1] if quick else tilts[:2]:
            out.append(Instance(
                name=f"l-shape-tilt{tilt:g}-{tag}", family="rotated",
                shape=shp_rotate(l_shape(12 * cw, 9 * ch, 5 * cw, 4 * ch), tilt),
                obstacles=(), cell_width=cw, cell_height=ch,
                known_optimum=(12 * 9 - 5 * 4),
                notes="rigid rotation of an exact tiling: same optimum"))

        # --- obstacles: the D class, and holes in the usable region --------- #
        out.append(Instance(
            name=f"room-pillars-{tag}", family="obstacles",
            shape=rectangle(12 * cw, 9 * ch),
            obstacles=(rectangle(2 * cw, 2 * ch, (3 * cw, 3 * ch)),
                       rectangle(cw, 3 * ch, (8 * cw, 2 * ch))),
            cell_width=cw, cell_height=ch,
            known_optimum=(12 * 9 - 2 * 2 - 1 * 3),
            notes="obstacles on the lattice: still an exact tiling"))

        out.append(Instance(
            name=f"room-pillars-offgrid-{tag}", family="obstacles",
            shape=rectangle(12 * cw, 9 * ch),
            obstacles=(rectangle(1.7 * cw, 2.3 * ch, (3.4 * cw, 3.1 * ch)),
                       shp_rotate(rectangle(2 * cw, cw, (7 * cw, 5 * ch)), 27),
                       ),
            cell_width=cw, cell_height=ch,
            notes="one off-lattice obstacle and one oblique one"))

        # --- curved: no wall to align to, the R gate's negative control ----- #
        # Resolution is kept low deliberately. Exact translation costs one
        # evaluation per face of the critical arrangement, i.e. quadratic in the
        # number of DISTINCT vertex coordinates, and a curve contributes a
        # vertex per segment: a 256-gon disc measured 64,517 evaluations against
        # a 32-gon's ~900. A finer circle would change what the corpus costs
        # without changing what it tests -- these instances are here because
        # they have no dominant wall, and 32 segments already have none.
        out.append(Instance(
            name=f"disc-{tag}", family="curved",
            shape=Point(0, 0).buffer(5 * cw, 8), obstacles=(),
            cell_width=cw, cell_height=ch,
            notes="no dominant orientation; rotating cannot pay"))

        out.append(Instance(
            name=f"stadium-{tag}", family="curved",
            shape=rectangle(10 * cw, 4 * ch).buffer(1.5 * cw, quad_segs=4),
            obstacles=(), cell_width=cw, cell_height=ch,
            notes="straight sides, round ends: a partial wall family"))

        # --- seeded random plans -------------------------------------------- #
        for seed in seeds:
            out.append(Instance(
                name=f"plan-seed{seed}-{tag}", family="random",
                shape=random_rectilinear(seed, unit=2 * cw), obstacles=(),
                cell_width=cw, cell_height=ch,
                notes=f"seeded rectilinear plan (seed={seed})"))

        # --- traced through the image pipeline ------------------------------ #
        for tilt in (0.0, 23.0):
            base = shp_rotate(l_shape(12 * cw, 9 * ch, 5 * cw, 4 * ch), tilt)
            shape, obstacles = traced(base, scale=cw / 12.0)
            out.append(Instance(
                name=f"traced-l{tilt:g}-{tag}", family="traced",
                shape=shape, obstacles=tuple(obstacles),
                cell_width=cw, cell_height=ch,
                notes="rasterised and re-traced: quantised, decimated walls"))

    return out


def summary(instances: Sequence[Instance]) -> str:
    """One line per family: how many instances and how many carry a known
    optimum. Printed by the driver so a run states what it ran on."""
    families = {}
    for inst in instances:
        n, known = families.get(inst.family, (0, 0))
        families[inst.family] = (n + 1, known + (inst.known_optimum is not None))
    lines = [f"  {fam:<12s} {n:3d} instances, {k:3d} with a known optimum"
             for fam, (n, k) in sorted(families.items())]
    return "\n".join(lines)
