"""
grid_packer.py
==============

Lay a regular grid over an arbitrary shape that contains obstacles, and find the
grid placement that yields the MOST complete cells (and the fewest partial ones).

Definitions
-----------
- shape      : the outer boundary you are allowed to fill (any polygon, not just
               a rectangle: L-shapes, circles approximated as polygons, etc.)
- obstacles  : holes / blocked areas inside the shape that the grid must avoid.
- usable     : shape minus obstacles. This is the region a cell must sit inside.
- complete   : a grid cell lying ENTIRELY inside the usable region.
- partial    : a grid cell that overlaps the usable region but pokes outside it
               (crosses the boundary) or clips an obstacle.
- outside    : a cell with no usable area at all -> ignored.

The grid can slide in every direction (the "up / down / left / right / center"
shift) and optionally rotate, because moving the grid by a fraction of a cell
changes how many cells fall cleanly inside the shape. We sweep those placements
and keep the best one.

Requires: shapely, matplotlib, numpy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep


@dataclass
class Placement:
    """Result of evaluating one grid placement (offset + angle)."""
    dx: float
    dy: float
    angle: float
    complete: int
    partial: int
    complete_cells: List[Polygon] = field(default_factory=list)
    partial_cells: List[Polygon] = field(default_factory=list)
    usable_area: float = 0.0

    @property
    def complete_area(self) -> float:
        return sum(c.area for c in self.complete_cells)

    @property
    def coverage(self) -> float:
        """Fraction of the usable region captured by COMPLETE cells (0..1)."""
        return self.complete_area / self.usable_area if self.usable_area else 0.0

    def __repr__(self) -> str:
        return (f"Placement(dx={self.dx:.3f}, dy={self.dy:.3f}, "
                f"angle={self.angle:.1f}, complete={self.complete}, "
                f"partial={self.partial}, coverage={self.coverage:.1%})")


class GridPacker:
    def __init__(
        self,
        shape: Polygon,
        obstacles: Optional[Sequence[Polygon]] = None,
        cell_width: float = 1.0,
        cell_height: float = 1.0,
    ):
        if not isinstance(shape, Polygon):
            raise TypeError("shape must be a shapely Polygon")
        if cell_width <= 0 or cell_height <= 0:
            raise ValueError("cell dimensions must be positive")

        self.shape = shape
        self.obstacles = list(obstacles) if obstacles else []
        self.cw = float(cell_width)
        self.ch = float(cell_height)

        # usable region = shape with the obstacles punched out
        if self.obstacles:
            self.usable = shape.difference(unary_union(self.obstacles))
        else:
            self.usable = shape

        # pivot used for any rotation so results are reproducible
        self._pivot = self.shape.centroid

    @classmethod
    def from_image(
        cls,
        image,
        *,
        cell_width: float = 1.0,
        cell_height: float = 1.0,
        simplify_tol: float = 2.0,
        min_area: float = 64.0,
        scale: float = 1.0,
    ) -> "GridPacker":
        """Build a GridPacker from an image (file path or numpy array).

        The outer boundary detected in the image becomes the shape; enclosed
        interior regions become obstacles. Coordinates are in pixels (y-up)
        unless `scale` (units per pixel) is given. See
        image_boundary.polygons_from_image for the detection parameters.
        """
        from image_boundary import polygons_from_image

        shape, obstacles = polygons_from_image(
            image, simplify_tol=simplify_tol, min_area=min_area, scale=scale)
        return cls(shape, obstacles,
                   cell_width=cell_width, cell_height=cell_height)

    # ------------------------------------------------------------------ #
    # cell generation + classification
    # ------------------------------------------------------------------ #
    def _generate_cells(self, region: Polygon, dx: float, dy: float) -> List[Polygon]:
        """Tile `region`'s bounding box with cells whose grid lines are shifted
        by (dx, dy). Only cells whose bbox touches the region are kept."""
        minx, miny, maxx, maxy = region.bounds
        w, h = self.cw, self.ch

        # snap the first grid line so that the lattice passes through the offset
        x0 = math.floor((minx - dx) / w) * w + dx
        y0 = math.floor((miny - dy) / h) * h + dy

        cells = []
        x = x0
        while x < maxx:
            y = y0
            while y < maxy:
                cells.append(box(x, y, x + w, y + h))
                y += h
            x += w
        return cells

    def evaluate(self, dx: float = 0.0, dy: float = 0.0, angle: float = 0.0) -> Placement:
        """Classify every cell for one placement.

        Rotation trick: instead of rotating the grid, we rotate the usable
        region by -angle, run an axis-aligned analysis, then rotate the
        resulting cells back by +angle so they line up with the original shape.
        """
        if angle:
            work = rotate(self.usable, -angle, origin=self._pivot)
        else:
            work = self.usable

        prepared = prep(work)
        cells = self._generate_cells(work, dx, dy)

        complete, partial = [], []
        for c in cells:
            if prepared.contains(c):        # wholly inside usable -> complete
                complete.append(c)
            elif prepared.intersects(c):    # straddles boundary/obstacle -> partial
                partial.append(c)
            # else: no usable area -> ignore

        if angle:  # rotate cells back into the original frame for display
            complete = [rotate(c, angle, origin=self._pivot) for c in complete]
            partial = [rotate(c, angle, origin=self._pivot) for c in partial]

        return Placement(
            dx=dx, dy=dy, angle=angle,
            complete=len(complete), partial=len(partial),
            complete_cells=complete, partial_cells=partial,
            usable_area=self.usable.area,
        )

    # ------------------------------------------------------------------ #
    # search for the best placement
    # ------------------------------------------------------------------ #
    def _vertex_offsets(self) -> Tuple[List[float], List[float]]:
        """Offsets that snap grid lines onto the shape's / obstacles' own edges.
        These alignment points are where the optimum almost always sits, and a
        uniform sweep can step right over them, so we test them explicitly."""
        geoms = [self.shape.exterior] + [ob.exterior for ob in self.obstacles]
        xs, ys = set(), set()
        for ring in geoms:
            for x, y in ring.coords:
                xs.add(round(x % self.cw, 9))
                ys.add(round(y % self.ch, 9))
        return sorted(xs), sorted(ys)

    def optimize(
        self,
        steps: int = 12,
        angles: Sequence[float] = (0.0,),
        partial_penalty: float = 0.0,
        snap_to_edges: bool = True,
    ) -> Tuple[Placement, List[Placement]]:
        """Sweep grid offsets (and angles) and return the best placement.

        steps           : offsets tested per axis within one cell period.
                          The grid is periodic, so we only scan dx in [0, cw),
                          dy in [0, ch). Higher = finer search, slower.
        angles          : rotations to try, in degrees. Use e.g. range(0, 90, 15)
                          to let the grid rotate as well as slide.
        partial_penalty : objective is  complete - partial_penalty * partial.
                          0 -> maximize complete, break ties by fewer partials.
        snap_to_edges   : also test offsets that align grid lines with shape and
                          obstacle edges (where the optimum usually lives). Keep
                          this on; a plain uniform sweep can miss sharp optima.

        Returns (best_placement, all_placements_sorted_best_first).
        """
        dxs = list(np.linspace(0, self.cw, steps, endpoint=False))
        dys = list(np.linspace(0, self.ch, steps, endpoint=False))

        if snap_to_edges:
            vx, vy = self._vertex_offsets()
            dxs = sorted(set(dxs) | set(vx))
            dys = sorted(set(dys) | set(vy))

        results: List[Placement] = []
        for angle in angles:
            for dx in dxs:
                for dy in dys:
                    results.append(self.evaluate(dx, dy, angle))

        def score(p: Placement):
            # higher complete is better; fewer partials breaks ties
            return (p.complete - partial_penalty * p.partial, -p.partial)

        results.sort(key=score, reverse=True)
        return results[0], results

    # ------------------------------------------------------------------ #
    # visualization
    # ------------------------------------------------------------------ #
    def plot(self, placement: Placement, ax=None, title: Optional[str] = None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 7))

        def draw(poly, **kw):
            ax.add_patch(MplPolygon(np.array(poly.exterior.coords), **kw))

        # shape outline
        draw(self.shape, closed=True, fill=False, edgecolor="black", lw=2, zorder=1)

        # obstacles
        for ob in self.obstacles:
            draw(ob, closed=True, facecolor="#444", edgecolor="black",
                 alpha=0.85, zorder=4)

        # cells
        for c in placement.partial_cells:
            draw(c, closed=True, facecolor="#f4a259", edgecolor="#b56a1e",
                 alpha=0.55, lw=0.6, zorder=2)
        for c in placement.complete_cells:
            draw(c, closed=True, facecolor="#5cb85c", edgecolor="#2f6f2f",
                 alpha=0.75, lw=0.6, zorder=3)

        minx, miny, maxx, maxy = self.shape.bounds
        pad = max(maxx - minx, maxy - miny) * 0.05
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal")
        ax.grid(True, ls=":", alpha=0.3)

        if title is None:
            title = (f"complete={placement.complete}  partial={placement.partial}  "
                     f"coverage={placement.coverage:.1%}  "
                     f"(dx={placement.dx:.2f}, dy={placement.dy:.2f}, "
                     f"angle={placement.angle:.0f}°)")
        ax.set_title(title)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#5cb85c", alpha=0.75, label="complete"),
            Patch(facecolor="#f4a259", alpha=0.55, label="partial"),
            Patch(facecolor="#444", label="obstacle"),
        ], loc="upper right", fontsize=9)
        return ax
