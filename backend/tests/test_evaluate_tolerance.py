"""Tests for the geometric tolerance in `GridPacker.evaluate`.

`evaluate` implements rotation by rotating the usable region by -angle,
analysing axis-aligned, and rotating the cells back. That round trip is not
bit-exact -- a wall that should sit at x = 0.4 comes back as
0.40000000000000036 -- while `_wrap` rounds the critical offsets it feeds in to
`_COORD_DECIMALS` decimals, putting the grid line at exactly 0.4. A cell that
tiles the region exactly then falls a few ulp short of the wall, `contains` is
strictly false, and the flush placement -- the optimum -- is misreported as
partial. The bias is systematic and lands precisely on the aligned placements
at non-zero angles, i.e. on the case the rotation method exists to handle.

`evaluate` therefore tests containment against the region grown by `_GEOM_TOL`.
These tests pin down both halves of that: that the tolerance is large enough to
absorb the rounding (the recovery tests) and small enough that it never absorbs
a geometrically real overlap (the "not loose" tests).
"""
import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import MultiPolygon, Polygon, box

from grid_packer import _COORD_DECIMALS, _GEOM_TOL, GridPacker, _grow


# A 12x9 rectangle tiles exactly with 3x3 cells: 4 columns x 3 rows = 12 cells,
# zero partials. Any placement reporting fewer than 12 has lost cells to
# floating point, not to geometry.
def exact_tiling_rect() -> Polygon:
    return Polygon([(0, 0), (12, 0), (12, 9), (0, 9)])


EXPECTED_COMPLETE = 12


# --------------------------------------------------------------------- #
# the constant: one quantity, not two magic numbers
# --------------------------------------------------------------------- #
def test_tolerance_strictly_covers_the_offset_rounding():
    """`_GEOM_TOL` must exceed the worst-case displacement `_wrap` can
    introduce, or the rounding it exists to absorb can still bite."""
    worst_case_rounding = 0.5 * 10.0 ** -_COORD_DECIMALS
    assert _GEOM_TOL > worst_case_rounding


def test_tolerance_stays_far_below_any_real_geometric_resolution():
    """...and must stay negligible against the cell sizes actually in use, so
    it can never absorb an overlap anyone would call real."""
    assert _GEOM_TOL < 1e-6


# --------------------------------------------------------------------- #
# headline: a shape that tiles exactly reports every cell, at any angle
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [12.0, 23.0, 31.0])
def test_exact_tiling_is_recovered_at_a_tilted_angle(theta):
    """A 12x9 rectangle tilted by theta, evaluated at theta, is rotated back to
    axis-aligned and tiles exactly. Before the tolerance this reported 6."""
    rect = exact_tiling_rect()
    packer = GridPacker(shp_rotate(rect, theta, origin=rect.centroid),
                        cell_width=3, cell_height=3)

    best, _ = packer.optimize_exact(angles=(theta,))

    assert best.complete == EXPECTED_COMPLETE
    # The whole region is accounted for by complete cells.
    assert best.coverage == pytest.approx(1.0)


@pytest.mark.parametrize("theta", [12.0, 23.0, 31.0])
def test_partials_left_at_a_tilted_angle_carry_no_area(theta):
    """A KNOWN residual, deliberately left in scope-free: the tolerance is on
    `contains` only, so `intersects` still reports cells that merely graze the
    boundary. `_generate_cells` tiles `work.bounds`, and the rotation round trip
    leaves maxx at 12.000000000000002, so it emits an extra column outside the
    region whose cells touch it along a line. They carry ZERO usable area, so
    they inflate `partial` without affecting `complete` or `coverage`.

    This test documents the residual rather than blessing it: if `partial` is
    ever tightened, it should go to 0 here and this test should be updated.
    """
    rect = exact_tiling_rect()
    packer = GridPacker(shp_rotate(rect, theta, origin=rect.centroid),
                        cell_width=3, cell_height=3)

    best, _ = packer.optimize_exact(angles=(theta,))

    for cell in best.partial_cells:
        overlap = cell.intersection(packer.usable).area
        assert overlap < 1e-12, "a partial cell with real area would be a bug"


def test_exact_tiling_is_recovered_untilted():
    """No regression at angle 0, where the rotation round trip never runs. A
    changed count here would mean the tolerance is doing more than intended."""
    packer = GridPacker(exact_tiling_rect(), cell_width=3, cell_height=3)

    best, _ = packer.optimize_exact()

    assert best.complete == EXPECTED_COMPLETE
    assert best.partial == 0


# --------------------------------------------------------------------- #
# the tolerance is NOT a fudge factor
# --------------------------------------------------------------------- #
# Each eps leaves the region genuinely short of tiling: the last column of
# cells pokes out by eps. The smallest eps here is 10x _GEOM_TOL, so these
# assertions fail if anyone widens the tolerance by even one order of
# magnitude -- and fail loudly at the 1e-4 a "just make it work" patch reaches
# for.
@pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-6, 1e-7])
def test_a_real_overlap_is_still_partial(eps):
    """A region eps short of tiling exactly: the overhanging column must stay
    partial. The tolerance absorbs rounding, never geometry."""
    packer = GridPacker(box(0, 0, 12 - eps, 9), cell_width=3, cell_height=3)

    result = packer.evaluate(0.0, 0.0, 0.0)

    assert result.complete == 9        # 3 columns clean, the 4th pokes out
    assert result.partial == 3


def test_the_same_region_tiles_when_the_shortfall_is_removed():
    """Control for the test above: with eps = 0 the very same construction
    gives all 12, so the partials there come from the shortfall and not from
    the way the instance is built."""
    packer = GridPacker(box(0, 0, 12, 9), cell_width=3, cell_height=3)

    result = packer.evaluate(0.0, 0.0, 0.0)

    assert result.complete == EXPECTED_COMPLETE
    assert result.partial == 0


# --------------------------------------------------------------------- #
# holes: a positive buffer must SHRINK obstacles, not grow them
# --------------------------------------------------------------------- #
def test_tolerance_never_lets_a_cell_clip_an_obstacle():
    """An obstacle poking 1e-4 into an otherwise clean cell keeps that cell
    partial. If the buffer grew interior rings instead of shrinking them, the
    obstacle would retreat and the clipped cell would be called complete."""
    obstacle = box(6 - 1e-4, 3, 9, 6)          # covers one cell, clips another
    packer = GridPacker(box(0, 0, 12, 9), [obstacle], cell_width=3, cell_height=3)
    assert len(packer.usable.interiors) == 1   # the obstacle is a real hole

    result = packer.evaluate(0.0, 0.0, 0.0)

    # 12 cells: 10 clean, 1 clipped by the 1e-4 overhang, 1 fully covered
    assert result.complete == 10
    assert result.partial == 2

    clipped = box(3, 3, 6, 6)
    assert not any(c.equals(clipped) for c in result.complete_cells)


def test_growing_a_region_shrinks_its_holes():
    """The direction of the buffer on interior rings, asserted directly."""
    holed = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                    [[(3, 3), (6, 3), (6, 6), (3, 6)]])

    grown = _grow(holed)

    assert len(grown.interiors) == 1
    assert Polygon(grown.interiors[0]).area < Polygon(holed.interiors[0]).area
    assert grown.area > holed.area


# --------------------------------------------------------------------- #
# the awkward geometries `usable` can actually be
# --------------------------------------------------------------------- #
def test_tolerance_does_not_bridge_the_gap_in_a_multipolygon():
    """An obstacle that cuts the shape in two makes `usable` a MultiPolygon.
    Growing it must not weld the pieces back together, or a cell spanning the
    cut would be called complete."""
    divider = box(4.4, -1, 5.9, 10)
    packer = GridPacker(box(0, 0, 12, 9), [divider], cell_width=3, cell_height=3)
    assert isinstance(packer.usable, MultiPolygon)

    grown = _grow(packer.usable)
    assert isinstance(grown, MultiPolygon)
    assert len(grown.geoms) == len(packer.usable.geoms)

    spanning = box(3, 0, 6, 3)                 # straddles the 1.5-wide cut
    result = packer.evaluate(0.0, 0.0, 0.0)
    assert not any(c.equals(spanning) for c in result.complete_cells)


@pytest.mark.parametrize("geom", [
    Polygon(),                                  # empty
    Polygon([(0, 0), (1, 0), (1, 0), (0, 0)]),  # zero-area, repeated point
    Polygon([(0, 0), (1, 1), (2, 2), (0, 0)]),  # collinear sliver
])
def test_growing_a_degenerate_geometry_does_not_raise(geom):
    """`usable` is a `difference`, which can leave slivers behind; buffering
    one must degrade gracefully rather than blow up the whole evaluation."""
    assert _grow(geom) is not None
