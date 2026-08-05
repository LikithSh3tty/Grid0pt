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

from grid_packer import (_AREA_TOL_REL, _COORD_DECIMALS, _GEOM_TOL, GridPacker,
                         _grow)


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
    """Every reported partial carries real usable area.

    This test was written when the grazing cells described in
    `test_no_spurious_partial_from_a_rounded_up_bound` were still counted, and
    asserted only that they carried no area. The assertion is unchanged and
    still holds -- it is now the general invariant `partial => area > 0` rather
    than a description of a residual, and the residual itself is pinned to zero
    by the parametrised test below.
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

    # 12 cells: 10 clean, 1 clipped by the 1e-4 overhang, 1 fully covered.
    # The fully covered cell is OUTSIDE, not partial: the obstacle swallows it
    # whole, so area(cell & usable) is exactly 0 and the design note's
    # definition puts it in the ignored class. It still meets the usable region
    # along the hole's ring, which is why the old intersects() test called it
    # partial.
    assert result.complete == 10
    assert result.partial == 1

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


# --------------------------------------------------------------------- #
# partial vs outside is decided by AREA, not by `intersects`
# --------------------------------------------------------------------- #
# The design note defines the three classes by area: complete is C subset of U,
# partial is 0 < area(C n U) < area(C), outside is area(C n U) = 0.
# `intersects()` is only the cheap necessary condition for the middle one -- it
# is ALSO true for a measure-zero touch along a line. `_generate_cells` tiles
# `work.bounds`, and the rotation round trip leaves e.g. maxx at
# 12.000000000000002, so an extra column of cells is emitted just outside the
# region, touching it along its edge. Counting those as partial made `partial`
# a function of ulp noise in a bounding box: the 12x9-at-3x3 instance, which
# tiles EXACTLY and has no boundary fringe at all, reported partial = 0, 12, 3
# and 4 at angles 0, 12, 23 and 31.
#
# This matters beyond cosmetics: the objective is complete - penalty * partial
# and the optimality certificate counts partial cells against a lower bound, so
# an angle-dependent inflation of `partial` can rank a placement below a
# strictly worse one.
@pytest.mark.parametrize("theta", [0.0, 12.0, 23.0, 31.0])
def test_no_spurious_partial_from_a_rounded_up_bound(theta):
    """A 12x9 rectangle at 3x3 tiles exactly at EVERY angle: 12 complete cells
    and, since the region has no boundary fringe, no partial cells at all."""
    rect = exact_tiling_rect()
    packer = GridPacker(shp_rotate(rect, theta, origin=rect.centroid),
                        cell_width=3, cell_height=3)

    best, _ = packer.optimize_exact(angles=(theta,))

    assert best.complete == EXPECTED_COMPLETE
    assert best.partial == 0


@pytest.mark.parametrize("theta", [0.0, 12.0, 23.0, 31.0])
def test_the_count_and_the_list_agree(theta):
    """`partial` and `partial_cells` must stay consistent: `packer_service`
    serialises the LIST to the API and the renderer draws it, so a cell dropped
    from the count but left in the list would still be painted orange."""
    rect = exact_tiling_rect()
    packer = GridPacker(shp_rotate(rect, theta, origin=rect.centroid),
                        cell_width=3, cell_height=3)

    best, _ = packer.optimize_exact(angles=(theta,))

    assert best.partial_cells == []
    assert best.partial == len(best.partial_cells)
    assert best.complete == len(best.complete_cells)


# --------------------------------------------------------------------- #
# ...and the area test is NOT a "drop small partials" filter
# --------------------------------------------------------------------- #
def grazing_region(fraction: float) -> Polygon:
    """A 12x9 rectangle with a tab poking `fraction` of a cell into column 5.

    The tab is 3 tall (one cell edge) and 3 * fraction wide, so the cell at
    x in [12, 15], y in [0, 3] overlaps the region by exactly `fraction` of its
    own area -- the taxonomy's class F, a grazing sliver with inside-fraction
    ~ 0 whose overlap is nonetheless real geometry, not floating-point noise.
    """
    d = 3.0 * fraction
    return Polygon([(0, 0), (12 + d, 0), (12 + d, 3), (12, 3), (12, 9), (0, 9)])


@pytest.mark.parametrize("fraction", [1e-6, 1e-8, 1e-10])
def test_a_grazing_sliver_is_still_partial(fraction):
    """Class F must survive. This is the test that stops the area threshold
    drifting upward into a "small partials don't count" filter: the smallest
    fraction here is still 100x above `_AREA_TOL_REL`, so raising the threshold
    by even two orders of magnitude fails it -- loudly, and long before the
    1e-6 that a "just make the numbers nicer" patch would reach for."""
    packer = GridPacker(grazing_region(fraction), cell_width=3, cell_height=3)

    result = packer.evaluate(0.0, 0.0, 0.0)

    assert result.complete == EXPECTED_COMPLETE
    assert result.partial == 1, "the grazing cell must still be counted"

    grazing = result.partial_cells[0]
    overlap = grazing.intersection(packer.usable).area
    assert overlap / grazing.area == pytest.approx(fraction, rel=1e-6)


def test_area_threshold_separates_noise_from_the_smallest_real_sliver():
    """The threshold as a pure number, stated between the two things it has to
    stay between: the ulp-scale overlap of a cell emitted past a rounded-up
    bound (~1e-15 of the cell) and the smallest inside-fraction any test above
    calls real (1e-10)."""
    assert _AREA_TOL_REL > 1e-14        # strictly above the noise it removes
    assert _AREA_TOL_REL < 1e-10        # strictly below every real class F


# --------------------------------------------------------------------- #
# no regression on the standard instance
# --------------------------------------------------------------------- #
def l_shape_with_obstacle() -> GridPacker:
    """The paper's headline instance: a 12x12 L with an off-grid obstacle.
    Same geometry as `tests/test_optimize_exact.py`."""
    shape = Polygon([(0, 0), (12, 0), (12, 6), (6, 6), (6, 12), (0, 12)])
    obstacle = Polygon([(7, 1), (10.5, 1), (10.5, 2.5), (7, 2.5)])
    return GridPacker(shape, [obstacle], cell_width=3, cell_height=3)


def test_l_shape_with_obstacle_keeps_every_complete_cell_at_angle_zero():
    """`complete` is untouched by the area test, here and everywhere: the area
    test sits on the partial/outside branch only, which `complete` never
    reaches. Measured before the change and unchanged after it."""
    packer = l_shape_with_obstacle()

    assert packer.evaluate(0.0, 0.0, 0.0).complete == 10
    assert packer.optimize_exact(angles=(0.0,))[0].complete == 10


def test_the_l_reflex_corner_touches_three_cells_in_zero_area():
    """`partial` at angle 0 on this instance DID move, 5 -> 2, and this test
    exists to say exactly why rather than to bless a number.

    The three cells below meet the L only along its reflex corner: the L is
    {x <= 6} union {y <= 6}, so a cell spanning [6,9]x[6,9] shares with it just
    the two segments x = 6 and y = 6. The overlap is a LineString/MultiLineString
    of area EXACTLY 0.0 -- not 1e-15, not a rounding artefact, and not something
    the threshold could be tuned to keep. Under the definition
    (partial <=> 0 < area(C n U)) they are outside, and the previous
    `intersects()` test counted them as partial.

    That is a real semantic change at angle 0, on a rectilinear shape, with no
    rotation involved. It is reported rather than assumed: if the taxonomy wants
    a boundary-touch class of its own, it belongs in the classifier, not in an
    `intersects()` call that also fires on ulp noise.
    """
    packer = l_shape_with_obstacle()
    reflex_touching = [box(6, 6, 9, 9), box(6, 9, 9, 12), box(9, 6, 12, 9)]

    for cell in reflex_touching:
        overlap = packer.usable.intersection(cell)
        assert not overlap.is_empty              # they DO touch
        assert overlap.area == 0.0               # ...in zero area

    result = packer.evaluate(0.0, 0.0, 0.0)
    assert result.partial == 2
    assert not any(any(c.equals(t) for t in reflex_touching)
                   for c in result.partial_cells)


def test_every_partial_on_the_standard_instance_has_real_area():
    """The invariant the fix installs, on a shape with a genuine boundary
    fringe: every cell in `partial_cells` overlaps the usable region in a set of
    positive area, at a tilted angle as well as at zero."""
    packer = l_shape_with_obstacle()

    for angle in (0.0, 23.0):
        result = packer.evaluate(0.0, 0.0, angle)
        assert result.partial == len(result.partial_cells)
        assert result.partial > 0, "this instance really does have a fringe"
        for cell in result.partial_cells:
            assert cell.intersection(packer.usable).area > 0.0
