"""Tests for the per-instance optimality certificate (design note section 9).

The taxonomy splits partial cells into three optimality regimes, and regime X --
classes E and F, features smaller than a cell -- cannot be recovered by any
placement. That turns them from a failure into a certificate: they bound how
well anyone could do on this instance, so a solved placement can report how far
from optimal it could possibly be.

This module pins down:

  * the floor is a real lower bound -- asserted at ARBITRARY placements, not
    just at the solved one, because that is what "no placement can do better"
    means;
  * the correction to section 9.1. The note's floor is L / (c * s) over the full
    perimeter, which is wrong: boundary lying along grid lines is covered by the
    edges of complete cells. The simplest instance there is -- a room that tiles
    exactly, zero partials -- refutes it, and the regression is asserted here;
  * regime X is actually detected on a feature genuinely narrower than a cell;
  * the bound's own assumption is measured, not asserted, so a caller can see
    when the reported gap does not stand.
"""
import math

import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon, box

from grid_packer import GridPacker


# --------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------- #
def room(width=12.0, height=9.0, tilt=0.0) -> Polygon:
    """A plain rectangular room, optionally tilted about its own centroid.

    12x9 at 3x3 cells tiles EXACTLY: 12 complete cells, no fringe whatever. It
    is the instance that refutes the note's boundary floor, because its
    perimeter is 42 and its partial count is 0.
    """
    base = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    return shp_rotate(base, tilt, origin=base.centroid) if tilt else base


def l_shape() -> Polygon:
    return Polygon([(0, 0), (12, 0), (12, 6), (6, 6), (6, 12), (0, 12)])


def dumbbell() -> Polygon:
    """Two rooms joined by a corridor narrower than a cell.

    The corridor is 0.4 wide against a 1.5 cell, so no placement can make the
    cells covering it complete -- there is not a cell's width of region to fill
    them with. This is regime X by construction, and the only instance here that
    produces class-E cells.
    """
    return Polygon([(0, 0), (6, 0), (6, 3.3), (10, 3.3), (10, 0), (16, 0),
                    (16, 6), (10, 6), (10, 3.7), (6, 3.7), (6, 6), (0, 6)])


def solved(packer: GridPacker):
    """Solve, then re-read the winner with the taxonomy on."""
    best, _ = packer.optimize_guided()
    return packer.evaluate(best.dx, best.dy, best.angle, classify=True)


#: Instances spanning aligned, tilted, holed, non-perpendicular and curved.
CORPUS = [
    ("axis-aligned room", room(), [], 3.0, 3.0),
    ("room tilted 23", room(tilt=23), [], 3.0, 3.0),
    ("L-shape", l_shape(), [], 3.0, 3.0),
    ("L + obstacle", l_shape(), [box(7, 1, 10.5, 2.5)], 3.0, 3.0),
    ("dumbbell", dumbbell(), [], 1.5, 1.5),
    ("16-gon", Polygon([(6 * math.cos(2 * math.pi * i / 16),
                        6 * math.sin(2 * math.pi * i / 16))
                       for i in range(16)]), [], 3.0, 3.0),
]


# --------------------------------------------------------------------- #
# the floor is a lower bound
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name,shape,obstacles,cw,ch", CORPUS)
def test_the_floor_never_exceeds_the_partial_count_at_any_placement(
        name, shape, obstacles, cw, ch):
    """The bound is checked at ARBITRARY placements, not the solved one.

    "No placement can have fewer partials than the floor" is a claim about every
    placement, so a floor that only held at the optimum would be worthless --
    and would hide exactly the failure mode the note's version has.
    """
    packer = GridPacker(shape, obstacles, cell_width=cw, cell_height=ch)

    for dx, dy, angle in [(0.0, 0.0, 0.0), (0.3, 0.7, 0.0),
                          (0.0, 0.0, 17.0), (1.1, 0.4, 41.0)]:
        placement = packer.evaluate(dx, dy, angle, classify=True)
        certificate = packer.certificate(placement)

        assert certificate.floor <= placement.partial, (
            f"{name} at ({dx}, {dy}, {angle}): floor {certificate.floor} "
            f"exceeds the {placement.partial} partials actually there")
        assert certificate.gap >= 0


@pytest.mark.parametrize("name,shape,obstacles,cw,ch", CORPUS)
def test_the_floor_is_never_below_the_irreducible_count(
        name, shape, obstacles, cw, ch):
    """Regime-X cells are partial under every placement, so they alone are a
    floor; combining them with the covering bound must not weaken it."""
    packer = GridPacker(shape, obstacles, cell_width=cw, cell_height=ch)
    placement = solved(packer)

    certificate = packer.certificate(placement)

    assert certificate.floor >= certificate.irreducible


# --------------------------------------------------------------------- #
# the correction to section 9.1
# --------------------------------------------------------------------- #
def test_an_exactly_tiling_room_has_a_floor_of_zero():
    """The instance that refutes the note's boundary floor.

    Section 9.1 bounds the partial count by L / (c * s), reasoning that "the
    region boundary of length L must be covered by partial cells". A 12x9 room
    at 3x3 cells has perimeter 42 and a cell diagonal of 4.24, so that formula
    demands at least 10 partial cells. The room tiles exactly into 12 complete
    cells and ZERO partials: its boundary runs along grid lines, where it is
    covered by the edges of COMPLETE cells.

    A floor above the achieved value is not a conservative bound, it is a false
    one -- it would report a negative optimality gap and certify a perfect
    placement as improvable.
    """
    packer = GridPacker(room(), cell_width=3, cell_height=3)
    placement = packer.evaluate(0.0, 0.0, 0.0, classify=True)

    certificate = packer.certificate(placement)

    assert placement.complete == 12
    assert placement.partial == 0

    # What the note's formula would have demanded, for the record.
    assert certificate.boundary_length == pytest.approx(42.0)
    assert certificate.boundary_length / certificate.cell_diagonal > 9

    # What is actually forced: nothing. Every wall can lie on a grid line.
    assert certificate.forced_length == pytest.approx(0.0)
    assert certificate.floor == 0
    assert certificate.gap == 0


@pytest.mark.parametrize("tilt", [12.0, 23.0, 31.0])
def test_a_tilted_rooms_two_wall_families_are_one_direction(tilt):
    """A rectangle's edges run at `tilt` and `tilt + 90`, which are the SAME
    direction modulo the alignment period -- both can lie on grid lines at once,
    one along the cell width and one along its height. So a tilted room forces
    no boundary either, and a grid turned onto it tiles exactly.

    This is also the regression for the binning: fixed-width orientation bins
    put an arbitrary edge somewhere, and a wall family landing on one gets split
    in two, reporting half the perimeter as forced on a room with a zero floor.
    """
    packer = GridPacker(room(tilt=tilt), cell_width=3, cell_height=3)
    placement = solved(packer)

    certificate = packer.certificate(placement)

    assert certificate.forced_length == pytest.approx(0.0, abs=1e-6)
    assert certificate.best_direction == pytest.approx(tilt, abs=0.5)


def test_non_perpendicular_walls_do_force_boundary():
    """The bound is not vacuous: a parallelogram has two wall families that are
    NOT 90 degrees apart, so no grid angle can lay both on grid lines and the
    forced length is genuinely positive."""
    side = 12.0
    dx = side * math.cos(math.radians(35))
    dy = side * math.sin(math.radians(35))
    rhombus = Polygon([(0, 0), (side, 0), (side + dx, dy), (dx, dy)])

    packer = GridPacker(rhombus, cell_width=2.5, cell_height=2.5)
    placement = solved(packer)

    certificate = packer.certificate(placement)

    assert certificate.forced_length == pytest.approx(24.0, abs=0.5)
    assert certificate.floor >= 7
    assert certificate.gap >= 0


# --------------------------------------------------------------------- #
# regime X
# --------------------------------------------------------------------- #
def test_a_corridor_narrower_than_a_cell_is_certified_irreducible():
    """The certificate's core claim, on a feature that genuinely cannot be
    recovered: a 0.4-wide corridor against a 1.5 cell. No offset and no angle
    puts a whole cell inside it, so those cells are partial under every
    placement -- which is a bound, not a failure."""
    packer = GridPacker(dumbbell(), cell_width=1.5, cell_height=1.5)
    placement = solved(packer)

    certificate = packer.certificate(placement)

    irreducible = [c for c in placement.partial_classes if not c.recoverable]

    assert certificate.irreducible >= 3
    assert certificate.irreducible == len(irreducible)
    assert all(c.cut_count >= 2 for c in irreducible)   # a band, class E
    assert certificate.floor >= certificate.irreducible


def test_an_exact_tiling_leaves_nothing_to_reclaim():
    """`recoverable_area` is the stop criterion of section 8.4: zero means there
    is no upside left, so no reason to keep turning."""
    packer = GridPacker(room(), cell_width=3, cell_height=3)
    exact = packer.evaluate(0.0, 0.0, 0.0, classify=True)
    offset = packer.evaluate(1.4, 0.7, 0.0, classify=True)

    assert packer.certificate(exact).recoverable_area == pytest.approx(0.0)
    assert packer.certificate(offset).recoverable_area > 0.0


# --------------------------------------------------------------------- #
# the bound's assumption is measured, not asserted
# --------------------------------------------------------------------- #
def test_a_cell_holding_more_boundary_than_its_diagonal_is_not_certified():
    """The covering bound assumes each cell carries one straight crossing, so a
    cell can hold at most a diagonal's worth of boundary. A boundary that wiggles
    inside a cell breaks that, which would make the floor too high and the gap
    too flattering -- so it is measured and reported rather than assumed.
    """
    packer = GridPacker(l_shape(), [box(7, 1, 10.5, 2.5)],
                        cell_width=3, cell_height=3)
    placement = solved(packer)

    certificate = packer.certificate(placement)

    # This instance has an obstacle whose whole outline sits inside one cell.
    assert certificate.observed_max_chord > certificate.cell_diagonal
    assert not certificate.certified

    # The floor still holds here, but the caller is told not to lean on it.
    assert certificate.gap >= 0


def test_a_clean_instance_is_certified():
    packer = GridPacker(room(tilt=23), cell_width=3, cell_height=3)

    certificate = packer.certificate(solved(packer))

    assert certificate.observed_max_chord <= certificate.cell_diagonal
    assert certificate.certified


def test_the_certificate_needs_the_taxonomy():
    """Reading it off an unclassified placement would silently report an empty
    regime X and a floor that ignores it.

    The offset placement is used deliberately: the L-shape at (0, 0) has no
    partial cells at all, and a placement with nothing to classify IS fully
    classified, so it is allowed through rather than refused.
    """
    packer = GridPacker(l_shape(), cell_width=3, cell_height=3)
    unclassified = packer.evaluate(1.4, 0.7, 0.0)

    assert unclassified.partial > 0
    with pytest.raises(ValueError, match="classify"):
        packer.certificate(unclassified)

    vacuous = packer.evaluate(0.0, 0.0, 0.0)
    assert vacuous.partial == 0
    assert packer.certificate(vacuous).floor == 0
