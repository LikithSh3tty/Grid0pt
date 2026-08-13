"""Tests for the partial-cell floor computed rather than assumed.

The covering bound of section 9.1 is a real bound with a real assumption: it
divides the boundary that must cross cell interiors by the most one cell can
hold, taking that to be the cell's diagonal, which is true only if each cell
carries a single straight crossing. A boundary that wiggles inside one cell
breaks it, so the certificate measures the assumption and declines when it
fails -- on 9 of the full corpus's 72 instances.

It can be computed instead, from the same machinery the complete count uses. A
cell is complete when its corner lies in the region ERODED by the cell; it
overlaps the region at all when its corner lies in the region DILATED by the
cell. So

    partial(dx, dy) = (lattice points in the dilation) - (lattice points in the erosion)

and folding both modulo the lattice makes that a difference of two overlap
depths -- one function on one torus, piecewise constant, minimised exactly.

WHY THE VERTICES ARE NOT ENOUGH, which is the mistake this replaces. For the
complete count the pieces are closed, depth is upper semi-continuous, and a
MAXIMUM therefore survives at a vertex of the arrangement. A minimum does not:
sampling vertices returns a value above the true minimum, which is the wrong
side for a floor. So every cell of the arrangement is sampled -- faces by a
representative point, edges by a midpoint, vertices themselves -- since the
function is constant on each and nothing else can be assumed about where the
minimum sits.
"""
import numpy as np
import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon

from grid_packer import GridPacker, _dilate_by_cell


def room(width=12.0, height=9.0, tilt=0.0) -> Polygon:
    box = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    return shp_rotate(box, tilt) if tilt else box


def packer(shape, obstacles=(), cw=3.0, ch=3.0) -> GridPacker:
    return GridPacker(shape, list(obstacles), cell_width=cw, cell_height=ch)


def dense_min_partial(pk: GridPacker, steps: int = 40, angle: float = 0.0) -> int:
    return min(pk.evaluate(dx, dy, angle).partial
               for dx in np.linspace(0, pk.cw, steps, endpoint=False)
               for dy in np.linspace(0, pk.ch, steps, endpoint=False))


# --------------------------------------------------------------------------- #
# the dilation: which corners put a cell in touch with the region at all
# --------------------------------------------------------------------------- #

def test_dilating_by_the_cell_reaches_one_cell_beyond_the_region():
    """A cell whose corner sits a whole cell before the region still clips it;
    one further out cannot. So the dilation is the region grown by exactly the
    cell on the near sides."""
    dilated = _dilate_by_cell(room(12.0, 9.0), 3.0, 3.0)

    minx, miny, maxx, maxy = dilated.bounds
    assert minx == pytest.approx(-3.0, abs=1e-6)
    assert miny == pytest.approx(-3.0, abs=1e-6)
    assert maxx == pytest.approx(12.0, abs=1e-6)
    assert maxy == pytest.approx(9.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# the floor
# --------------------------------------------------------------------------- #

def test_a_room_that_tiles_exactly_has_a_floor_of_zero():
    """The instance that falsifies the covering bound. A 12x9 room at 3x3 tiles
    into 12 complete cells and no partials, while length over diagonal claims a
    floor of 10 -- a floor above an achievable value is not a floor."""
    assert packer(room()).partial_floor() == 0


def test_the_floor_is_never_above_what_a_placement_achieves():
    """The defining property, checked against a sweep that shares none of its
    reasoning. A floor that exceeds an attainable count is worse than no floor,
    because the gap computed from it would certify a lie."""
    shapes = {
        "room": room(),
        "room-odd": room(13.0, 10.0),
        "room-tilt23": room(tilt=23.0),
        "l-shape": Polygon([(0, 0), (12, 0), (12, 6), (6, 6), (6, 9), (0, 9)]),
        "disc": Point(0, 0).buffer(7.0, 12),
    }
    for name, shape in shapes.items():
        floor = packer(shape).partial_floor()
        assert floor <= dense_min_partial(packer(shape)), name


def test_the_floor_holds_at_every_offset_not_just_the_sampled_ones():
    """Stronger than the sweep above: random offsets, none of them special."""
    pk = packer(room(13.0, 10.0))
    floor = pk.partial_floor()

    rng = np.random.default_rng(7)
    for _ in range(25):
        dx, dy = rng.uniform(0, pk.cw), rng.uniform(0, pk.ch)
        assert pk.evaluate(float(dx), float(dy)).partial >= floor


def test_the_floor_is_computed_where_the_covering_bound_declines():
    """The point of the exercise. A serrated wall puts more boundary inside one
    cell than the cell's diagonal, which is exactly the assumption the covering
    bound rests on -- it declines here, and this does not."""
    teeth = [(0, 0)]
    for i in range(12):
        teeth += [(i + 0.5, 0.8), (i + 1.0, 0.0)]
    serrated = Polygon(teeth + [(12, 9), (0, 9)])
    pk = packer(serrated)

    placement = pk.evaluate(0.0, 0.0, classify=True)
    assert not pk.certificate(placement).certified

    assert pk.partial_floor() <= dense_min_partial(packer(serrated))


def test_the_floor_respects_obstacles():
    pk = packer(room(), (room(3.0, 3.0),))

    assert pk.partial_floor() <= dense_min_partial(packer(room(), (room(3.0, 3.0),)))


def test_the_floor_is_reported_for_the_angle_it_was_asked_about():
    pk = packer(room(tilt=23.0))

    assert pk.partial_floor(angle=23.0) <= dense_min_partial(
        packer(room(tilt=23.0)), angle=23.0)


# --------------------------------------------------------------------------- #
# what a caller sees
# --------------------------------------------------------------------------- #

def test_the_certificate_carries_the_computed_floor_when_asked():
    """Opt-in, like every other proof here: it costs a fraction of a second
    rather than nothing, and a plain request should not start paying for it."""
    pk = packer(room(13.0, 10.0))
    placement = pk.evaluate(0.0, 0.0, classify=True)

    plain = pk.certificate(placement)
    computed = pk.certificate(placement, exact_floor=True)

    assert plain.angle_floor is None
    assert computed.angle_floor == 8


def test_the_computed_floor_beats_the_covering_bound_where_that_says_nothing():
    """The covering bound returns 0 on a room whose every placement leaves 8
    partial cells -- true, useless, and indistinguishable from a good result."""
    pk = packer(room(13.0, 10.0))
    placement = pk.evaluate(0.0, 0.0, classify=True)

    certificate = pk.certificate(placement, exact_floor=True)

    assert certificate.floor == 0
    assert certificate.angle_floor == 8
    assert certificate.angle_gap == placement.partial - 8


def test_the_two_floors_are_not_the_same_claim():
    """One holds over rotation and can decline; the other holds only at this
    angle and never declines. Reporting them as one number would overstate
    whichever was weaker."""
    pk = packer(room(13.0, 10.0))
    placement = pk.evaluate(0.0, 0.0, classify=True)

    certificate = pk.certificate(placement, exact_floor=True)

    assert certificate.angle_floor > certificate.floor
    assert certificate.certified is True


def test_the_service_reports_the_computed_floor_when_certifying():
    from packer_service import run_packing

    shape = [(0.0, 0.0), (13.0, 0.0), (13.0, 10.0), (0.0, 10.0)]
    plain = run_packing(shape, [], 3.0, 3.0, rotate=False)
    certified = run_packing(shape, [], 3.0, 3.0, rotate=True, certify=True)

    assert "angle_partial_floor" not in plain["stats"]
    assert certified["stats"]["angle_partial_floor"] >= 0
    assert certified["stats"]["angle_partial_gap"] >= 0


# --------------------------------------------------------------------------- #
# over every angle, not just the one asked about
# --------------------------------------------------------------------------- #
# Rotating by theta moves a point at radius r by r*theta, so a window of angles
# is bracketed on BOTH sides:
#
#     shrink(R, radius*half) is inside R(theta) is inside grow(R, radius*half)
#
# The dilation is monotone in the region and so is the erosion, so over the
# window the touching set can only shrink to the dilation of the shrunken
# region, and the complete set can only grow to the erosion of the grown one.
# Their difference bounds the partial count from below for every angle at once,
# which is what turns a floor at one angle into a floor at all of them.

def test_a_window_of_zero_is_the_floor_at_that_angle():
    pk = packer(room(13.0, 10.0))

    assert pk.partial_floor_bound(0.0, 0.0) == pk.partial_floor(0.0)


def test_the_window_floor_is_under_every_angle_it_covers():
    """Checked against placements at angles inside the window, since a floor
    that some angle beats is not a floor."""
    pk = packer(room(13.0, 10.0))
    centre, half_window = 10.0, 4.0

    floor = pk.partial_floor_bound(centre, half_window)

    for theta in np.linspace(centre - half_window, centre + half_window, 9):
        assert floor <= dense_min_partial(packer(room(13.0, 10.0)), steps=12,
                                          angle=float(theta))


def test_a_wider_window_can_only_lower_the_floor():
    pk = packer(room(13.0, 10.0))

    floors = [pk.partial_floor_bound(10.0, w) for w in (0.0, 0.5, 2.0, 8.0)]

    assert floors == sorted(floors, reverse=True)


def test_a_room_that_tiles_is_proven_to_need_no_partials_at_any_angle():
    """The whole point: a claim over rotation with nothing assumed. A 12x9 room
    tiles exactly, so zero partials is both achievable and unbeatable, and the
    search has to close rather than merely report a small number."""
    best, certificate = packer(room()).certify_partials()

    assert certificate.floor == 0
    assert certificate.achieved == 0
    assert certificate.optimal
    assert best.partial == 0


def test_the_proven_floor_is_never_above_what_any_angle_achieves():
    pk = packer(room(13.0, 10.0))

    _, certificate = pk.certify_partials()

    for angle in (0.0, 17.0, 33.0, 61.0):
        assert certificate.floor <= dense_min_partial(
            packer(room(13.0, 10.0)), steps=12, angle=angle)


def test_a_budget_too_small_reports_an_open_floor_rather_than_a_proof():
    """A disc, because a room that tiles is proven straight from the seed and
    never spends a node -- so it cannot demonstrate what running out looks
    like. On a disc the count barely varies with angle, nothing prunes, and the
    budget is what stops the search."""
    pk = packer(Point(0, 0).buffer(7.0, 12))

    _, certificate = pk.certify_partials(max_nodes=2)

    assert not certificate.optimal
    assert certificate.floor <= certificate.achieved


def test_the_search_finds_the_offset_that_minimises_partials_not_completes():
    """Regression: the two objectives are not the same search.

    A 12x9 room tilted 23 degrees still tiles exactly, so a 3x3 grid turned to
    23 degrees leaves ZERO partial cells. The search reported 8, and called it
    proven, because its incumbent came from the solver that maximises complete
    cells -- which returns one placement per angle and need not be the one
    leaving fewest partials. A floor above an achievable count is not a floor,
    and claiming it is optimal is the worst failure this code has.
    """
    pk = packer(room(tilt=23.0))

    best, certificate = pk.certify_partials()

    assert certificate.floor == 0
    assert certificate.achieved == 0
    assert best.partial == 0


def test_a_leaf_that_still_beats_the_incumbent_is_not_called_proven():
    """The other half of that failure. A window too narrow to split is dropped
    from the queue; if its floor is still under the incumbent, the space was
    not closed and the certificate must not say it was."""
    pk = packer(Point(0, 0).buffer(7.0, 12))

    _, certificate = pk.certify_partials(max_nodes=12)

    assert certificate.optimal == (certificate.floor == certificate.achieved
                                   and certificate.exhausted)
