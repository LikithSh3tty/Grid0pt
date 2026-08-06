"""Tests for orientation-guided rotation (design note section 8).

Rotation is the part of the placement problem that does not collapse onto a
finite periodic set the way translation does, so the note replaces the blind
angle scan with a read-off: the oblique partial cells vote for the angle, by the
weighted circular mean of their chord orientations. This module pins down the
claims that method makes:

  * the headline -- a tilted room's own tilt is recovered with no angle scan,
    and the recovered angle is the true argmax over theta, checked against a
    reference scan rather than assumed;
  * the guarantee -- `optimize_guided` is never worse than its own un-rotated
    base, and never worse than the fixed-angle scan it replaces. A vote that
    misfires must cost evaluations and nothing else;
  * the gate -- R separates a shape with a dominant wall from one without, with
    the measured values asserted on both sides rather than just the branch taken;
  * the circular mean -- taken on the symmetry circle, so orientations at 1 and
    179 degrees average to 0 and not to 90. A naive mean gets this exactly
    backwards, so it is asserted directly;
  * the two corrections this implementation makes to the note's section 8, each
    with the measurement that motivated it: the vote circle is the ALIGNMENT
    period (always 90) rather than the placement period, and the prescribed
    golden-section refine is replaced by uniform sampling.

Runtimes are kept modest by using small grids: these instances are chosen for
the morphology they exercise, not for their size.
"""
import math

import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon

from grid_packer import (
    R_MIN,
    REFINE_HALF_WINDOW,
    GridPacker,
    _circular_mean,
)


# --------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------- #
def tilted_rect(width: float, height: float, degrees: float) -> Polygon:
    """A rectangle rotated about its own centroid.

    12x9 at 3x3 cells is the workhorse below: it tiles EXACTLY into 12 complete
    cells when the grid is aligned with it, and manages only 6 at theta = 0. The
    gap between those two numbers is the whole of what guided rotation buys.
    """
    base = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    return shp_rotate(base, degrees, origin=base.centroid)


def ngon(n: int, radius: float = 6.0) -> Polygon:
    """A regular n-gon -- the shape with no dominant wall direction."""
    return Polygon([(radius * math.cos(2 * math.pi * i / n),
                     radius * math.sin(2 * math.pi * i / n))
                    for i in range(n)])


def l_shape() -> Polygon:
    return Polygon([(0, 0), (12, 0), (12, 6), (6, 6), (6, 12), (0, 12)])


def rhombus(degrees: float, side: float = 12.0) -> Polygon:
    """A parallelogram: two wall families that are NOT perpendicular.

    No single grid angle can flush both families, so the optimum is a
    compromise that the vote's candidates straddle rather than hit. This is the
    one instance in this module where the local refine changes the answer, which
    makes it the instance that decides whether the refine earns its place.
    """
    dx = side * math.cos(math.radians(degrees))
    dy = side * math.sin(math.radians(degrees))
    return Polygon([(0, 0), (side, 0), (side + dx, dy), (dx, dy)])


def area_bound(packer: GridPacker) -> int:
    """The most complete cells any placement could possibly have.

    Complete cells are disjoint and lie inside the usable region, so their count
    can never exceed area(U) / area(C) whatever the offset or the angle. On an
    instance where the bound is attained it is a GLOBAL optimality certificate
    over the whole (dx, dy, theta) space -- strictly stronger than a reference
    angle scan, and free. The 12x9-at-3x3 room is chosen for exactly that: its
    area is 108 and its cell area 9, so 12 complete cells is unimprovable.
    """
    return int(packer.usable.area // (packer.cw * packer.ch))


#: Instances spanning the cases the method has to survive: already aligned,
#: rectilinear, no dominant wall, genuinely tilted, and tilted with rectangular
#: cells (where the placement period is 180 rather than 90).
INSTANCES = [
    ("axis-aligned rectangle", Polygon([(0, 0), (12, 0), (12, 9), (0, 9)]), 3.0, 3.0),
    ("L-shape", l_shape(), 3.0, 3.0),
    ("16-gon", ngon(16), 3.0, 3.0),
    ("rectangle tilted 23", tilted_rect(12, 9, 23), 3.0, 3.0),
    ("rectangle tilted 7, rectangular cells", tilted_rect(12, 9, 7), 3.0, 2.0),
]


# --------------------------------------------------------------------- #
# the headline: the tilt is read off, not searched for
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("tilt", [12.0, 23.0, 31.0])
def test_a_tilted_room_votes_for_its_own_tilt(tilt):
    """The fringe names the wall orientation with no angle scan at all.

    A 12x9 room at 3x3 cells tiles exactly when the grid matches its walls, so
    the target is unambiguous: 12 complete cells at grid angle `tilt`. At
    theta = 0 the same room manages 6.
    """
    packer = GridPacker(tilted_rect(12, 9, tilt), cell_width=3, cell_height=3)

    base, _ = packer.optimize_exact(angles=(0.0,))
    best, _ = packer.optimize_guided()
    vote = best.rotation_vote

    assert base.complete == 6                       # what not turning costs
    assert vote.angle == pytest.approx(tilt, abs=0.5)
    assert vote.resultant > 0.9                     # one wall family, no noise
    assert best.complete == 12                      # the exact tiling
    assert best.angle == pytest.approx(tilt, abs=0.5)


@pytest.mark.parametrize("tilt", [12.0, 23.0, 31.0])
def test_the_voted_angle_is_globally_optimal(tilt):
    """The claim of section 8.3 -- that phi* IS the optimum -- certified.

    Complete cells are disjoint and inside the region, so no placement at any
    angle can exceed area(U)/area(C). This room attains that bound, so reaching
    it proves the voted angle is a global optimum over the whole (dx, dy, theta)
    space. A reference angle scan could only ever say "nothing better was
    sampled".
    """
    packer = GridPacker(tilted_rect(12, 9, tilt), cell_width=3, cell_height=3)

    best, _ = packer.optimize_guided()

    assert area_bound(packer) == 12
    assert best.complete == 12


def test_guided_costs_far_less_than_the_scan_it_replaces():
    """The point of the read-off: a better answer, at a fraction of the cost.

    Measured against a COARSE 5-degree ladder -- 18 angles, half the 90 a
    1-degree scan needs -- which still cannot reach 23 degrees and so tops out
    at 6 where the vote reaches the unimprovable 12.

    The cost split is pinned deliberately, because it is the honest version of
    this claim: the vote itself is what is cheap. Naming the angle and solving
    translation there takes ~110 evaluations against the ladder's 1800, a 16x
    saving. Turning the local refine on multiplies that by roughly eight, since
    each probe is a full exact-translation solve, and it still comes in under
    the ladder. Which side of that trade to take is the ablation of
    `REFINE_METHOD`, and it should be reported rather than buried.

    On THIS instance the trade does not even arise: the vote's angle leaves the
    walls flush, so the recoverable-area stop declines to enter the refine at
    all (see `RECOVERABLE_AREA_MIN`) and the default costs what the un-refined
    solve costs, plus the one probe that decided it. The forced-refine figure is
    still measured here, because it is the cost the stop is avoiding.
    """
    packer = GridPacker(tilted_rect(12, 9, 23), cell_width=3, cell_height=3)

    ladder = tuple(float(a) for a in range(0, 90, 5))
    scan_best, scan_batch = packer.optimize_exact(angles=ladder)

    unrefined, _ = packer.optimize_guided(refine="none")
    refined, _ = packer.optimize_guided(refine="grid", recover_min=0.0)
    default, _ = packer.optimize_guided()

    assert scan_best.complete == 6
    assert unrefined.complete == 12
    assert refined.complete == 12
    assert default.complete == 12

    assert unrefined.rotation_vote.evaluations < len(scan_batch) / 10
    assert refined.rotation_vote.evaluations < len(scan_batch)
    # The refine, not the vote, is where a guided solve spends its budget...
    assert refined.rotation_vote.evaluations > 4 * unrefined.rotation_vote.evaluations
    # ...which is exactly why the default does not spend it here.
    assert (default.rotation_vote.evaluations
            == unrefined.rotation_vote.evaluations + 1)


# --------------------------------------------------------------------- #
# the guarantee: turning can never cost anything
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name,shape,cw,ch", INSTANCES)
def test_guided_is_never_worse_than_not_rotating(name, shape, cw, ch):
    """The never-worse guarantee, including where the vote is unhelpful.

    The un-rotated base stays in the result pool and the winner is the argmax
    over it, so a misfiring vote costs evaluations and nothing else.
    """
    packer = GridPacker(shape, cell_width=cw, cell_height=ch)

    base, _ = packer.optimize_exact(angles=(0.0,))
    guided, _ = packer.optimize_guided()

    assert guided.complete >= base.complete


@pytest.mark.parametrize("name,shape,cw,ch", INSTANCES)
def test_guided_is_never_worse_than_the_fixed_angle_scan(name, shape, cw, ch):
    """The comparison against the baseline the repo keeps for its ablation:
    the old uniform offset sweep over a fixed 15-degree angle ladder."""
    packer = GridPacker(shape, cell_width=cw, cell_height=ch)
    period = int(packer.rotation_period)

    old, _ = packer.optimize(steps=10, angles=tuple(range(0, period, 15)))
    guided, _ = packer.optimize_guided()

    assert guided.complete >= old.complete


def test_guided_strictly_beats_the_fixed_angle_scan_on_a_tilted_room():
    """Never-worse would be satisfied by doing nothing; this is the upside.

    23 degrees is deliberately not a multiple of 15, so the fixed ladder cannot
    reach it however finely the offsets are swept.
    """
    packer = GridPacker(tilted_rect(12, 9, 23), cell_width=3, cell_height=3)

    old, _ = packer.optimize(steps=10, angles=tuple(range(0, 90, 15)))
    guided, _ = packer.optimize_guided()

    assert old.complete == 6
    assert guided.complete == 12


# --------------------------------------------------------------------- #
# the gate: is there a wall worth turning for?
# --------------------------------------------------------------------- #
def test_a_room_has_a_dominant_wall_and_a_disc_does_not():
    """R is asserted on both sides of the gate, not merely the branch taken.

    A regular polygon approximating a circle has its chord orientations spread
    uniformly, so they cancel on the symmetry circle; a rectangle's two families
    are 90 degrees apart, which IS zero on that circle, so they reinforce.
    """
    disc = GridPacker(ngon(32), cell_width=3, cell_height=3)
    room = GridPacker(tilted_rect(12, 9, 23), cell_width=3, cell_height=3)

    def vote_for(packer):
        base, _ = packer.optimize_exact(angles=(0.0,))
        classified = packer.evaluate(base.dx, base.dy, 0.0, classify=True)
        return packer._dominant_orientations(classified)

    disc_vote = vote_for(disc)
    room_vote = vote_for(room)

    assert disc_vote.chord_count > 20               # it does have a fringe
    assert disc_vote.resultant < 0.2                # but it points nowhere
    assert not disc_vote.confident(R_MIN)

    assert room_vote.resultant > 0.9
    assert room_vote.confident(R_MIN)

    assert disc_vote.resultant < R_MIN < room_vote.resultant


def test_a_low_confidence_vote_leaves_the_grid_where_it_was():
    """Below the gate the method declines to turn rather than turning badly."""
    packer = GridPacker(ngon(32), cell_width=3, cell_height=3)

    base, _ = packer.optimize_exact(angles=(0.0,))
    best, _ = packer.optimize_guided()

    assert not best.rotation_vote.confident(R_MIN)
    assert best.angle == 0.0
    assert best.complete == base.complete


# --------------------------------------------------------------------- #
# the circular mean, and the circle it is taken on
# --------------------------------------------------------------------- #
def test_orientations_average_on_the_symmetry_circle():
    """1 and 179 degrees are 2 degrees apart, not 178.

    A chord is undirected and the grid is symmetric under quarter turns, so
    these two orientations are nearly the same wall. The naive arithmetic mean
    reports 90 -- the worst possible answer, exactly 45 degrees off both. The
    multiplier m is what fixes it, so this asserts the fixed value directly.
    """
    mean, resultant, weight = _circular_mean([(1.0, 1.0), (179.0, 1.0)], 4.0)

    assert mean % 90.0 == pytest.approx(0.0, abs=0.5)
    assert mean % 90.0 != pytest.approx(90.0, abs=1.0)
    assert resultant > 0.99                         # they agree, so R is high
    assert weight == pytest.approx(2.0)


def test_perpendicular_walls_reinforce_rather_than_cancel():
    """The reason the alignment circle is 90 and not 180.

    A rectangular room presents two wall families 90 degrees apart. On the
    alignment circle those are the SAME direction -- both are flush with the
    grid, one along the cell width and one along its height -- so they must add
    up, not cancel.
    """
    mean, resultant, _ = _circular_mean([(23.0, 1.0), (113.0, 1.0)], 4.0)

    assert mean % 90.0 == pytest.approx(23.0, abs=0.5)
    assert resultant > 0.99


def test_the_vote_needs_the_taxonomy():
    """Voting off an unclassified placement would silently see an empty fringe
    and report R = 0, so it is refused instead."""
    packer = GridPacker(tilted_rect(12, 9, 23), cell_width=3, cell_height=3)
    unclassified = packer.evaluate(0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="classify"):
        packer._dominant_orientations(unclassified)


# --------------------------------------------------------------------- #
# periods: the placement circle and the alignment circle are different
# --------------------------------------------------------------------- #
def test_square_cells_have_a_quarter_turn_period_and_rectangular_ones_a_half():
    """Turning a rectangular cell by 90 degrees swaps its width and height,
    which is a genuinely different tiling, so its angles range over [0, 180)."""
    square = GridPacker(l_shape(), cell_width=3, cell_height=3)
    oblong = GridPacker(l_shape(), cell_width=3, cell_height=2)

    assert square.rotation_period == 90.0
    assert oblong.rotation_period == 180.0


def test_the_vote_circle_is_alignment_not_the_placement_period():
    """The note writes m = 360/P with P the PLACEMENT period. For rectangular
    cells that is wrong, and expensively so.

    Alignment always has period 90: a wall is flush with the grid exactly when
    its orientation is 0 mod 90, whatever the cell's aspect ratio, because the
    grid has both horizontal and vertical lines. Averaging a rectangular cell's
    two perpendicular wall families on the placement circle (m = 2) makes them
    ANTIPODAL, so they cancel, R collapses and the method declines to turn for
    a room that is plainly tilted.

    Both readings are kept runnable so the paper can report this as a
    measurement rather than an assertion.
    """
    packer = GridPacker(tilted_rect(12, 9, 23), cell_width=3, cell_height=2)
    base, _ = packer.optimize_exact(angles=(0.0,))
    classified = packer.evaluate(base.dx, base.dy, 0.0, classify=True)

    alignment = packer._dominant_orientations(classified)
    placement = packer._dominant_orientations(
        classified, vote_period=packer.rotation_period)

    assert alignment.multiplier == 4.0
    assert placement.multiplier == 2.0

    # The two perpendicular families cancel on the placement circle.
    assert alignment.resultant > 0.9
    assert placement.resultant < 0.2

    # And the cancellation costs real cells: the note's reading gates itself
    # off and never turns.
    aligned_best, _ = packer.optimize_guided()
    note_best, _ = packer.optimize_guided(vote_period=packer.rotation_period)

    assert note_best.angle == 0.0
    assert note_best.complete == base.complete
    assert aligned_best.complete > note_best.complete


# --------------------------------------------------------------------- #
# the local refine: what the evidence supports
# --------------------------------------------------------------------- #
def test_the_refine_recovers_the_optimum_on_a_two_family_shape():
    """A parallelogram's two wall families are not perpendicular, so no angle
    flushes both and the optimum is a compromise the candidates straddle.

    This is the case the note introduces the refine for, and the only instance
    in this module where it changes the answer -- which is what justifies
    keeping it on by default despite its cost.
    """
    packer = GridPacker(rhombus(35.0), cell_width=2.5, cell_height=2.5)

    # A witness that 6 is attainable, instead of a 90-angle scan to find it.
    # The optimum sits on two plateaus, 30-33 and 61-64 degrees; 32 is one of
    # them and costs a single solve to check.
    witness, _ = packer.optimize_exact(angles=(32.0,))

    unrefined, _ = packer.optimize_guided(refine="none")
    refined, _ = packer.optimize_guided(refine="grid")

    assert witness.complete == 6
    assert unrefined.complete == 5
    assert refined.complete == 6

    # Note WHERE it got there: not by polishing a candidate onto the 30-33
    # plateau, but by finding an equally good placement a few degrees off the
    # wall-aligned candidate at 0. The refine is not a wall-alignment step; it
    # explores the neighbourhood the vote could not name. WHICH point of that
    # neighbourhood it lands on is not a property worth pinning -- the plateau
    # holds several, and solving the dy axis rather than enumerating it moved
    # the winner from 1.9 to 4.4 degrees without changing the count -- so what
    # is asserted is that it stayed inside the refine window and off the
    # plateau the candidates already offered.
    assert 0.0 < refined.angle <= REFINE_HALF_WINDOW


def test_golden_section_does_not_beat_uniform_sampling():
    """The note prescribes golden-section; the evidence does not support it.

    Golden-section assumes a unimodal continuous objective. N_complete(theta) is
    integer-valued and piecewise-constant, so on a plateau the bracket contracts
    on a comparison between two equal values and the search converges to an
    arbitrary interior point. At an equal probe budget it never wins.

    The strength of the case against it has changed, and honestly reporting that
    is the point of this test. While translation was solved by enumerating the
    offset arrangement, golden-section actually LOST here -- 5 against uniform
    sampling's 6 -- because the placement it converged on could not be rescued
    by the offsets that search could see. With the dy axis solved outright
    (`optimize_columns`), the same angle now yields the optimum and golden
    reaches 6 too. So the claim is no longer "it loses", it is "it does not
    win", which is still enough to keep uniform sampling as the default and
    golden as an ablation.
    """
    packer = GridPacker(rhombus(35.0), cell_width=2.5, cell_height=2.5)

    grid_refined, _ = packer.optimize_guided(refine="grid")
    golden_refined, _ = packer.optimize_guided(refine="golden")
    enumerated, _ = packer.optimize_guided(refine="golden", translation="exact")

    assert golden_refined.complete <= grid_refined.complete
    # ...and the measurement that used to make the case against it.
    assert enumerated.complete < grid_refined.complete


@pytest.mark.parametrize("name,shape,cw,ch", INSTANCES)
def test_every_refine_method_keeps_the_never_worse_guarantee(name, shape, cw, ch):
    """Whichever refine is chosen, the base is still in the pool."""
    packer = GridPacker(shape, cell_width=cw, cell_height=ch)
    base, _ = packer.optimize_exact(angles=(0.0,))

    for method in ("none", "grid", "golden"):
        best, _ = packer.optimize_guided(refine=method)
        assert best.complete >= base.complete, method


def test_an_unknown_refine_method_is_refused():
    packer = GridPacker(l_shape(), cell_width=3, cell_height=3)

    with pytest.raises(ValueError, match="refine"):
        packer.optimize_guided(refine="newton")
