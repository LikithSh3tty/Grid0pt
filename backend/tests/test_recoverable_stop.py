"""Tests for the recoverable-area stop (design note section 8.4, bullet 4).

The note ends its align-then-solve-exactly pipeline with a stop criterion:
"stop when the recoverable area of the fringe (sum over f > tau of (1-f) *
area(C)) is below threshold -- no upside left to turn for". The quantity it
names is already computed and reported by the certificate; what these tests pin
down is WHERE in the pipeline reading it actually buys anything, because the
obvious placement does not.

The obvious placement is the entrance: measure the base fringe, and refuse to
rotate at all when there is nothing to reclaim. Measured, that separates
nothing. A 24x18 room tilted 12 degrees -- the instance where rotating is worth
+14 complete cells -- carries 2.89 cells of recoverable area at theta = 0, and a
disc, where rotating is worth nothing at all, carries 3.10. Any threshold that
silences the disc also silences the headline result. The quantity is not a
should-we-rotate signal; R already is one (0.029 for the disc against 1.000 for
the tilted room), and that gate is already in place.

Where the reading does separate is one stage later, between the candidate solves
and the local refine. Once the vote's angle has been solved exactly, an instance
whose walls are now flush has a fringe holding EXACTLY 0.00 cells of recoverable
area, while an instance still compromising between two wall families holds
1.2-1.7. The refine is the expensive stage -- 8 further exact translation solves,
measured at 105 -> 905 evaluations on the tilted room -- and on every instance
where the fringe came out flush it moved the count by nothing. So the stop is
read there: one classifying evaluation decides whether to spend eight solves.

These tests assert both halves: that it fires where the refine is provably
pointless, and that it stays quiet while upside remains.
"""
import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon

from grid_packer import RECOVERABLE_AREA_MIN, GridPacker


CELL = 3.0


def room(width=24.0, height=18.0, tilt=0.0) -> Polygon:
    """A plain rectangular room, optionally tilted off the axes."""
    box = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    return shp_rotate(box, tilt) if tilt else box


def trapezoid(tilt=10.0) -> Polygon:
    """Two wall families that cannot both be aligned at once.

    The sloping sides sit at different orientations from the horizontal ones, so
    whatever angle the vote picks leaves part of the fringe oblique -- which is
    precisely the case the note's local refine exists for, and therefore the
    case the stop must not silence.
    """
    return shp_rotate(Polygon([(0, 0), (30, 0), (24, 18), (6, 18)]), tilt)


def packer_for(shape: Polygon) -> GridPacker:
    return GridPacker(shape, [], cell_width=CELL, cell_height=CELL)


def fringe_cells(packer: GridPacker, placement) -> float:
    """The placement's recoverable area, in cell areas."""
    classified = packer.evaluate(placement.dx, placement.dy, placement.angle,
                                 classify=True)
    return packer._recoverable_area(classified) / (CELL * CELL)


# --------------------------------------------------------------------------- #
# the measurement the default threshold sits between
# --------------------------------------------------------------------------- #

def test_an_aligned_fringe_has_no_recoverable_area_left():
    """After the vote's angle is solved, a tilted room's fringe is flush.

    This is the number the stop keys on. It is asserted exactly, not merely as
    "small", because the whole argument for skipping the refine is that there is
    no area left for the refine to find.
    """
    packer = packer_for(room(tilt=23.0))
    best, _ = packer.optimize_guided(refine="none")

    assert best.angle == pytest.approx(23.0, abs=0.5)
    assert fringe_cells(packer, best) == pytest.approx(0.0, abs=1e-6)


def test_a_compromised_fringe_keeps_recoverable_area():
    """A shape with two wall families still has upside after the candidates.

    Both halves of the separation are asserted in the same units the threshold
    is expressed in, so the default's margin is visible in the test suite rather
    than buried in a constant's docstring.
    """
    packer = packer_for(trapezoid())
    best, _ = packer.optimize_guided(refine="none")

    assert fringe_cells(packer, best) > RECOVERABLE_AREA_MIN


def test_the_entry_gate_would_not_have_separated_anything():
    """Why the stop is not read at the entrance (the note's literal placement).

    The base fringe of an instance where rotating pays +14 cells is no larger
    than the base fringe of one where rotating pays nothing. Pinning this stops
    a future reader from "fixing" the placement of the stop back to the entry.
    """
    def base_fringe(packer: GridPacker):
        """(recoverable cells, R) of the un-rotated exact solve."""
        best, _ = packer.optimize_exact(angles=(0.0,))
        classified = packer.evaluate(best.dx, best.dy, 0.0, classify=True)
        return (packer._recoverable_area(classified) / (CELL * CELL),
                packer._dominant_orientations(classified).resultant)

    tilted_area, tilted_r = base_fringe(packer_for(room(tilt=12.0)))
    disc_area, disc_r = base_fringe(packer_for(Point(0, 0).buffer(9.0, 24)))

    # The instance that MUST rotate does not stand out from the one that must
    # not -- so no threshold on this quantity could gate the rotation.
    assert disc_area > tilted_area * 0.5
    # What does separate them is R, and that gate is already in place.
    assert tilted_r > 0.9
    assert disc_r < 0.35


# --------------------------------------------------------------------------- #
# the stop itself
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tilt", [12.0, 23.0, 31.0])
def test_the_refine_is_skipped_once_the_fringe_is_flush(tilt):
    """The expensive stage is not entered when it has nothing to find.

    Asserted as a cost RATIO rather than an absolute count so the test measures
    the stop and not the size of the instance.
    """
    packer = packer_for(room(tilt=tilt))

    stopped, _ = packer.optimize_guided()
    spent, _ = packer.optimize_guided(recover_min=0.0)

    assert stopped.complete == spent.complete
    assert stopped.rotation_vote.evaluations < spent.rotation_vote.evaluations / 4


def test_the_stop_stays_quiet_while_upside_remains():
    """A compromised fringe still gets its refine.

    Without this the stop would be indistinguishable from deleting the refine.
    The full refine budget is spent, plus the single probe that decided to spend
    it -- which is the whole downside of the stop when it does not fire.
    """
    packer = packer_for(trapezoid())

    stopped, _ = packer.optimize_guided()
    spent, _ = packer.optimize_guided(recover_min=0.0)

    assert stopped.rotation_vote.evaluations == spent.rotation_vote.evaluations + 1


def test_the_stop_is_disabled_at_zero():
    """recover_min = 0 restores the note's un-gated pipeline for the ablation."""
    packer = packer_for(room(tilt=23.0))

    off, _ = packer.optimize_guided(recover_min=0.0)
    on, _ = packer.optimize_guided(recover_min=RECOVERABLE_AREA_MIN)

    assert off.rotation_vote.evaluations > on.rotation_vote.evaluations


@pytest.mark.parametrize("name, shape", [
    ("aligned room", room()),
    ("tilted room", room(tilt=23.0)),
    ("trapezoid", trapezoid()),
    ("tilted L", shp_rotate(Polygon([(0, 0), (24, 0), (24, 9),
                                     (12, 9), (12, 18), (0, 18)]), 20.0)),
])
def test_the_stop_never_costs_a_complete_cell(name, shape):
    """The saving is free: same count on every instance, and the cost when the
    stop declines to fire is bounded by the one probe it spent asking."""
    packer = packer_for(shape)

    stopped, _ = packer.optimize_guided()
    spent, _ = packer.optimize_guided(recover_min=0.0)

    assert stopped.complete == spent.complete, name
    assert stopped.rotation_vote.evaluations <= spent.rotation_vote.evaluations + 1


def test_the_probe_the_stop_spends_is_counted():
    """The stop reads the fringe with a real evaluation, and says so.

    The reported evaluation count is the paper's cost metric, so the one
    classifying call the stop spends has to appear in it -- otherwise the method
    would look cheaper than it is by exactly the amount it charges to decide.
    """
    packer = packer_for(room(tilt=23.0))

    unrefined, _ = packer.optimize_guided(refine="none")
    stopped, _ = packer.optimize_guided()

    # Same solves, plus the single probe that decided against the refine.
    assert (stopped.rotation_vote.evaluations
            == unrefined.rotation_vote.evaluations + 1)
