"""Tests for the rotation certificate: optimality over angles, not just within one.

Translation is solved exactly for any shape, so `optimize_erosion` returns

    M(theta) = the most complete cells any placement at angle theta can hold,

and the remaining question is the one the fringe vote only guesses at: is the
angle it named the best angle? Nothing in the vote answers that. A certificate
has to bound M over a CONTINUUM of angles, and the way to do it is not to hunt
for the angles where M jumps -- a placement is tight in three degrees of freedom,
so an event needs three simultaneous contacts and the candidates run cubic in the
boundary size -- but to bound M over a whole angular window at once.

Rotating by theta moves a point at radius r from the pivot by r*|theta|, so every
angle within `half_window` of `angle` sees a region contained in this one grown
by radius * half_window. Erosion is monotone, so the overlap depth of THAT grown
region is an upper bound on M for every angle in the window:

    max over the window of M(theta)  <=  rotation_bound(angle, half_window)

which is `optimize_erosion`'s own machinery run on a slightly fattened region.
Everything else -- the branch and bound, the certificate it produces -- rests on
that one inequality, so it is tested first and directly, against a dense scan
that shares none of its reasoning.
"""
import numpy as np
import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon

from grid_packer import GridPacker


def room(width=36.0, height=27.0, tilt=0.0) -> Polygon:
    box = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    return shp_rotate(box, tilt) if tilt else box


def l_shape() -> Polygon:
    return Polygon([(0, 0), (36, 0), (36, 15), (15, 15), (15, 27), (0, 27)])


def packer(shape, obstacles=(), cw=3.0, ch=3.0) -> GridPacker:
    return GridPacker(shape, list(obstacles), cell_width=cw, cell_height=ch)


def exact_at(pk: GridPacker, angle: float) -> int:
    """M(angle): the most complete cells any placement at that angle holds."""
    best, _ = pk.optimize_erosion(angles=(angle,))
    return best.complete


# --------------------------------------------------------------------------- #
# the inequality everything rests on
# --------------------------------------------------------------------------- #

def test_a_window_of_zero_bounds_nothing_but_its_own_angle():
    """With no window the fattening vanishes and the bound is the exact count.
    This is what makes the branch and bound terminate rather than merely
    converge: shrink the window far enough and the bound stops being a bound."""
    pk = packer(room(tilt=23.0))

    assert pk.rotation_bound(23.0, 0.0) == exact_at(pk, 23.0)


def test_the_bound_dominates_every_angle_inside_its_window():
    """The claim, checked against a scan that shares none of its reasoning.

    A dense scan cannot prove a bound holds everywhere, but it can refute one,
    and refuting is what a test is for. The scan is the only arbiter here that
    does not reuse the monotonicity argument the bound is derived from.
    """
    pk = packer(room(tilt=23.0))
    centre, half_window = 20.0, 3.0

    bound = pk.rotation_bound(centre, half_window)

    for theta in np.linspace(centre - half_window, centre + half_window, 13):
        assert exact_at(packer(room(tilt=23.0)), float(theta)) <= bound


def test_the_bound_sees_a_flush_optimum_hiding_inside_the_window():
    """The case a scan misses and the bound must not.

    A 36x27 room tilts rigidly, so a 3x3 grid still tiles it exactly at 23
    degrees and nowhere near it -- 108 cells at one angle, 84 a degree away.
    A window centred at 20 contains that spike, so its bound has to be at least
    108 even though the angle it was computed at holds far fewer. A bound that
    only reflected its own centre would prune the interval holding the optimum.
    """
    pk = packer(room(tilt=23.0))

    assert exact_at(pk, 20.0) < 108
    assert pk.rotation_bound(20.0, 5.0) >= 108


@pytest.mark.parametrize("shape", [room(tilt=23.0), l_shape(),
                                   Point(0, 0).buffer(13.0, 12)])
def test_the_bound_only_grows_with_the_window(shape):
    """Monotone in the window, since a wider window fattens the region more.
    A branch and bound needs this: splitting an interval must never raise the
    bound on either half."""
    pk = packer(shape)

    bounds = [pk.rotation_bound(20.0, w) for w in (0.0, 0.5, 2.0, 8.0)]

    assert bounds == sorted(bounds)


def test_the_bound_closes_onto_the_exact_count():
    """Why the search terminates. As the window shrinks the fattening does too,
    so the bound falls to the value at the centre -- an interval narrow enough
    is decided rather than split forever."""
    pk = packer(room(tilt=23.0))
    exact = exact_at(pk, 11.0)

    assert pk.rotation_bound(11.0, 8.0) > exact
    assert pk.rotation_bound(11.0, 1e-6) == exact


def test_the_bound_holds_with_obstacles_in_the_way():
    """Obstacles are holes, and fattening the region SHRINKS them -- the right
    direction for an upper bound, and the easy thing to get backwards."""
    pk = packer(room(), (room(6.0, 6.0),))
    centre, half_window = 12.0, 2.0

    bound = pk.rotation_bound(centre, half_window)

    for theta in np.linspace(centre - half_window, centre + half_window, 9):
        assert exact_at(packer(room(), (room(6.0, 6.0),)), float(theta)) <= bound


def test_a_negative_window_is_refused():
    with pytest.raises(ValueError):
        packer(room()).rotation_bound(0.0, -1.0)


def test_the_bound_costs_no_evaluations():
    """It is geometry, not placement scoring, so it must not touch the counter
    the paper reports as cost."""
    pk = packer(room(tilt=23.0))

    pk.rotation_bound(20.0, 5.0)

    assert pk.evaluations == 0
