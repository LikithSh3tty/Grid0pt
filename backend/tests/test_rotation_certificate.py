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


def test_a_window_that_nearly_closes_an_obstacle_does_not_break_the_geometry():
    """Regression: GEOS could not difference the swept complement here.

    Widening the region for a window SHRINKS its holes, and this window shrinks
    a 3-wide pillar to 0.8. Sweeping the ring of a hole that thin produced
    parallelograms whose union self-touched, which is an invalid polygon, and
    the difference that follows it failed outright with "unable to assign free
    hole to a shell" -- a crash, not a wrong answer, and only on an instance
    with obstacles at one particular window.

    Found by certifying the evaluation corpus, which is why these numbers look
    arbitrary: they are the window the search actually reached.
    """
    pillars = (Polygon([(9, 9), (15, 9), (15, 15), (9, 15)]),
               Polygon([(24, 6), (27, 6), (27, 15), (24, 15)]))
    pk = packer(room(), pillars)

    assert pk.rotation_bound(59.0625, 2.8125) > 0


@pytest.mark.parametrize("half_window", [2.8125, 5.625, 11.25])
def test_eroding_a_widened_region_survives_a_nearly_closed_hole(half_window):
    """The same failure at the helper that actually raises it.

    The region is widened exactly as `rotation_bound` widens it, constants and
    all, because the invalidity lives at particular coordinates: a plain buffer,
    or sweeping the region instead of its complement, both produce valid
    geometry and a test that asserts nothing.
    """
    from shapely.affinity import rotate as shp_rot
    from grid_packer import (_BUFFER_INSCRIBED_RATIO, _BUFFER_QUAD_SEGS,
                             _erode_by_cell)

    pillars = (Polygon([(9, 9), (15, 9), (15, 15), (9, 15)]),
               Polygon([(24, 6), (27, 6), (27, 15), (24, 15)]))
    pk = packer(room(), pillars)
    centre, radius = pk._turning_circle
    slack = radius * np.radians(half_window) / _BUFFER_INSCRIBED_RATIO
    work = shp_rot(pk.usable, -59.0625, origin=centre).buffer(
        slack, quad_segs=_BUFFER_QUAD_SEGS)

    eroded = _erode_by_cell(work, 3.0, 3.0)

    assert eroded.is_valid
    assert not eroded.is_empty


def test_the_bound_costs_no_evaluations():
    """It is geometry, not placement scoring, so it must not touch the counter
    the paper reports as cost."""
    pk = packer(room(tilt=23.0))

    pk.rotation_bound(20.0, 5.0)

    assert pk.evaluations == 0


# --------------------------------------------------------------------------- #
# the search the bound makes possible
# --------------------------------------------------------------------------- #
#: Small enough to certify in seconds. A 12x9 room tiles exactly at 3x3, and
#: tilting it rigidly cannot change what a grid can do to it, so 12 is the
#: optimum at 23 degrees and provably nowhere better at any other angle.
def small_room(tilt=0.0) -> Polygon:
    return shp_rotate(Polygon([(0, 0), (12, 0), (12, 9), (0, 9)]), tilt) \
        if tilt else Polygon([(0, 0), (12, 0), (12, 9), (0, 9)])


def test_it_proves_the_optimum_of_a_shape_whose_optimum_is_known():
    """The instance where the answer is known independently of any solver: a
    room whose sides are multiples of the cell tiles perfectly, and rotating it
    rigidly cannot change what a grid can do to it. So 12 is the optimum, and
    the certificate has to both reach it and close on it."""
    pk = packer(small_room(tilt=23.0))

    best, certificate = pk.certify_rotation()

    assert best.complete == 12
    assert certificate.bound == 12
    assert certificate.optimal


def test_the_bound_it_reports_is_never_below_what_it_achieved():
    """A certificate claiming less than the placement it certifies would be
    incoherent, and is the failure mode a wrong inequality shows up as."""
    for shape in (small_room(), small_room(tilt=23.0), l_shape()):
        pk = packer(shape)

        best, certificate = pk.certify_rotation(max_nodes=40)

        assert certificate.bound >= best.complete
        assert certificate.complete == best.complete


def test_it_never_returns_less_than_the_pipeline_it_certifies():
    """The vote's answer seeds the search, so the certificate can only improve
    on it. A certificate that cost a cell would be worse than no certificate."""
    pk = packer(small_room(tilt=23.0))
    guided, _ = packer(small_room(tilt=23.0)).optimize_guided()

    best, _ = pk.certify_rotation()

    assert best.complete >= guided.complete


def test_a_budget_too_small_says_so_instead_of_claiming_optimality():
    """The honest failure. A search that runs out of budget still holds a valid
    upper bound -- it just has not closed the gap -- and it must report that
    rather than quoting the incumbent as optimal."""
    pk = packer(room(tilt=23.0))          # the big room: no chance in 3 nodes

    best, certificate = pk.certify_rotation(max_nodes=3)

    assert not certificate.exhausted
    assert not certificate.optimal
    assert certificate.bound >= best.complete


def test_the_search_is_seedable_so_a_known_placement_can_be_certified():
    """Certifying a placement someone already has is the other use for this, and
    it must not throw away an incumbent better than the vote's."""
    pk = packer(small_room(tilt=23.0))
    seed, _ = packer(small_room(tilt=23.0)).optimize_erosion(angles=(23.0,))

    best, certificate = pk.certify_rotation(seed=seed)

    assert best.complete == 12
    assert certificate.optimal


def test_no_angle_outside_the_period_needs_examining():
    """The grid repeats, so the search space is the placement period and not
    the circle: 90 degrees for square cells, 180 when a quarter turn swaps the
    cell's sides into a genuinely different tiling."""
    assert packer(small_room(), cw=3.0, ch=3.0).rotation_period == 90.0
    assert packer(small_room(), cw=2.0, ch=3.0).rotation_period == 180.0


def test_a_budget_below_one_node_is_refused():
    with pytest.raises(ValueError):
        packer(small_room()).certify_rotation(max_nodes=0)


# --------------------------------------------------------------------------- #
# what a caller sees
# --------------------------------------------------------------------------- #

def coords(poly: Polygon):
    return [(float(x), float(y)) for x, y in poly.exterior.coords[:-1]]


def test_the_service_does_not_certify_unless_asked():
    """Proving the angle costs orders more than finding it, so a plain request
    must not start paying for it. Absent keys, not false ones: a client that
    did not ask cannot be handed a claim it has no way to interpret."""
    from packer_service import run_packing

    stats = run_packing(coords(small_room(tilt=23.0)), [], 3.0, 3.0,
                        rotate=True)["stats"]

    assert "rotation_bound" not in stats
    assert "rotation_optimal" not in stats


def test_the_service_reports_a_proven_optimum_when_asked():
    """The claim the whole certificate exists to let the API make: not
    'the best this found', but 'no placement of this grid on this region does
    better, at any angle'."""
    from packer_service import run_packing

    result = run_packing(coords(small_room(tilt=23.0)), [], 3.0, 3.0,
                         rotate=True, certify=True)

    assert result["stats"]["complete"] == 12
    assert result["stats"]["rotation_bound"] == 12
    assert result["stats"]["rotation_optimal"] is True
    assert result["stats"]["rotation_gap"] == 0


def test_certifying_cannot_return_a_worse_placement_than_not_certifying():
    from packer_service import run_packing

    plain = run_packing(coords(small_room(tilt=23.0)), [], 3.0, 3.0, rotate=True)
    certified = run_packing(coords(small_room(tilt=23.0)), [], 3.0, 3.0,
                            rotate=True, certify=True)

    assert certified["stats"]["complete"] >= plain["stats"]["complete"]


def test_the_endpoint_accepts_the_flag_and_defaults_it_off():
    from fastapi.testclient import TestClient
    from server import app

    client = TestClient(app)
    body = {"shape": coords(small_room(tilt=23.0)),
            "cell_width": 3.0, "cell_height": 3.0, "rotate": True}

    plain = client.post("/api/pack/polygon", json=body).json()
    certified = client.post("/api/pack/polygon",
                            json={**body, "certify": True}).json()

    assert "rotation_optimal" not in plain["stats"]
    assert certified["stats"]["rotation_optimal"] is True


# --------------------------------------------------------------------------- #
# tightening the bound: cover the window in pieces
# --------------------------------------------------------------------------- #
# Growing by radius x half-window is exact only at the turning circle and
# conservative everywhere inside it, and the error is linear in the window. So
# cover a wide window with several narrow ones instead: each sub-window is grown
# by a fraction of the slack, and their union still contains every angle in the
# window. Tighter for more geometry -- which turns out to be cheaper too, since
# the smaller region erodes to fewer pieces than the fat one did.

def test_splitting_the_window_tightens_the_bound():
    pk = packer(room(tilt=23.0))

    whole = pk.rotation_bound(30.0, 45.0, subwindows=1)
    split = pk.rotation_bound(30.0, 45.0, subwindows=4)

    assert split < whole


def test_the_split_bound_still_covers_every_angle_in_the_window():
    """Tightening is only allowed to remove slack, never to stop being a bound.
    A dense scan of the window is the check, since it shares no reasoning with
    the covering argument."""
    pk = packer(room(tilt=23.0))
    centre, half_window = 20.0, 6.0

    bound = pk.rotation_bound(centre, half_window, subwindows=4)

    for theta in np.linspace(centre - half_window, centre + half_window, 25):
        assert exact_at(packer(room(tilt=23.0)), float(theta)) <= bound


def test_a_wide_window_is_split_and_a_narrow_one_is_not():
    """The split earns its cost only while the window is worth more slack than
    a cell; below that it buys almost nothing and pays full geometry for it, so
    the count is read off the slack rather than fixed."""
    pk = packer(room(tilt=23.0))

    assert pk._bound_subwindows(45.0) > 1
    assert pk._bound_subwindows(0.05) == 1


def test_splitting_does_not_break_the_window_of_zero():
    pk = packer(room(tilt=23.0))

    assert pk.rotation_bound(23.0, 0.0) == exact_at(pk, 23.0)


# --------------------------------------------------------------------------- #
# the vote weight, settled by the certificate
# --------------------------------------------------------------------------- #
# The vote weights each chord by w = L * g(f), with f the inside fraction of the
# cell it came from. g up-weights cells with much to reclaim and starves
# hopeless slivers. Which g was never measurable before: two candidate angles
# five thousandths of a degree apart, and no way to say which was right.
#
# Certifying all 72 full-corpus instances answers it. g(f) = f^2 reaches the
# proven optimum on 67 of them against g(f) = f's 65, is never worse on any, and
# costs the same. So the default is f^2, on evidence rather than on the note.

def test_the_vote_weight_defaults_to_the_measured_one():
    from grid_packer import VOTE_WEIGHT_POWER

    assert VOTE_WEIGHT_POWER == 2.0


def test_the_default_weight_reaches_an_optimum_the_previous_one_missed():
    """The instance that settled it. The certificate proves 77 is the most any
    placement can hold at any angle; the old default returns 76."""
    traced = Polygon([(0, 0), (36, 0), (36, 15), (15, 15), (15, 27), (0, 27)])
    tilted = shp_rotate(traced, 23.0)

    previous, _ = packer(tilted).optimize_guided(weight_power=1.0)
    current, _ = packer(tilted).optimize_guided()

    assert current.complete >= previous.complete
