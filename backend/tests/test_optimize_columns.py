"""Tests for the column solver: the dy axis solved rather than enumerated.

`optimize_exact` removed the resolution from the offset search but kept its
shape -- a double loop over Vx * Vy faces, each a full reclassification of all
N cells. `optimize_columns` keeps the dx enumeration and replaces the inner one
with an exact 1-D solve, which makes two claims that have to be separated:

  * it agrees with `optimize_exact` wherever that search is exact, which is
    every rectilinear instance. Agreement is asserted on the OPTIMUM, not on the
    offsets, because two different placements can both be optimal;
  * it strictly beats it where that search is not exact. A slanted edge flips a
    cell where a lattice corner grazes it, which is not any vertex's offset, so
    the enumeration can miss the optimum outright. The counterexample is pinned
    here with a dense sweep as the arbiter, so the claim rests on a measurement
    and not on the argument that produced it.

Cost is asserted too, since removing an enumeration is the entire point.
"""
import numpy as np
import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon

from grid_packer import (GridPacker, _best_lattice_offset, _erode_intervals,
                         _full_width_intervals)


def room(width=36.0, height=27.0, tilt=0.0, origin=(0.0, 0.0)) -> Polygon:
    ox, oy = origin
    box = Polygon([(ox, oy), (ox + width, oy),
                   (ox + width, oy + height), (ox, oy + height)])
    return shp_rotate(box, tilt) if tilt else box


def l_shape() -> Polygon:
    return Polygon([(0, 0), (36, 0), (36, 15), (15, 15), (15, 27), (0, 27)])


def packer(shape, obstacles=(), cw=3.0, ch=3.0) -> GridPacker:
    return GridPacker(shape, list(obstacles), cell_width=cw, cell_height=ch)


def dense_best(pk: GridPacker, steps: int, angle: float = 0.0) -> int:
    best = -1
    for dx in np.linspace(0, pk.cw, steps, endpoint=False):
        for dy in np.linspace(0, pk.ch, steps, endpoint=False):
            best = max(best, pk.evaluate(dx, dy, angle).complete)
    return best


# --------------------------------------------------------------------------- #
# the pieces
# --------------------------------------------------------------------------- #

def test_full_width_intervals_find_the_span_of_a_plain_room():
    """A column of a 36x27 room admits a full-width cut at every height in it."""
    pk = packer(room())
    from shapely.prepared import prep
    from grid_packer import _grow

    intervals = _full_width_intervals(pk.usable, prep(_grow(pk.usable)),
                                      6.0, 9.0, 0.0, 27.0)

    assert len(intervals) == 1
    lo, hi = intervals[0]
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert hi == pytest.approx(27.0, abs=1e-6)


def test_full_width_intervals_break_where_a_slanted_edge_crosses_the_column():
    """The breakpoint a vertex-only search misses.

    A triangle has no vertex inside the column, yet the height at which its
    sloping side crosses the column's own side line is exactly where full-width
    coverage stops. If those crossings were not breakpoints, the interval would
    run to the apex and report cells that are not there.
    """
    triangle = Polygon([(0, 0), (30, 0), (0, 30)])
    pk = packer(triangle)
    from shapely.prepared import prep
    from grid_packer import _grow

    intervals = _full_width_intervals(pk.usable, prep(_grow(pk.usable)),
                                      6.0, 9.0, 0.0, 30.0)

    # x + y = 30 on the hypotenuse, so the full width [6, 9] is covered only up
    # to y = 21, where the slope crosses the column's right side.
    assert len(intervals) == 1
    lo, hi = intervals[0]
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert hi == pytest.approx(21.0, abs=1e-3)


def test_eroding_by_the_cell_height_drops_intervals_that_cannot_hold_a_cell():
    assert _erode_intervals([(0.0, 10.0)], 3.0) == [(0.0, 7.0)]
    assert _erode_intervals([(0.0, 2.0)], 3.0) == []


def lattice_hits(columns, period, offset):
    """Count by definition, for use as the arbiter below."""
    total = 0
    for intervals in columns:
        for lo, hi in intervals:
            j = np.ceil((lo - offset) / period)
            while offset + j * period <= hi + 1e-12:
                total += 1
                j += 1
    return total


COLUMN_SETS = [
    [[(0.0, 6.0)], [(0.5, 6.5)]],          # two columns pulling different ways
    [[(1.0, 2.0)]],                        # an interval shorter than the period
    [[(0.0, 0.4)], [(2.9, 3.4)]],          # hits only at one narrow offset band
    [[(0.2, 9.1)], [(0.2, 9.1)], [(4.0, 5.0)]],
    [[]],                                  # a column that admits nothing
]


@pytest.mark.parametrize("columns", COLUMN_SETS)
def test_the_lattice_offset_matches_a_dense_scan(columns):
    """The exact 1-D solve is checked against brute force, which is the only
    honest way to assert "no offset can beat this": a dense scan makes no
    structural assumption, so it cannot share a blind spot with the solver."""
    period = 3.0
    count, offset = _best_lattice_offset(columns, period)

    scanned = max(lattice_hits(columns, period, d)
                  for d in np.linspace(0, period, 2000, endpoint=False))

    assert count == scanned
    assert lattice_hits(columns, period, offset) == count


# --------------------------------------------------------------------------- #
# agreement where the enumeration is exact
# --------------------------------------------------------------------------- #

RECTILINEAR = {
    "room": (room(), ()),
    "room-offset": (room(origin=(1.1, 0.7)), ()),
    "l-shape": (l_shape(), ()),
    "room-pillars": (room(), (room(6, 6, origin=(9, 9)),
                              room(3, 9, origin=(24, 6)))),
    "room-offgrid-pillar": (room(), (room(5.1, 6.9, origin=(10.2, 9.3)),)),
}


@pytest.mark.parametrize("name", sorted(RECTILINEAR))
def test_the_column_solver_agrees_with_the_enumeration(name):
    shape, obstacles = RECTILINEAR[name]

    enumerated, _ = packer(shape, obstacles).optimize_exact()
    columns, _ = packer(shape, obstacles).optimize_columns()

    assert columns.complete == enumerated.complete


@pytest.mark.parametrize("cw, ch", [(3.0, 3.0), (2.0, 3.0), (3.0, 2.0)])
def test_agreement_holds_for_rectangular_cells(cw, ch):
    enumerated, _ = packer(l_shape(), cw=cw, ch=ch).optimize_exact()
    columns, _ = packer(l_shape(), cw=cw, ch=ch).optimize_columns()

    assert columns.complete == enumerated.complete


def test_the_reported_placement_really_scores_what_is_claimed():
    """The sweep chooses the placement; `evaluate` scores it. A disagreement
    between the two would make every count in the paper suspect."""
    pk = packer(room(tilt=23.0))
    best, results = pk.optimize_columns()

    assert best.complete == pk.evaluate(best.dx, best.dy, best.angle).complete
    assert best.complete == max(p.complete for p in results)


# --------------------------------------------------------------------------- #
# dominance where the enumeration is not exact
# --------------------------------------------------------------------------- #

def test_the_column_solver_finds_what_the_enumeration_misses_on_a_tilted_room():
    """The counterexample, with a dense sweep as the arbiter.

    A vertex-derived dy set cannot express "the offset at which a lattice corner
    grazes this slanted wall", so the enumeration tops out below the true
    optimum. The dense sweep is the ground truth here precisely because it makes
    no structural assumption at all.
    """
    pk = packer(room(tilt=23.0))

    enumerated, _ = pk.optimize_exact()
    columns, _ = pk.optimize_columns()
    truth = dense_best(pk, steps=120)

    assert enumerated.complete < truth
    assert columns.complete == truth


@pytest.mark.parametrize("tilt", [7.0, 12.0, 31.0, 44.0])
def test_the_column_solver_is_never_worse_on_a_tilted_room(tilt):
    shape = room(tilt=tilt)

    enumerated, _ = packer(shape).optimize_exact()
    columns, _ = packer(shape).optimize_columns()

    assert columns.complete >= enumerated.complete


def test_the_column_solver_is_never_worse_on_a_curved_region():
    disc = Point(0, 0).buffer(15.0, 16)

    enumerated, _ = packer(disc).optimize_exact()
    columns, _ = packer(disc).optimize_columns()

    assert columns.complete >= enumerated.complete


# --------------------------------------------------------------------------- #
# cost: the enumeration is gone, not hidden
# --------------------------------------------------------------------------- #

def test_the_inner_enumeration_is_gone():
    """One evaluation per dx, not per (dx, dy) pair.

    Asserted structurally rather than as a speed ratio: the number of
    evaluations must equal the number of dx faces, which is what "the dy axis is
    solved, not sampled" means operationally.
    """
    from grid_packer import _face_samples

    pk = packer(room(tilt=23.0))
    _, results = pk.optimize_columns()

    vx, _ = pk._critical_offsets(0.0)
    assert len(results) == len(_face_samples(vx, pk.cw))
    assert pk.evaluations == len(results)


def test_the_saving_grows_with_the_boundary_complexity():
    """A many-vertex boundary is where the old cost was quadratic.

    The disc is the instance that made the point: its evaluation count under the
    enumeration is the square of its distinct vertex coordinates, and here it is
    linear in them.
    """
    disc = Point(0, 0).buffer(15.0, 16)

    enumerated_packer = packer(disc)
    enumerated_packer.optimize_exact()
    columns_packer = packer(disc)
    columns_packer.optimize_columns()

    assert columns_packer.evaluations * 10 < enumerated_packer.evaluations
