"""Tests for the erosion solver: BOTH translation axes solved, for any shape.

`optimize_columns` solved the dy axis outright but kept dx as an enumeration
over vertex-derived critical offsets. That set is exhaustive only when the
boundary is rectilinear: a slanted edge flips a cell where a lattice corner
grazes it, an event that is a diagonal line in (dx, dy) and belongs to no
vertex's offset. So the dy fix moved the caveat onto dx rather than removing it.

`optimize_erosion` removes it. Written as a set,

    complete cell at (dx + i*cw, dy + j*ch)  <=>  that corner lies in
    F = { p : p + [0,cw]x[0,ch] is inside the usable region }

which is the region eroded by the cell -- computed once, exactly, with polygon
operations. The count is then the number of points of a lattice landing in a
FIXED set, so folding F modulo the lattice turns the whole search into "where is
the deepest overlap of these folded pieces", and that maximum is attained at a
vertex of their arrangement. Finitely many candidates, no sampling on either
axis, no assumption about the boundary.

The claims, and how each is pinned:

  * the pieces are what they say they are -- erosion and fold are asserted
    against hand-computable geometry;
  * the maximum found is a PROVEN optimum, not a better search. The overlap
    depth is an upper bound on the complete count at any offset whatsoever, so
    achieving it certifies optimality without a sweep to compare against. That
    is asserted directly, including on shapes no sweep could settle;
  * it agrees with the existing solvers wherever those are exact, and is never
    worse anywhere else -- including the instances that already separate
    `optimize_columns` from `optimize_exact`.
"""
import numpy as np
import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon

from grid_packer import (GridPacker, _erode_by_cell, _fold_tiles,
                         _ranked_offsets)


def room(width=36.0, height=27.0, tilt=0.0, origin=(0.0, 0.0)) -> Polygon:
    ox, oy = origin
    box = Polygon([(ox, oy), (ox + width, oy),
                   (ox + width, oy + height), (ox, oy + height)])
    return shp_rotate(box, tilt) if tilt else box


def l_shape() -> Polygon:
    return Polygon([(0, 0), (36, 0), (36, 15), (15, 15), (15, 27), (0, 27)])


def packer(shape, obstacles=(), cw=3.0, ch=3.0) -> GridPacker:
    return GridPacker(shape, list(obstacles), cell_width=cw, cell_height=ch)


#: The instance the dx enumeration gets wrong. Its two slanted sides put the
#: best column boundary at a dx that is no vertex's offset, so a vertex-derived
#: dx set tops out at 59 complete cells where 60 are available.
DX_GAP_SHAPE = Polygon([(0, 0), (34, 0), (28, 22), (5, 22)])


def dense_best(pk: GridPacker, steps: int, angle: float = 0.0) -> int:
    best = -1
    for dx in np.linspace(0, pk.cw, steps, endpoint=False):
        for dy in np.linspace(0, pk.ch, steps, endpoint=False):
            best = max(best, pk.evaluate(dx, dy, angle).complete)
    return best


# --------------------------------------------------------------------------- #
# the erosion: which grid-line corners can carry a complete cell
# --------------------------------------------------------------------------- #

def test_eroding_a_room_by_the_cell_leaves_the_corners_a_cell_fits_at():
    """A 3x3 cell fits in a 36x27 room exactly when its corner is in
    [0, 33] x [0, 24] -- the room shrunk by one cell on the far sides."""
    eroded = _erode_by_cell(room(), 3.0, 3.0)

    minx, miny, maxx, maxy = eroded.bounds
    assert minx == pytest.approx(0.0, abs=1e-6)
    assert miny == pytest.approx(0.0, abs=1e-6)
    assert maxx == pytest.approx(33.0, abs=1e-6)
    assert maxy == pytest.approx(24.0, abs=1e-6)


def test_the_erosion_drops_corners_whose_cell_would_leave_the_shape():
    """The L's missing arm. (12, 12) carries a cell; (16, 16) cannot, and the
    difference is invisible to a bounding box -- both points are inside the
    shape's bounds and (16, 16) is outside the shape itself."""
    eroded = _erode_by_cell(l_shape(), 3.0, 3.0)

    assert eroded.covers(Point(12.0, 12.0))
    assert not eroded.covers(Point(16.0, 16.0))


def test_the_erosion_drops_corners_whose_cell_would_clip_an_obstacle():
    """A cell must clear obstacles too, so the erosion has to punch a hole
    around each one that is the obstacle GROWN by a cell, not the obstacle."""
    pk = packer(room(), (room(6.0, 6.0, origin=(9.0, 9.0)),))
    eroded = _erode_by_cell(pk.usable, 3.0, 3.0)

    assert not eroded.covers(Point(8.0, 10.0))   # cell [8,11]x[10,13] clips it
    assert eroded.covers(Point(6.0, 10.0))       # cell [6,9]x[10,13] just clears


def test_a_shape_too_small_for_one_cell_erodes_to_nothing():
    assert _erode_by_cell(room(2.0, 2.0), 3.0, 3.0).is_empty


# --------------------------------------------------------------------------- #
# the fold: the lattice count becomes an overlap depth
# --------------------------------------------------------------------------- #

def test_the_fold_keeps_the_flush_placement_a_lower_dimensional_piece_carries():
    """The whole point of folding, and the case a careless fold loses.

    A 36x27 room tiles exactly at 3x3, so the optimum is 12 x 9 = 108 and it
    exists at ONE offset. The rightmost column of corners sits on x = 33, which
    is the far EDGE of the erosion -- a piece of zero area. Drop it as
    degenerate and the answer comes out 11 x 8 = 88, four rows and a column
    short of a placement that plainly exists.
    """
    tiles = _fold_tiles(_erode_by_cell(room(), 3.0, 3.0), 3.0, 3.0)

    depth, dx, dy = _ranked_offsets(tiles, 3.0, 3.0)[0]

    assert depth == 108
    assert dx == pytest.approx(0.0, abs=1e-6)
    assert dy == pytest.approx(0.0, abs=1e-6)


def test_the_ranking_is_ordered_deepest_first():
    tiles = _fold_tiles(_erode_by_cell(l_shape(), 3.0, 3.0), 3.0, 3.0)

    depths = [d for d, _, _ in _ranked_offsets(tiles, 3.0, 3.0)]

    assert depths == sorted(depths, reverse=True)


def test_the_depth_matches_the_count_it_stands_for():
    """Depth is not a proxy: it is the complete-cell count at that offset, so
    the placement the ranking names has to score exactly its own depth."""
    pk = packer(room(tilt=23.0))
    eroded = _erode_by_cell(pk.usable, pk.cw, pk.ch)
    tiles = _fold_tiles(eroded, pk.cw, pk.ch)

    depth, dx, dy = _ranked_offsets(tiles, pk.cw, pk.ch)[0]

    assert pk.evaluate(dx, dy).complete == depth


# --------------------------------------------------------------------------- #
# the optimum is proven, not compared
# --------------------------------------------------------------------------- #

PROVEN = {
    "room": room(),
    "room-offset": room(origin=(1.1, 0.7)),
    "l-shape": l_shape(),
    "room-tilt23": room(tilt=23.0),
    "room-tilt7": room(tilt=7.0),
    "triangle": Polygon([(0, 0), (31, 0), (0, 23)]),
    "triangle-tilt": shp_rotate(Polygon([(0, 0), (31, 0), (0, 23)]), 17.0),
    "trapezoid": DX_GAP_SHAPE,
    "pentagon": Polygon([(0, 0), (30, 4), (34, 25), (12, 31), (-4, 17)]),
    "disc": Point(0, 0).buffer(13.0, 12),
}


@pytest.mark.parametrize("name", sorted(PROVEN))
def test_the_result_attains_the_upper_bound_on_every_shape(name):
    """The claim that makes this exact for ALL shapes, asserted directly.

    Overlap depth is the complete-cell count as a function of offset, over the
    whole torus and with nothing sampled -- so its maximum bounds what ANY
    placement can achieve. A result equal to that bound is optimal by proof, and
    no dense sweep is needed to say so (nor could one, on a curved boundary).
    """
    pk = packer(PROVEN[name])
    tiles = _fold_tiles(_erode_by_cell(pk.usable, pk.cw, pk.ch), pk.cw, pk.ch)
    bound = _ranked_offsets(tiles, pk.cw, pk.ch)[0][0]

    best, _ = pk.optimize_erosion()

    assert best.complete == bound


@pytest.mark.parametrize("cw, ch", [(3.0, 3.0), (2.0, 3.0), (3.0, 2.0), (2.5, 4.0)])
def test_the_bound_is_attained_for_rectangular_cells_too(cw, ch):
    pk = packer(room(tilt=13.0), cw=cw, ch=ch)
    tiles = _fold_tiles(_erode_by_cell(pk.usable, cw, ch), cw, ch)
    bound = _ranked_offsets(tiles, cw, ch)[0][0]

    best, _ = pk.optimize_erosion()

    assert best.complete == bound


def test_the_reported_placement_really_scores_what_is_claimed():
    """The solver chooses the placement; `evaluate` remains the single
    definition of what a placement scores. The two must not drift apart."""
    pk = packer(room(tilt=23.0))

    best, results = pk.optimize_erosion()

    assert best.complete == pk.evaluate(best.dx, best.dy, best.angle).complete
    assert best.complete == max(p.complete for p in results)


# --------------------------------------------------------------------------- #
# against the solvers it replaces
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
def test_it_agrees_with_the_column_solver_where_that_is_exact(name):
    shape, obstacles = RECTILINEAR[name]

    columns, _ = packer(shape, obstacles).optimize_columns()
    erosion, _ = packer(shape, obstacles).optimize_erosion()

    assert erosion.complete == columns.complete


NON_RECTILINEAR = {
    "room-tilt7": room(tilt=7.0),
    "room-tilt23": room(tilt=23.0),
    "room-tilt31": room(tilt=31.0),
    "room-tilt44": room(tilt=44.0),
    "triangle": Polygon([(0, 0), (31, 0), (0, 23)]),
    "trapezoid": DX_GAP_SHAPE,
    "trapezoid-tilt": shp_rotate(DX_GAP_SHAPE, 9.0),
    "disc": Point(0, 0).buffer(13.0, 12),
}


@pytest.mark.parametrize("name", sorted(NON_RECTILINEAR))
def test_it_is_never_worse_than_the_column_solver(name):
    shape = NON_RECTILINEAR[name]

    columns, _ = packer(shape).optimize_columns()
    erosion, _ = packer(shape).optimize_erosion()

    assert erosion.complete >= columns.complete


@pytest.mark.parametrize("name", sorted(NON_RECTILINEAR))
def test_it_is_never_worse_than_a_dense_sweep(name):
    """The sweep is the only arbiter that shares no structure with either
    solver, so it is the check that the exactness argument did not simply
    reproduce its own blind spot."""
    shape = NON_RECTILINEAR[name]

    erosion, _ = packer(shape).optimize_erosion()

    assert erosion.complete >= dense_best(packer(shape), steps=45)


def test_it_solves_the_dx_axis_the_column_solver_still_enumerates():
    """The gap the erosion closes, on the instance that exhibits it.

    `optimize_columns` samples dx from vertex offsets, and on this shape the
    optimum needs a dx that is not one -- so it reports a count below the
    proven bound while the erosion reaches it. Pinned as a strict inequality
    because a non-strict one would pass even if the two agreed everywhere,
    which would mean this solver bought nothing.
    """
    pk = packer(DX_GAP_SHAPE)
    tiles = _fold_tiles(_erode_by_cell(pk.usable, pk.cw, pk.ch), pk.cw, pk.ch)
    bound = _ranked_offsets(tiles, pk.cw, pk.ch)[0][0]

    columns, _ = packer(DX_GAP_SHAPE).optimize_columns()
    erosion, _ = packer(DX_GAP_SHAPE).optimize_erosion()

    assert columns.complete < bound
    assert erosion.complete == bound


# --------------------------------------------------------------------------- #
# cost: nothing is enumerated on either axis
# --------------------------------------------------------------------------- #

def test_neither_axis_costs_an_evaluation():
    """`optimize_exact` pays Vx * Vy evaluations and `optimize_columns` pays
    Vx. Solving both axes leaves the offset search costing no evaluations at
    all beyond confirming the answer it computed."""
    pk = packer(room(tilt=23.0))

    pk.optimize_erosion()

    assert pk.evaluations == 1


def test_the_cost_does_not_grow_with_the_boundary_complexity():
    """The disc is the instance that made the point: 256 boundary vertices cost
    the enumeration 64k evaluations and the column solver 256. Here the count is
    independent of the boundary altogether."""
    disc = Point(0, 0).buffer(15.0, 64)

    pk = packer(disc)
    pk.optimize_erosion()

    assert pk.evaluations == 1


# --------------------------------------------------------------------------- #
# wired in: this is what the pipeline, the service and the table now solve with
# --------------------------------------------------------------------------- #

def proven_bound(shape, cw=3.0, ch=3.0) -> int:
    pk = packer(shape, cw=cw, ch=ch)
    tiles = _fold_tiles(_erode_by_cell(pk.usable, cw, ch), cw, ch)
    return _ranked_offsets(tiles, cw, ch)[0][0]


def test_the_guided_pipeline_solves_translation_this_way_by_default():
    """An `r_min` above 1 shuts the rotation gate, so what comes back is the
    translation solve at theta = 0 and nothing else -- which is exactly the
    default being asserted."""
    base, _ = packer(DX_GAP_SHAPE).optimize_guided(r_min=1.1)

    assert base.complete == proven_bound(DX_GAP_SHAPE)


def test_the_older_translation_solvers_are_still_reachable():
    """They are ablations of the results table, so the switch has to keep
    working -- and has to still be worse, or the ablation measures nothing."""
    eroded, _ = packer(DX_GAP_SHAPE).optimize_guided(r_min=1.1)
    columns, _ = packer(DX_GAP_SHAPE).optimize_guided(r_min=1.1,
                                                      translation="columns")

    assert columns.complete < eroded.complete


def test_an_unknown_translation_solver_is_refused():
    with pytest.raises(ValueError):
        packer(room()).optimize_guided(translation="nonsense")


def test_the_service_serves_the_placement_the_solver_proves():
    """The gap is not an internal detail -- an unrotated request for this shape
    used to come back a cell short of what the grid can do on it."""
    from packer_service import run_packing

    result = run_packing(list(DX_GAP_SHAPE.exterior.coords)[:-1], [],
                         cell_width=3.0, cell_height=3.0, rotate=False)

    assert result["stats"]["complete"] == proven_bound(DX_GAP_SHAPE)


def test_the_results_table_measures_the_solver_and_what_it_replaced():
    """Both have to be in the registry: the method as the default, and the
    previous default as the ablation that attributes the difference to it."""
    from evaluation import methods as methods_module

    names = {m.name for m in methods_module.build()}

    assert "erosion" in names
    assert "abl-columns" in names


def test_it_rejects_an_empty_angle_list():
    with pytest.raises(ValueError):
        packer(room()).optimize_erosion(angles=())


def test_one_placement_is_returned_per_angle():
    pk = packer(room(tilt=23.0))

    _, results = pk.optimize_erosion(angles=(0.0, 23.0))

    assert len(results) == 2
    assert sorted(p.angle for p in results) == [0.0, 23.0]
