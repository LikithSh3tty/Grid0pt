"""Tests for the exact translation search (Method 1: critical offsets).

The objective N_complete(dx, dy) is piecewise-constant in the grid offsets: a
cell can only flip between complete and partial when a grid line touches a
vertex of the usable region. The critical set is therefore exhaustive, not a
sample, and these tests pin that claim down against a dense uniform sweep.
"""
import numpy as np
import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import MultiPolygon, Polygon

from grid_packer import GridPacker, _face_samples


# --------------------------------------------------------------------- #
# instances
# --------------------------------------------------------------------- #
def l_shape() -> Polygon:
    """A 12x12 L: full-width lower arm, half-width upper arm."""
    return Polygon([(0, 0), (12, 0), (12, 6), (6, 6), (6, 12), (0, 12)])


def off_grid_obstacle() -> Polygon:
    """An obstacle whose edges do NOT sit on cell boundaries, so it
    contributes critical offsets of its own."""
    return Polygon([(7, 1), (10.5, 1), (10.5, 2.5), (7, 2.5)])


# Cost is read off `GridPacker.evaluations`, the packer's own counter. This file
# used to wrap the packer in a subclass to count for itself; the counter now
# lives on the packer because the evaluation harness needs the same number for
# methods whose cost is not a product of loop bounds, and two counters would be
# two chances to be wrong about the paper's cost column.


# Offsets per axis in the ground-truth uniform sweep. The narrowest
# constant-count face on the headline instance is 0.5 wide out of a period of
# 3, so 80 samples per axis (step 0.0375) resolves every face more than ten
# times over -- dense enough to be a fair ground truth, cheap enough for CI.
# The same comparison at 150 and 200 per axis returns the same optimum; it just
# takes minutes rather than seconds.
DENSE_STEPS = 80
ROTATED_DENSE_STEPS = 40   # rotated evaluations cost ~3x (cells are rotated back)


def dense_sweep_best(packer, steps, angle=0.0):
    """Ground truth: the best complete count over a dense uniform grid of
    offsets, computed straight through evaluate()."""
    best = -1
    for dx in np.linspace(0, packer.cw, steps, endpoint=False):
        for dy in np.linspace(0, packer.ch, steps, endpoint=False):
            best = max(best, packer.evaluate(dx, dy, angle).complete)
    return best


# --------------------------------------------------------------------- #
# headline regression: exact == dense sweep, at a fraction of the cost
# --------------------------------------------------------------------- #
def test_exact_matches_dense_sweep_on_l_shape_with_obstacle():
    """The paper's headline claim, on the L-shape + obstacle instance.

    A dense uniform sweep and the critical set must return the SAME optimum,
    with the critical set using orders of magnitude fewer evaluations. This is
    the slowest test in the file by design: the dense sweep is the ground truth.
    """
    shape, obstacle = l_shape(), off_grid_obstacle()

    exact_packer = GridPacker(shape, [obstacle], cell_width=3, cell_height=3)
    best, results = exact_packer.optimize_exact()

    dense_packer = GridPacker(shape, [obstacle], cell_width=3, cell_height=3)
    dense_best = dense_sweep_best(dense_packer, steps=DENSE_STEPS)

    # the optimum itself: identical
    assert best.complete == dense_best

    # every placement the exact search returned is a real evaluation, and the
    # best of them is the reported best
    assert best.complete == max(p.complete for p in results)
    assert exact_packer.evaluations == len(results)

    # ...and it cost at least an order of magnitude less
    assert dense_packer.evaluations == DENSE_STEPS * DENSE_STEPS
    assert exact_packer.evaluations * 10 < dense_packer.evaluations


def test_exact_cannot_be_beaten_by_a_dense_sweep():
    """A dense sweep only ever samples points inside faces the exact search
    already visits, so exact >= dense always -- never the other way round."""
    packer = GridPacker(l_shape(), [off_grid_obstacle()], cell_width=3, cell_height=3)
    best, _ = packer.optimize_exact()
    assert best.complete >= dense_sweep_best(packer, steps=37)


# --------------------------------------------------------------------- #
# exact vs the retained brute-force baseline
# --------------------------------------------------------------------- #
BASELINE_INSTANCES = {
    "rectangle": (Polygon([(0.4, 0.3), (11.4, 0.3), (11.4, 8.3), (0.4, 8.3)]), []),
    "l_shape": (l_shape(), []),
    "l_shape_with_obstacle": (l_shape(), [off_grid_obstacle()]),
    "two_obstacles": (
        Polygon([(0, 0), (12, 0), (12, 12), (0, 12)]),
        [
            Polygon([(1.3, 1.7), (4.3, 1.7), (4.3, 4.2), (1.3, 4.2)]),
            Polygon([(7.5, 6.5), (10.9, 6.5), (10.9, 9.1), (7.5, 9.1)]),
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(BASELINE_INSTANCES))
def test_exact_is_at_least_as_good_as_the_uniform_sweep(name):
    shape, obstacles = BASELINE_INSTANCES[name]
    packer = GridPacker(shape, obstacles, cell_width=3, cell_height=3)

    baseline, _ = packer.optimize(steps=10)
    exact, _ = packer.optimize_exact()

    assert exact.complete >= baseline.complete


# --------------------------------------------------------------------- #
# _critical_offsets: where the vertices come from
# --------------------------------------------------------------------- #
def test_critical_offsets_are_read_from_the_usable_region():
    """Offsets must include the obstacle's edges -- they live on an INTERIOR
    ring of `usable`, which a walk over exteriors alone would miss."""
    packer = GridPacker(l_shape(), [off_grid_obstacle()], cell_width=3, cell_height=3)
    assert packer.usable.geom_type == "Polygon"
    assert len(packer.usable.interiors) == 1

    xs, ys = packer._critical_offsets()
    assert xs == [0.0, 1.0, 1.5]      # 0/6/12 -> 0; obstacle 7 -> 1, 10.5 -> 1.5
    assert ys == [0.0, 1.0, 2.5]      # 0/6/12 -> 0; obstacle 1 -> 1, 2.5 -> 2.5


def test_critical_offsets_walk_every_polygon_of_a_multipolygon():
    """An obstacle that cuts the shape in two makes `usable` a MultiPolygon;
    the vertices of BOTH pieces are critical."""
    shape = Polygon([(0, 0), (12, 0), (12, 9), (0, 9)])
    divider = Polygon([(4.4, -1), (5.9, -1), (5.9, 10), (4.4, 10)])
    packer = GridPacker(shape, [divider], cell_width=3, cell_height=3)

    assert isinstance(packer.usable, MultiPolygon)
    assert len(packer.usable.geoms) == 2

    xs, ys = packer._critical_offsets()
    # 0/12 -> 0 (left piece), 4.4 -> 1.4 and 5.9 -> 2.9 (the cut walls)
    assert xs == [0.0, 1.4, 2.9]
    assert ys == [0.0]

    best, _ = packer.optimize_exact()
    assert best.complete > 0


def test_critical_offsets_are_reduced_into_one_cell_period():
    packer = GridPacker(l_shape(), [off_grid_obstacle()], cell_width=3, cell_height=3)
    xs, ys = packer._critical_offsets()
    assert all(0.0 <= v < packer.cw for v in xs)
    assert all(0.0 <= v < packer.ch for v in ys)
    assert xs == sorted(set(xs))
    assert ys == sorted(set(ys))


# --------------------------------------------------------------------- #
# _face_samples
# --------------------------------------------------------------------- #
def test_face_samples_put_one_sample_strictly_inside_every_interval():
    period = 3.0
    vals = [0.0, 1.0, 2.5]
    samples = _face_samples(vals, period)

    # the intervals of the circle cut by the critical values, wrap included
    intervals = [(0.0, 1.0), (1.0, 2.5), (2.5, period)]
    for lo, hi in intervals:
        assert any(lo < s < hi for s in samples), f"no sample inside ({lo}, {hi})"


def test_face_samples_keep_the_critical_values_themselves():
    samples = _face_samples([0.0, 1.0, 2.5], 3.0)
    assert {0.0, 1.0, 2.5}.issubset(set(samples))


def test_face_samples_stay_inside_one_period():
    for vals, period in ([0.0, 1.0, 2.5], 3.0), ([0.7], 4.0), ([], 2.0):
        samples = _face_samples(vals, period)
        assert samples, "face samples must never be empty"
        assert all(0.0 <= s < period for s in samples)
        assert samples == sorted(set(samples))


def test_face_samples_handle_degenerate_input():
    # no vertices at all: the whole circle is one face, plus the 0 reference
    assert _face_samples([], 3.0) == [0.0, 1.5]
    # a single critical value: two faces
    assert _face_samples([1.0], 4.0) == [0.0, 0.5, 1.0, 2.5]


def test_face_samples_wrap_values_outside_the_period():
    # 7 mod 3 == 1: the same grid line, so the same face structure
    assert _face_samples([7.0], 3.0) == _face_samples([1.0], 3.0)


def test_face_samples_rejects_a_non_positive_period():
    with pytest.raises(ValueError, match="period must be positive"):
        _face_samples([0.0], 0.0)


# --------------------------------------------------------------------- #
# rotation: the critical set must come from the ROTATED geometry
# --------------------------------------------------------------------- #
THETA = 23.0


def tilted_instance():
    """A 12.9 x 9.9 rectangle whose corners sit at (0.7, 0.4), tilted by THETA
    about its own centroid -- which is also the packer's rotation pivot, so
    evaluating at angle=THETA rotates it back to axis-aligned."""
    rect = Polygon([(0.7, 0.4), (13.6, 0.4), (13.6, 10.3), (0.7, 10.3)])
    return shp_rotate(rect, THETA, origin="centroid")


def test_critical_offsets_at_an_angle_come_from_the_rotated_geometry():
    packer = GridPacker(tilted_instance(), cell_width=3, cell_height=3)

    at_angle = packer._critical_offsets(THETA)
    at_zero = packer._critical_offsets(0.0)

    # evaluate() analyses the region rotated by -angle, so at THETA the grid
    # sees the ORIGINAL axis-aligned rectangle: 0.7 and 13.6 mod 3 -> 0.7, 1.6;
    # 0.4 and 10.3 mod 3 -> 0.4, 1.3.
    assert at_angle[0] == pytest.approx([0.7, 1.6])
    assert at_angle[1] == pytest.approx([0.4, 1.3])

    # the angle-0 offsets (the tilted corners) are a completely different set
    assert at_zero != at_angle
    assert not set(at_zero[0]) & set(at_angle[0])
    assert not set(at_zero[1]) & set(at_angle[1])


def test_exact_at_an_angle_matches_a_dense_sweep_at_that_angle():
    packer = GridPacker(tilted_instance(), cell_width=3, cell_height=3)

    best, _ = packer.optimize_exact(angles=(THETA,))
    assert best.angle == THETA
    assert best.complete == dense_sweep_best(
        packer, steps=ROTATED_DENSE_STEPS, angle=THETA)


def test_exact_recovers_a_sharp_optimum_the_angle_zero_offsets_miss():
    """A 12x9 rectangle tilted by THETA: complete cells only tile it when the
    grid is flush with its walls, an optimum of measure zero in (dx, dy).
    Using the angle-0 critical offsets at THETA -- the bug this guards against
    -- searches the wrong set of lines and lands short."""
    rect = Polygon([(0.7, 0.4), (12.7, 0.4), (12.7, 9.4), (0.7, 9.4)])
    packer = GridPacker(shp_rotate(rect, THETA, origin="centroid"),
                        cell_width=3, cell_height=3)

    best, _ = packer.optimize_exact(angles=(THETA,))

    wrong_x, wrong_y = packer._critical_offsets(0.0)
    naive = max(
        packer.evaluate(dx, dy, THETA).complete
        for dx in _face_samples(wrong_x, packer.cw)
        for dy in _face_samples(wrong_y, packer.ch)
    )
    assert best.complete > naive


# --------------------------------------------------------------------- #
# contract: same shape of result as optimize()
# --------------------------------------------------------------------- #
def test_optimize_exact_returns_all_placements_sorted_best_first():
    packer = GridPacker(l_shape(), [off_grid_obstacle()], cell_width=3, cell_height=3)
    best, results = packer.optimize_exact()

    assert results[0] is best
    scores = [(p.complete, -p.partial) for p in results]
    assert scores == sorted(scores, reverse=True)


def test_optimize_exact_penalizes_partials_when_asked():
    packer = GridPacker(l_shape(), [off_grid_obstacle()], cell_width=3, cell_height=3)
    _, results = packer.optimize_exact(partial_penalty=0.5)
    scores = [p.complete - 0.5 * p.partial for p in results]
    assert scores == sorted(scores, reverse=True)


def test_optimize_exact_covers_several_angles():
    packer = GridPacker(l_shape(), cell_width=3, cell_height=3)
    best, results = packer.optimize_exact(angles=(0.0, 30.0))
    assert {p.angle for p in results} == {0.0, 30.0}
    assert best.complete == max(p.complete for p in results)


def test_optimize_exact_rejects_an_empty_angle_list():
    packer = GridPacker(l_shape(), cell_width=3, cell_height=3)
    with pytest.raises(ValueError, match="at least one angle"):
        packer.optimize_exact(angles=())


def test_optimize_does_not_change():
    """optimize() is retained unchanged as the paper's ablation baseline:
    a uniform linspace sweep, unioned with the shape/obstacle snap offsets."""
    packer = GridPacker(l_shape(), [off_grid_obstacle()], cell_width=3, cell_height=3)
    best, results = packer.optimize(steps=10)

    vx, vy = packer._vertex_offsets()
    uniform = set(np.linspace(0, 3, 10, endpoint=False))
    assert len(results) == len(uniform | set(vx)) * len(uniform | set(vy))
    assert len(results) == 132
    assert best.complete == 10
