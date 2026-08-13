"""Tests for memoising repeated packing requests.

Packing is deterministic: the same outline, obstacles, cell size and flags give
the same placement every time, and the work is entirely CPU. A UI that lets
someone toggle rotation back and forth, or two people asking about the same
floor plan, re-derive an answer already computed -- and with `certify` that is
tens of seconds rather than one.

What is asserted here is what a cache must not get wrong: that it distinguishes
requests that differ in ANY input, that it never returns a different answer than
computing would have, and that it is bounded so a long-lived server cannot grow
without limit.
"""
import pytest

from packer_service import _cache_clear, _cache_info, run_packing

SQUARE = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]
PILLAR = [(3.0, 3.0), (6.0, 3.0), (6.0, 6.0), (3.0, 6.0)]


@pytest.fixture(autouse=True)
def clear():
    _cache_clear()
    yield
    _cache_clear()


def test_an_identical_request_is_served_from_the_cache():
    first = run_packing(SQUARE, [], 3.0, 3.0, rotate=False)
    hits_before = _cache_info().hits

    second = run_packing(SQUARE, [], 3.0, 3.0, rotate=False)

    assert _cache_info().hits == hits_before + 1
    assert second == first


def test_the_cached_answer_is_the_answer_computing_gives():
    """A cache that returns something else is worse than no cache. Compared
    against a run with the cache emptied in between, so the second result is
    genuinely recomputed rather than read back."""
    cached = run_packing(SQUARE, [PILLAR], 3.0, 3.0, rotate=True)
    _cache_clear()
    recomputed = run_packing(SQUARE, [PILLAR], 3.0, 3.0, rotate=True)

    assert cached == recomputed


@pytest.mark.parametrize("changed", [
    {"cell_width": 2.0},
    {"cell_height": 2.0},
    {"rotate": True},
    {"certify": True},
])
def test_every_input_is_part_of_the_key(changed):
    """Anything that changes the answer has to change the key. A cache keyed on
    the shape alone would serve a 3x3 packing to someone asking for 2x3."""
    base = dict(shape_points=SQUARE, obstacle_points=[], cell_width=3.0,
                cell_height=3.0, rotate=False)
    run_packing(**base)
    misses_before = _cache_info().misses

    run_packing(**{**base, **changed})

    assert _cache_info().misses == misses_before + 1


def test_the_obstacles_are_part_of_the_key():
    run_packing(SQUARE, [], 3.0, 3.0, rotate=False)
    misses_before = _cache_info().misses

    run_packing(SQUARE, [PILLAR], 3.0, 3.0, rotate=False)

    assert _cache_info().misses == misses_before + 1


def test_a_differently_wound_outline_is_the_same_request():
    """The same polygon listed from a different starting vertex is the same
    polygon. Missing that would leave the cache useless to a client that
    rebuilds its coordinate list."""
    rotated_listing = SQUARE[2:] + SQUARE[:2]
    run_packing(SQUARE, [], 3.0, 3.0, rotate=False)
    hits_before = _cache_info().hits

    run_packing(rotated_listing, [], 3.0, 3.0, rotate=False)

    assert _cache_info().hits == hits_before + 1


def test_bad_input_is_not_cached_as_a_result():
    with pytest.raises(ValueError):
        run_packing(SQUARE, [], -1.0, 3.0, rotate=False)

    assert _cache_info().currsize == 0


def test_the_cache_is_bounded():
    """A server answering arbitrary outlines must not grow without limit."""
    assert _cache_info().maxsize is not None
    assert _cache_info().maxsize > 0
