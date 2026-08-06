"""Tests for what the packing endpoint reports (design note section 10).

The solver swap has to be invisible to existing clients and legible to the
paper. So this module pins both halves: the response SHAPE is unchanged -- same
keys, same geometry -- while `stats` gains the vote's confidence and the
placement's optimality certificate as additive fields.
"""
import math

import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon

from packer_service import run_packing


ROOM = [(0, 0), (12, 0), (12, 9), (0, 9)]


def _tilted(points, degrees):
    """Rotate a point list about its centroid, at full precision.

    The coordinates are NOT rounded for readability: a 12x9 room tiles exactly
    into 12 cells only while it is exactly 12x9, and rounding its corners to
    four decimals is enough to cost three complete cells.
    """
    poly = Polygon(points)
    turned = shp_rotate(poly, degrees, origin=poly.centroid)
    return [(x, y) for x, y in turned.exterior.coords[:-1]]


#: A room tilted 23 degrees. Not a multiple of the old 15-degree ladder, so the
#: fixed-angle baseline could never reach it however finely it swept offsets.
TILTED_ROOM = _tilted(ROOM, 23)


def stats_for(points, *, cell=3.0, rotate=False, obstacles=()):
    return run_packing(points, list(obstacles), cell, cell, rotate)


# --------------------------------------------------------------------- #
# the response shape is unchanged
# --------------------------------------------------------------------- #
def test_the_response_keeps_its_shape():
    result = stats_for(ROOM)

    assert set(result) == {"shape", "obstacles", "complete_cells",
                           "partial_cells", "stats"}
    assert result["shape"] == [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]
    assert result["obstacles"] == []
    assert len(result["complete_cells"]) == result["stats"]["complete"]
    assert len(result["partial_cells"]) == result["stats"]["partial"]


def test_the_original_stats_are_all_still_there():
    """Existing clients read these six; none may disappear or change meaning."""
    stats = stats_for(ROOM)["stats"]

    for key in ("complete", "partial", "coverage", "dx", "dy", "angle"):
        assert key in stats

    assert stats["complete"] == 12          # the room tiles exactly
    assert stats["partial"] == 0
    assert stats["coverage"] == pytest.approx(1.0)
    assert stats["angle"] == 0.0


# --------------------------------------------------------------------- #
# the certificate reaches the API
# --------------------------------------------------------------------- #
def test_the_certificate_is_reported():
    stats = stats_for(ROOM)["stats"]

    assert stats["irreducible"] == 0
    assert stats["partial_floor"] == 0
    assert stats["optimality_gap"] == 0     # certified optimal, no partials
    assert stats["certified"] is True
    assert stats["recoverable_area"] == pytest.approx(0.0)


def test_the_gap_is_never_negative_on_a_harder_instance():
    """A floor above the achieved partial count would be a false bound, and it
    would surface here as a negative gap."""
    result = stats_for(ROOM, cell=2.5, obstacles=[[(4, 3), (8, 3), (8, 6), (4, 6)]])
    stats = result["stats"]

    assert stats["partial"] > 0
    assert stats["optimality_gap"] >= 0
    assert stats["partial_floor"] <= stats["partial"]


# --------------------------------------------------------------------- #
# rotation: what the vote reports, and when
# --------------------------------------------------------------------- #
def test_rotation_diagnostics_appear_only_when_rotation_was_asked_for():
    """With rotate off there is no vote, so the confidence fields are absent
    rather than present and meaningless."""
    fixed = stats_for(ROOM)["stats"]
    turned = stats_for(ROOM, rotate=True)["stats"]

    for key in ("resultant", "rotated", "evaluations"):
        assert key not in fixed
        assert key in turned


def test_a_tilted_room_is_turned_onto_its_own_walls():
    """The end-to-end version of the headline: the endpoint returns a grid
    aligned to the walls, and says how confident the fringe was."""
    straight = stats_for(TILTED_ROOM)["stats"]
    turned = stats_for(TILTED_ROOM, rotate=True)["stats"]

    assert straight["angle"] == 0.0
    assert straight["complete"] == 6

    assert turned["angle"] == pytest.approx(23.0, abs=1.0)
    assert turned["complete"] == 12
    assert turned["rotated"] is True
    assert turned["resultant"] > 0.9


def test_a_shape_with_no_dominant_wall_is_left_alone():
    """Below the confidence gate the grid deliberately stays put, and the stats
    say so rather than silently reporting angle 0 as a choice."""
    import math

    disc = [(6 * math.cos(2 * math.pi * i / 32), 6 * math.sin(2 * math.pi * i / 32))
            for i in range(32)]

    turned = stats_for(disc, rotate=True)["stats"]

    assert turned["rotated"] is False
    assert turned["resultant"] < 0.35
    assert turned["angle"] == 0.0


def test_rotating_never_costs_completed_cells():
    """The never-worse guarantee, as seen from the endpoint."""
    for points in (ROOM, TILTED_ROOM):
        fixed = stats_for(points)["stats"]
        turned = stats_for(points, rotate=True)["stats"]

        assert turned["complete"] >= fixed["complete"]
