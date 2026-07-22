"""Tests for packer_service's pure packing functions (no HTTP)."""
import cv2
import numpy as np
import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon

from packer_service import _rotate_angles, run_packing, run_packing_from_image


def make_plan_image_bytes():
    img = np.zeros((200, 300), np.uint8)
    cv2.rectangle(img, (30, 20), (269, 179), 255, thickness=-1)
    cv2.rectangle(img, (100, 80), (139, 109), 0, thickness=-1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_run_packing_rectangle_evenly_divisible():
    result = run_packing(
        shape_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
        obstacle_points=[],
        cell_width=2, cell_height=2, rotate=False,
    )
    assert result["stats"]["complete"] == 25
    assert result["stats"]["partial"] == 0
    assert result["stats"]["coverage"] == pytest.approx(1.0)
    assert result["obstacles"] == []
    assert len(result["shape"]) == 4


def test_run_packing_with_obstacle():
    result = run_packing(
        shape_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
        obstacle_points=[[(2, 2), (6, 2), (6, 6), (2, 6)]],
        cell_width=2, cell_height=2, rotate=False,
    )
    assert len(result["obstacles"]) == 1
    assert result["stats"]["complete"] < 25


def test_run_packing_rejects_short_shape():
    with pytest.raises(ValueError, match="at least 3 points"):
        run_packing([(0, 0), (1, 1)], [], 1, 1, False)


def test_run_packing_rejects_non_positive_cell_size():
    with pytest.raises(ValueError, match="positive"):
        run_packing([(0, 0), (1, 0), (1, 1)], [], 0, 1, False)


def test_square_cells_sweep_a_quarter_turn():
    # rotating a square-celled grid by 90 deg reproduces the same grid,
    # so [0, 90) is the whole search space
    assert _rotate_angles(10, 10) == (0, 15, 30, 45, 60, 75)


def test_rectangular_cells_sweep_a_half_turn():
    # a w x h grid only repeats at 180 deg; 90 deg swaps w and h, which is
    # a genuinely different packing
    assert _rotate_angles(25, 6) == (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165)
    assert _rotate_angles(6, 25) == _rotate_angles(25, 6)


def test_rotation_finds_optimum_past_ninety_degrees():
    # Long thin shape tilted 100 deg, packed with long thin cells. The optimum
    # is 12 complete cells at 105 deg; the best a quarter-turn sweep can reach
    # is 8 at 75 deg. Only a half-turn sweep finds it.
    tilted = shp_rotate(Polygon([(0, 0), (60, 0), (60, 13), (0, 13)]),
                        100, origin="centroid")
    result = run_packing(
        shape_points=list(tilted.exterior.coords)[:-1],
        obstacle_points=[],
        cell_width=12, cell_height=3, rotate=True,
    )
    assert result["stats"]["angle"] >= 90
    assert result["stats"]["complete"] > 8


def test_run_packing_rejects_obstacle_covering_whole_shape():
    with pytest.raises(ValueError, match="no usable area"):
        run_packing(
            shape_points=[(2, 2), (6, 2), (6, 6), (2, 6)],
            obstacle_points=[[(0, 0), (10, 0), (10, 10), (0, 10)]],
            cell_width=2, cell_height=2, rotate=False,
        )


def test_run_packing_rejects_obstacles_covering_shape_between_them():
    with pytest.raises(ValueError, match="no usable area"):
        run_packing(
            shape_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            obstacle_points=[
                [(-1, -1), (11, -1), (11, 5), (-1, 5)],
                [(-1, 5), (11, 5), (11, 11), (-1, 11)],
            ],
            cell_width=2, cell_height=2, rotate=False,
        )


def test_run_packing_from_image_detects_obstacle():
    result = run_packing_from_image(
        make_plan_image_bytes(), cell_width=20, cell_height=20, rotate=False,
    )
    assert len(result["obstacles"]) == 1
    assert result["stats"]["complete"] > 0


def test_run_packing_from_image_rejects_bad_bytes():
    with pytest.raises(ValueError, match="could not read image"):
        run_packing_from_image(b"not an image", cell_width=10, cell_height=10, rotate=False)


def test_run_packing_from_image_rejects_blank_image():
    blank = np.zeros((50, 50), np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    with pytest.raises(ValueError, match="no boundary detected"):
        run_packing_from_image(buf.tobytes(), cell_width=10, cell_height=10, rotate=False)
