"""Tests for packer_service's pure packing functions (no HTTP)."""
import cv2
import numpy as np
import pytest

from packer_service import run_packing, run_packing_from_image


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
