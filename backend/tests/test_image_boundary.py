"""Tests for image_boundary.polygons_from_image using synthetic images."""
import numpy as np
import cv2
import pytest

from image_boundary import polygons_from_image


def make_plan_image():
    """White filled 240x160 rectangle on black, with a black 40x30 hole."""
    img = np.zeros((200, 300), np.uint8)
    cv2.rectangle(img, (30, 20), (269, 179), 255, thickness=-1)
    cv2.rectangle(img, (100, 80), (139, 109), 0, thickness=-1)
    return img


def test_filled_plan_shape_and_obstacle():
    shape, obstacles = polygons_from_image(make_plan_image())
    # outer boundary ~ 240x160 rectangle
    assert abs(shape.area - 240 * 160) / (240 * 160) < 0.05
    assert len(obstacles) == 1
    assert abs(obstacles[0].area - 40 * 30) / (40 * 30) < 0.15


def test_y_axis_is_flipped():
    # image row 20 is the TOP of the rectangle; after flip it must be
    # near the MAX y of the polygon (200 - 20 = 180)
    shape, _ = polygons_from_image(make_plan_image())
    miny, maxy = shape.bounds[1], shape.bounds[3]
    assert maxy == pytest.approx(180, abs=3)
    assert miny == pytest.approx(20, abs=3)


def test_dark_shape_on_light_background():
    img = np.full((200, 300), 255, np.uint8)
    cv2.rectangle(img, (30, 20), (269, 179), 0, thickness=-1)
    shape, obstacles = polygons_from_image(img)
    assert abs(shape.area - 240 * 160) / (240 * 160) < 0.05
    assert obstacles == []


def test_sketch_outline_with_gap_is_closed_and_filled():
    # unfilled outline, 3px pen, with a deliberate 3px gap in the top edge
    img = np.zeros((200, 300), np.uint8)
    cv2.rectangle(img, (30, 20), (269, 179), 255, thickness=3)
    img[19:25, 150:153] = 0  # cut a small gap in the stroke
    shape, _ = polygons_from_image(img)
    assert abs(shape.area - 240 * 160) / (240 * 160) < 0.10


def test_small_specks_are_dropped():
    img = make_plan_image()
    cv2.rectangle(img, (285, 5), (289, 9), 255, thickness=-1)   # 5x5 speck outside
    cv2.rectangle(img, (200, 100), (203, 103), 0, thickness=-1)  # 4x4 hole speck
    shape, obstacles = polygons_from_image(img)
    assert len(obstacles) == 1  # speck hole ignored, real hole kept
    assert abs(shape.area - 240 * 160) / (240 * 160) < 0.05


def test_scale_multiplies_coordinates():
    shape1, _ = polygons_from_image(make_plan_image())
    shape2, _ = polygons_from_image(make_plan_image(), scale=0.5)
    assert shape2.area == pytest.approx(shape1.area * 0.25, rel=1e-6)


def test_blank_image_raises():
    with pytest.raises(ValueError, match="no boundary detected"):
        polygons_from_image(np.zeros((100, 100), np.uint8))


def test_bad_path_raises():
    with pytest.raises(ValueError, match="could not read image"):
        polygons_from_image("does_not_exist_xyz.png")


def test_path_input_roundtrip(tmp_path):
    p = str(tmp_path / "plan.png")
    cv2.imwrite(p, make_plan_image())
    shape, obstacles = polygons_from_image(p)
    assert abs(shape.area - 240 * 160) / (240 * 160) < 0.05
    assert len(obstacles) == 1


def test_sketch_interior_is_shape_not_obstacle():
    img = np.zeros((200, 300), np.uint8)
    cv2.rectangle(img, (30, 20), (269, 179), 255, thickness=3)
    _, obstacles = polygons_from_image(img)
    assert obstacles == []


def test_sketch_with_drawn_obstacle():
    img = np.zeros((200, 300), np.uint8)
    cv2.rectangle(img, (30, 20), (269, 179), 255, thickness=3)
    cv2.rectangle(img, (100, 80), (139, 109), 255, thickness=-1)
    shape, obstacles = polygons_from_image(img)
    assert abs(shape.area - 240 * 160) / (240 * 160) < 0.10
    assert len(obstacles) == 1
    assert abs(obstacles[0].area - 40 * 30) / (40 * 30) < 0.30


def test_float_array_input_is_normalized():
    shape, obstacles = polygons_from_image(make_plan_image().astype(np.float64) / 255.0)
    assert abs(shape.area - 240 * 160) / (240 * 160) < 0.05
    assert len(obstacles) == 1
