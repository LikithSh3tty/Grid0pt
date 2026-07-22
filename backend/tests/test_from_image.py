"""Tests for GridPacker.from_image."""
import numpy as np
import cv2

from grid_packer import GridPacker


def make_plan_image():
    img = np.zeros((200, 300), np.uint8)
    cv2.rectangle(img, (30, 20), (269, 179), 255, thickness=-1)
    cv2.rectangle(img, (100, 80), (139, 109), 0, thickness=-1)
    return img


def test_from_image_builds_working_packer():
    packer = GridPacker.from_image(make_plan_image(),
                                   cell_width=20, cell_height=20)
    assert len(packer.obstacles) == 1
    # usable = outer area minus hole
    assert abs(packer.usable.area - (240 * 160 - 40 * 30)) / (240 * 160) < 0.06

    best, _ = packer.optimize(steps=8)
    # a 240x160 region with 20px cells fits ~12x8 cells; expect a healthy count
    assert best.complete >= 70
    assert best.coverage > 0.7


def test_from_image_passes_scale_through():
    packer = GridPacker.from_image(make_plan_image(), cell_width=10,
                                   cell_height=10, scale=0.5)
    # 240x160 px at 0.5 units/px -> 120x80 units
    minx, miny, maxx, maxy = packer.shape.bounds
    assert abs((maxx - minx) - 120) < 3
    assert abs((maxy - miny) - 80) < 3


def test_from_image_accepts_path(tmp_path):
    p = str(tmp_path / "plan.png")
    cv2.imwrite(p, make_plan_image())
    packer = GridPacker.from_image(p, cell_width=20, cell_height=20)
    assert len(packer.obstacles) == 1
