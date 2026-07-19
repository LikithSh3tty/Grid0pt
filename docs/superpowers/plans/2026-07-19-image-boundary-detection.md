# Image Boundary Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let GridPacker accept an image: detect the outer boundary and interior obstacles with OpenCV, convert them to shapely polygons, and run the existing placement algorithm unchanged.

**Architecture:** A new standalone module `image_boundary.py` converts an image (path or ndarray) into `(shape, obstacles)` shapely polygons via Otsu threshold → gap closing → region fill → contour hierarchy → Douglas-Peucker simplification. A thin `GridPacker.from_image()` classmethod wires it into the existing class. Core geometry code in `grid_packer.py` is otherwise untouched.

**Tech Stack:** Python 3, opencv-python, numpy, shapely, matplotlib, pytest.

## Global Constraints

- New dependency allowed: `opencv-python` only (plus `pytest` as a dev dependency).
- Coordinates returned in **pixel units** with y flipped (`y' = image_height - y`), multiplied by optional `scale` (units per pixel, default `1.0`).
- Error message for an empty detection must be verbatim: `no boundary detected — check contrast or close gaps in the outline`.
- Defaults: `simplify_tol=2.0` (px), `min_area=64.0` (px²), `scale=1.0`.
- Only the largest top-level region becomes the shape; its child contours become obstacles; contours with pixel area < `min_area` are dropped.
- Do not modify `GridPacker.__init__`, `evaluate`, `optimize`, or `plot` behavior.
- Commit messages: plain, no Co-Authored-By trailers.

---

### Task 1: `image_boundary.py` — detection module

**Files:**
- Create: `image_boundary.py`
- Create: `requirements.txt`
- Test: `tests/test_image_boundary.py`

**Interfaces:**
- Consumes: nothing from this codebase.
- Produces: `polygons_from_image(image, *, simplify_tol: float = 2.0, min_area: float = 64.0, scale: float = 1.0) -> tuple[shapely.geometry.Polygon, list[shapely.geometry.Polygon]]` where `image` is a `str` path or a numpy `ndarray` (grayscale or BGR). Task 2 imports exactly this name from `image_boundary`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_boundary.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_boundary.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'image_boundary'` (install `opencv-python` and `pytest` first if missing: `pip install opencv-python pytest`).

- [ ] **Step 3: Write the implementation**

Create `image_boundary.py`:

```python
"""
image_boundary.py
=================

Turn a raster image (clean plan or hand-drawn sketch) into shapely polygons
usable by GridPacker: the outer boundary becomes the shape, enclosed interior
regions become obstacles.

Pipeline: grayscale -> Otsu threshold (auto polarity) -> morphological close
(seals gaps in drawn outlines) -> flood-fill to solidify enclosed regions ->
contour hierarchy (RETR_CCOMP) -> Douglas-Peucker simplification -> y-flip
and optional unit scaling.

Requires: opencv-python, numpy, shapely
"""

from __future__ import annotations

from typing import List, Tuple, Union

import cv2
import numpy as np
from shapely.geometry import Polygon

Image = Union[str, np.ndarray]


def _to_grayscale(image: Image) -> np.ndarray:
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"could not read image: {image}")
        return img
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold, then pick polarity so the foreground (shape) is white.
    Heuristic: the image border is overwhelmingly background, so if the
    border comes out mostly white, the threshold polarity is inverted."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if border.mean() > 127:
        bw = cv2.bitwise_not(bw)
    return bw


def _solidify(bw: np.ndarray) -> np.ndarray:
    """Close small gaps in drawn outlines, then fill every enclosed region so
    an unfilled sketch outline becomes a solid mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # flood-fill the background starting from a background border pixel;
    # anything the fill cannot reach is enclosed -> becomes foreground
    h, w = bw.shape
    seed = None
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if bw[y, x] == 0:
            seed = (x, y)
            break
    if seed is None:  # foreground touches all corners; nothing to fill
        return bw
    ff = bw.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, seed, 255)
    return bw | cv2.bitwise_not(ff)


def _contour_to_polygon(cnt: np.ndarray, simplify_tol: float,
                        img_h: int, scale: float) -> Polygon | None:
    """Simplify a cv2 contour and convert to a valid shapely Polygon in
    y-up coordinates. Returns None for degenerate contours."""
    approx = cv2.approxPolyDP(cnt, simplify_tol, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    pts = [(float(x) * scale, float(img_h - y) * scale) for x, y in approx]
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    return poly


def polygons_from_image(
    image: Image,
    *,
    simplify_tol: float = 2.0,
    min_area: float = 64.0,
    scale: float = 1.0,
) -> Tuple[Polygon, List[Polygon]]:
    """Detect the outer boundary and interior obstacles in an image.

    image        : path to an image file, or a numpy array (grayscale or BGR).
    simplify_tol : Douglas-Peucker tolerance in pixels; higher = simpler polygons.
    min_area     : contours below this pixel area are treated as noise.
    scale        : units per pixel; multiplies all output coordinates.

    Returns (shape, obstacles): the largest detected region as a shapely
    Polygon, and the enclosed holes inside it as obstacle Polygons, in y-up
    coordinates scaled by `scale`.
    """
    gray = _to_grayscale(image)
    solid = _solidify(_binarize(gray))
    img_h = gray.shape[0]

    contours, hierarchy = cv2.findContours(
        solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(
            "no boundary detected — check contrast or close gaps in the outline")
    hierarchy = hierarchy[0]  # (N, 4): [next, prev, first_child, parent]

    # largest top-level contour above the noise floor -> the shape
    top = [i for i, hi in enumerate(hierarchy)
           if hi[3] == -1 and cv2.contourArea(contours[i]) >= min_area]
    if not top:
        raise ValueError(
            "no boundary detected — check contrast or close gaps in the outline")
    shape_idx = max(top, key=lambda i: cv2.contourArea(contours[i]))

    shape = _contour_to_polygon(contours[shape_idx], simplify_tol, img_h, scale)
    if shape is None:
        raise ValueError(
            "no boundary detected — check contrast or close gaps in the outline")

    obstacles: List[Polygon] = []
    for i, hi in enumerate(hierarchy):
        if hi[3] != shape_idx or cv2.contourArea(contours[i]) < min_area:
            continue
        poly = _contour_to_polygon(contours[i], simplify_tol, img_h, scale)
        if poly is not None:
            obstacles.append(poly)

    return shape, obstacles
```

Create `requirements.txt`:

```
numpy
shapely
matplotlib
opencv-python
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_image_boundary.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add image_boundary.py requirements.txt tests/test_image_boundary.py
git commit -m "Add image boundary detection module"
```

---

### Task 2: `GridPacker.from_image()` classmethod

**Files:**
- Modify: `grid_packer.py` (add a classmethod to `GridPacker`, directly below `__init__` which ends near line 92)
- Test: `tests/test_from_image.py`

**Interfaces:**
- Consumes: `polygons_from_image(image, *, simplify_tol, min_area, scale)` from `image_boundary` (Task 1).
- Produces: `GridPacker.from_image(image, *, cell_width: float = 1.0, cell_height: float = 1.0, simplify_tol: float = 2.0, min_area: float = 64.0, scale: float = 1.0) -> GridPacker`. Task 3's demo calls exactly this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_from_image.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_from_image.py -v`
Expected: FAIL with `AttributeError: type object 'GridPacker' has no attribute 'from_image'`.

- [ ] **Step 3: Add the classmethod**

In `grid_packer.py`, inside class `GridPacker`, immediately after `__init__` (after the line `self._pivot = self.shape.centroid`), add:

```python
    @classmethod
    def from_image(
        cls,
        image,
        *,
        cell_width: float = 1.0,
        cell_height: float = 1.0,
        simplify_tol: float = 2.0,
        min_area: float = 64.0,
        scale: float = 1.0,
    ) -> "GridPacker":
        """Build a GridPacker from an image (file path or numpy array).

        The outer boundary detected in the image becomes the shape; enclosed
        interior regions become obstacles. Coordinates are in pixels (y-up)
        unless `scale` (units per pixel) is given. See
        image_boundary.polygons_from_image for the detection parameters.
        """
        from image_boundary import polygons_from_image

        shape, obstacles = polygons_from_image(
            image, simplify_tol=simplify_tol, min_area=min_area, scale=scale)
        return cls(shape, obstacles,
                   cell_width=cell_width, cell_height=cell_height)
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add grid_packer.py tests/test_from_image.py
git commit -m "Add GridPacker.from_image classmethod"
```

---

### Task 3: Image demo in `demo.py`

**Files:**
- Modify: `demo.py` (add `image_example()` before the `__main__` block at line 61; extend the `__main__` block)

**Interfaces:**
- Consumes: `GridPacker.from_image(image, *, cell_width, cell_height)` (Task 2).
- Produces: `demo_plan.png` (synthesized input image) and `result_image.png` (packing result) in the repo root.

- [ ] **Step 1: Add the demo function**

In `demo.py`, add after `l_shape_example()` (its `plt.close(fig)` at line 58) and before `if __name__ == "__main__":`:

```python
def image_example():
    # synthesize a "floor plan": white L-shaped room with two dark obstacles,
    # then let from_image() detect the boundary and run the optimizer on it
    import cv2
    import numpy as np

    img = np.zeros((400, 600), np.uint8)
    cv2.fillPoly(img, [np.array([(40, 40), (560, 40), (560, 220),
                                 (300, 220), (300, 360), (40, 360)])], 255)
    cv2.rectangle(img, (120, 120), (200, 180), 0, thickness=-1)
    cv2.circle(img, (450, 130), 35, 0, thickness=-1)
    cv2.imwrite("demo_plan.png", img)

    packer = GridPacker.from_image("demo_plan.png", cell_width=40, cell_height=40)
    best, _ = packer.optimize(steps=10)

    print("\nIMAGE")
    print(f"  detected : shape area={packer.shape.area:.0f}px², "
          f"{len(packer.obstacles)} obstacle(s)")
    print("  optimized:", best)

    fig, ax = plt.subplots(figsize=(10, 7))
    packer.plot(best, ax=ax)
    fig.tight_layout()
    fig.savefig("result_image.png", dpi=110)
    plt.close(fig)
```

And change the `__main__` block to:

```python
if __name__ == "__main__":
    rectangle_example()
    l_shape_example()
    image_example()
    print("\nSaved: result_rectangle.png, result_lshape.png, result_image.png")
```

- [ ] **Step 2: Run the demo and verify output**

Run: `python demo.py`
Expected: prints the RECTANGLE and L-SHAPE sections as before, then an `IMAGE` section reporting 2 obstacle(s) and a Placement with `complete >= 40` and coverage above 60%; `demo_plan.png` and `result_image.png` exist afterwards.

- [ ] **Step 3: Visually verify the overlay**

Open `result_image.png` and confirm: the green/orange grid sits inside the L-shaped outline, the rectangular and circular obstacles are dark and avoided by complete cells, and the image is not upside-down (the L's foot is at the bottom-left, matching `demo_plan.png` flipped vertically — i.e. same orientation as the plan viewed normally).

- [ ] **Step 4: Run the full test suite once more**

Run: `python -m pytest tests/ -v`
Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add demo.py demo_plan.png result_image.png
git commit -m "Add image-input demo"
```
