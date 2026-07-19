"""
image_boundary.py
=================

Turn a raster image (clean plan or hand-drawn sketch) into shapely polygons
usable by GridPacker: the outer boundary becomes the shape, enclosed interior
regions become obstacles.

Pipeline: grayscale -> Otsu threshold (auto polarity) -> morphological close
(seals gaps in drawn outlines) -> contour hierarchy (RETR_TREE) with ring
detection (an unfilled outline collapses to its enclosed interior) ->
Douglas-Peucker simplification -> y-flip and optional unit scaling.

Requires: opencv-python, numpy, shapely
"""

from __future__ import annotations

from typing import List, Tuple, Union

import cv2
import numpy as np
from shapely.geometry import Polygon

Image = Union[str, np.ndarray]

DETECTION_ERROR = "no boundary detected — check contrast or close gaps in the outline"

# a contour whose child covers at least this fraction of it is an unfilled
# outline (pen-stroke ring); the usable region is the child interior
RING_RATIO = 0.85


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
    """Close small gaps in drawn outlines using morphological operations.
    Preserves holes inside shapes so they can be detected as obstacles."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    return bw


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


def _children(hierarchy: np.ndarray, idx: int) -> List[int]:
    """Indices of the direct children of contour `idx` in a RETR_TREE hierarchy."""
    out = []
    child = hierarchy[idx][2]
    while child != -1:
        out.append(child)
        child = hierarchy[child][0]
    return out


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
        solid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(DETECTION_ERROR)
    hierarchy = hierarchy[0]  # (N, 4): [next, prev, first_child, parent]

    # largest top-level contour above the noise floor -> the shape
    top = [i for i, h in enumerate(hierarchy)
           if h[3] == -1 and cv2.contourArea(contours[i]) >= min_area]
    if not top:
        raise ValueError(DETECTION_ERROR)
    shape_idx = max(top, key=lambda i: cv2.contourArea(contours[i]))
    shape_area = cv2.contourArea(contours[shape_idx])

    # ring detection: an unfilled outline is a thin ring whose inner edge
    # encloses almost the same area as its outer edge; the usable region is
    # that interior, and anything drawn inside it becomes an obstacle
    inner = [i for i in _children(hierarchy, shape_idx)
             if cv2.contourArea(contours[i]) >= RING_RATIO * shape_area]
    if inner:
        shape_idx = max(inner, key=lambda i: cv2.contourArea(contours[i]))

    shape = _contour_to_polygon(contours[shape_idx], simplify_tol, img_h, scale)
    if shape is None:
        raise ValueError(DETECTION_ERROR)

    obstacles: List[Polygon] = []
    for i in _children(hierarchy, shape_idx):
        if cv2.contourArea(contours[i]) < min_area:
            continue
        poly = _contour_to_polygon(contours[i], simplify_tol, img_h, scale)
        if poly is not None:
            obstacles.append(poly)

    return shape, obstacles
