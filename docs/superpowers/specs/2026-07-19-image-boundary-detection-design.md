# Image Boundary Detection for GridPacker — Design

**Date:** 2026-07-19
**Status:** Approved

## Goal

Extend the grid packer so it accepts an image (clean drawing/plan or hand-drawn
sketch) as input: detect the outer boundary and interior obstacles from the
image, convert them to shapely polygons, and run the existing placement
algorithm unchanged.

## Scope

- **In:** boundary + obstacle detection from raster images; a `from_image`
  convenience constructor; pixel coordinates with optional real-unit scale;
  a demo exercising the flow.
- **Out:** photographs of real scenes (segmentation), multi-shape images
  (only the largest region is used), GUI/interactive tooling.

## Architecture

Two units:

1. **`image_boundary.py`** (new module) — image in, polygons out.
   - `polygons_from_image(path_or_array, *, simplify_tol=2.0, min_area=64.0, scale=1.0) -> tuple[Polygon, list[Polygon]]`
   - Depends on: `opencv-python`, `numpy`, `shapely`. No knowledge of GridPacker.
2. **`GridPacker.from_image(...)`** (classmethod on existing class) — thin
   wrapper: calls `polygons_from_image`, passes results plus `cell_width` /
   `cell_height` through to `__init__`. Core geometry code is untouched.

## Detection pipeline (inside `polygons_from_image`)

1. Load image (path via `cv2.imread`, or accept an already-loaded ndarray);
   convert to grayscale.
2. **Otsu threshold**, auto-detecting polarity (shape may be dark-on-light or
   light-on-dark; pick the polarity where foreground does not touch most of
   the border).
3. **Morphological closing** (small kernel, ~5px) to seal gaps in hand-drawn
   outlines.
4. **Fill enclosed regions** (flood fill from the border; anything not reached
   is interior) so an unfilled sketch outline becomes a solid mask.
5. `cv2.findContours` with `RETR_CCOMP`: top-level contour with the largest
   area → **shape**; its child contours → **obstacles**.
6. Drop contours with area < `min_area` (specks/noise).
7. **Douglas-Peucker** simplification (`approxPolyDP`, epsilon =
   `simplify_tol` pixels) to clean pixel staircases.
8. Flip y (`y' = image_height - y`) so plots are not upside-down; multiply all
   coordinates by `scale` (units per pixel, default 1.0 = pixel units).
9. Validate polygons (`buffer(0)` fix-ups); return `(shape, obstacles)`.

## Error handling

- Unreadable path / unsupported format → `ValueError` with the path.
- No closed region found after processing → `ValueError`:
  "no boundary detected — check contrast or close gaps in the outline".
- Degenerate/self-intersecting contours are repaired with `buffer(0)`;
  if repair yields an empty geometry the contour is skipped.
- The polygon-based API is unchanged, so image failures cannot affect it.

## Demo & verification

- `demo.py` gains `image_example()`: synthesizes a test image (filled shape
  with holes, drawn with cv2), runs `GridPacker.from_image`, optimizes, and
  saves `result_image.png`.
- Verification: run demo; confirm detected polygon count, obstacle count, and
  a plausible complete-cell result; visually confirm the PNG overlay aligns
  with the source image.

## Dependencies

Adds `opencv-python` to existing `shapely`, `numpy`, `matplotlib`.

## Usage

```python
packer = GridPacker.from_image("floorplan.png", cell_width=40, cell_height=40)
best, _ = packer.optimize(steps=12, angles=range(0, 90, 15))
packer.plot(best)
```
