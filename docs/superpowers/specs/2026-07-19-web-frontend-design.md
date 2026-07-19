# Grid Packer Web Frontend — Design

**Date:** 2026-07-19
**Status:** Approved

## Goal

A simple local web UI to try the grid packer: upload an image OR enter a
polygon manually (click-to-draw with a synced text panel), set the grid cell
size, run the optimizer, and see the packed result.

## Scope

- **In:** FastAPI backend wrapping the existing `GridPacker`; React (Vite)
  frontend with Image and Draw input modes; cell width/height controls;
  rotation toggle; SVG result rendering with stats.
- **Out:** deployment/hosting, persistence, auth, multi-shape images,
  exposing search parameters (steps/angles) in the UI, editing detected
  image polygons.

## Architecture

Two units:

1. **`server.py`** (new, FastAPI) — API + static host. Run with
   `python server.py` → serves the built frontend at `http://localhost:8000`.
2. **`frontend/`** (new, Vite + React, no UI framework) — the browser app;
   built output in `frontend/dist` is what the server serves.

## Backend API

- `POST /api/pack/image` — multipart form: `file` (image),
  `cell_width` (float), `cell_height` (float), `rotate` (bool).
  Pipeline: `GridPacker.from_image(bytes → cv2.imdecode)` → `optimize()`.
- `POST /api/pack/polygon` — JSON body:
  `{shape: [[x,y],…], obstacles: [[[x,y],…],…], cell_width, cell_height, rotate}`.
  Pipeline: `GridPacker(Polygon(shape), obstacles)` → `optimize()`.
- Both return the same JSON:

```json
{
  "shape": [[x, y], ...],
  "obstacles": [[[x, y], ...], ...],
  "complete_cells": [[[x, y], ...], ...],
  "partial_cells": [[[x, y], ...], ...],
  "stats": {"complete": 0, "partial": 0, "coverage": 0.0,
             "dx": 0.0, "dy": 0.0, "angle": 0.0}
}
```

- Search parameters are fixed server-side: `steps=10`; angles `(0,)` when
  `rotate` is false, `range(0, 90, 15)` when true.
- Errors: invalid input (unparseable image, <3 shape points, non-positive
  cell size) → HTTP 400 with `{"detail": "<message>"}`; detection failures
  surface `image_boundary`'s message verbatim.
- Cells are returned as polygon coordinate lists (not bboxes) so rotated
  placements render correctly.
- New dependencies: `fastapi`, `uvicorn`, `python-multipart`.

## Frontend

Single page, two tabs:

- **Image tab:** file picker (browse or drag-drop), image preview, Run.
- **Draw tab:** SVG canvas — click to place vertices; close the current
  polygon via a "Close polygon" button or clicking the first vertex. First
  closed polygon is the shape; each subsequent closed polygon is an
  obstacle. Undo-last-point and Clear buttons. Beside the canvas, a text
  panel — line 1 is the shape, each further line an obstacle, points as
  `x,y x,y x,y …` — editable and synced both ways (canvas click updates
  text; valid text edits update canvas; invalid text shows an inline hint
  and does not clear the canvas).

Shared controls above the tabs: cell width, cell height (number inputs,
default 1 for draw mode, 40 for image mode), "allow rotation" checkbox, Run
button.

Result panel: SVG with y-axis flipped to match algorithm coordinates —
shape outlined, obstacles dark fill, complete cells green, partial cells
amber, plus a stats strip: complete count, partial count, coverage %, and
the placement's dx/dy/angle. Backend errors render as a dismissible banner.

Visual style: clean and minimal, line icons only (no emoji), consistent
with a premium tool aesthetic.

## Verification

- Backend: pytest tests for both endpoints via FastAPI's TestClient —
  polygon happy path (known rectangle → expected complete count), image
  happy path (synthetic PNG bytes), and 400s for bad polygon/blank image.
- Frontend: `npm run build` passes; manual check in the browser — draw a
  rectangle with an obstacle, run, see green/amber cells; upload
  `demo_plan.png`, run, see the L-shape packed.
