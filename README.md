# Grid0pt

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)
![Shapely](https://img.shields.io/badge/Shapely-geometry-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-image%20boundary-5C3EE8?logo=opencv&logoColor=white)

Give it an outline and a cell size, and it finds the grid placement that packs the most whole cells into that shape. Draw the outline, upload a floor plan, or hand-draw a sketch and photograph it. It tells you exactly where every cell lands.

The outline doesn't have to be a rectangle. L-shapes, irregular rooms, anything you can trace as a polygon works, and you can carve out obstacles (pillars, stairwells, existing fixtures) that the grid has to avoid.

## What it does

- **Packs a polygon with a regular grid** of `cell_width x cell_height` cells, maximizing complete (fully inside) cells and minimizing partial (clipped) ones.
- **Avoids obstacles.** Interior holes are subtracted from the shape before packing, so cells never land on blocked areas.
- **Solves the placement instead of scanning for it** — see below. There is no step size to tune and no sharp optimum to step over.
- **Says how good the answer is.** Every result carries a certificate bounding how far from optimal it could possibly be.
- **Reads shapes from images.** Upload a clean floor plan or a hand-drawn sketch and `image_boundary.py` extracts the outer boundary and interior obstacles automatically: Otsu thresholding, morphological closing to seal gaps in drawn lines, and contour-hierarchy analysis to tell a filled shape from an unfilled outline.
- **Draw shapes by hand** in the browser, or type in polygon coordinates directly, via the Draw tab's canvas.
- **Visualizes the result** (shape, obstacles, complete cells, and partial cells) synced between an image view and a canvas view.

## How the placement is found

Sliding the grid by a fraction of a cell changes how many cells fall cleanly inside, so the obvious approach is to try a lot of offsets and angles. That is what this used to do, and it has two problems: a finite sweep can step straight over the best placement, and nothing tells you how good the placement you got actually is.

Neither axis is searched by sampling any more.

**Translation is solved.** The complete-cell count is piecewise-constant in the offsets — it only changes when a grid line meets the boundary — so the horizontal offsets worth testing are a finite set read off the geometry. For each one, the vertical axis is not tested at all but solved: cut the region into the columns the grid makes, and within a column a cell is complete exactly when every horizontal slice of it spans the full column width. That makes each column a set of height intervals, and the count becomes the number of points of a periodic lattice landing in them — a one-dimensional problem with an exact answer. Cost per offset drops from "reclassify every cell, once per candidate" to a single interval sweep.

**Rotation is read off the shape.** Instead of scanning angles, the partial cells along the boundary vote: each obliquely-cut cell contributes its cut orientation, weighted by how much of the cell is inside and how long the cut is. The weighted circular mean is the orientation the walls want the grid to match, and the concentration of that vote decides whether rotating is worth doing at all — a room with a dominant wall gets turned onto it, a disc is left alone. Candidate angles are then solved exactly for translation, with the un-rotated placement kept in the running, so a bad vote costs a little time and never a worse answer.

**The result is certified.** Cells clipped by a feature smaller than a cell can never be made complete by any placement, and boundary that must cross a cell's interior forces a partial cell wherever the grid sits. Together these bound the fewest partial cells any placement could have, and the API reports the gap between that floor and what was achieved. A gap of zero means no placement of that grid on that region does better. When the bound's assumption doesn't hold for a given shape, the response says so rather than quoting a number that doesn't apply.

## Project layout

```
grid/
├── backend/
│   ├── grid_packer.py        # GridPacker: cell classification, both translation
│   │                         #   solvers, the rotation vote, the certificate
│   ├── image_boundary.py     # image -> (shape, obstacles) via OpenCV contour detection
│   ├── packer_service.py     # pure packing logic shared by the API and tests
│   ├── server.py             # FastAPI app: /api/pack/polygon, /api/pack/image,
│   │                         #   serves ../frontend/dist
│   ├── demo.py               # standalone script demo of GridPacker
│   ├── evaluation/           # corpus, baselines, ablations, metrics, reporting
│   ├── requirements.txt
│   └── tests/                # pytest suite: packer, image boundary, API, evaluation
└── frontend/
    ├── src/
    │   ├── App.jsx               # tabs, controls, run/error state
    │   ├── api.js                # fetch wrappers for the packing endpoints
    │   └── components/
    │       ├── ImageInput.jsx    # image upload tab
    │       ├── DrawCanvas.jsx    # freehand + coordinate polygon input
    │       └── ResultView.jsx    # renders the packed result
    └── vite.config.js
```

## Running it locally

You'll need Python 3.10+ and Node 18+.

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python server.py
```

Serves the API on `http://localhost:8000`. If `frontend/dist` exists, `server.py` also serves the built frontend from the same origin.

### 2. Frontend

In a second terminal, for hot-reload dev:

```bash
cd frontend
npm install
npm run dev
```

Or build it once and let the backend serve it:

```bash
cd frontend
npm install
npm run build
```

### Tests

```bash
cd backend
python -m pytest tests -q
```

Run them from `backend/` — the modules import each other flatly, so the repository root is the wrong working directory.

## API

**`POST /api/pack/polygon`**

```json
{
  "shape": [[0, 0], [100, 0], [100, 60], [0, 60]],
  "obstacles": [[[40, 20], [60, 20], [60, 40], [40, 40]]],
  "cell_width": 10,
  "cell_height": 10,
  "rotate": false
}
```

**`POST /api/pack/image`** takes a multipart form with an image `file`, plus `cell_width`, `cell_height`, and `rotate` fields. The outer boundary becomes the shape; enclosed interior regions become obstacles.

Both return the same shape:

```json
{
  "shape": [...],
  "obstacles": [...],
  "complete_cells": [...],
  "partial_cells": [...],
  "stats": {
    "complete": 42, "partial": 7, "coverage": 0.91,
    "dx": 3.5, "dy": 1.2, "angle": 15,

    "irreducible": 4, "partial_floor": 6, "optimality_gap": 1,
    "certified": true, "recoverable_area": 0.0,

    "resultant": 0.98, "rotated": true, "evaluations": 38
  }
}
```

The first block is the placement itself. The rest are diagnostics, and every one of them is additive — the geometry keys are unchanged, so an existing client can ignore them entirely.

| field | meaning |
| --- | --- |
| `irreducible` | partial cells no placement can rescue (features smaller than a cell) |
| `partial_floor` | fewest partial cells any placement of this grid on this region could have |
| `optimality_gap` | achieved partials minus that floor; `0` certifies the result optimal |
| `certified` | `false` when the floor's assumption doesn't hold here, so don't quote the gap |
| `recoverable_area` | area still outside the region across partials worth reclaiming |
| `resultant` | how concentrated the boundary orientations were, in `[0, 1]` |
| `rotated` | whether that was confident enough to turn the grid |
| `evaluations` | placements evaluated, the cost measure |

`resultant`, `rotated` and `evaluations` appear only when rotation was requested.

## Evaluation

`backend/evaluation/` measures the solver against the brute-force sweep it replaced, over a corpus generated from code rather than shipped as data — so running the generator reproduces the exact instances. Some families carry a *proven* optimum: a room whose sides are multiples of the cell tiles perfectly, and rotating that room rigidly can't change what a grid can do to it, so the tilted version keeps the same optimum and only a method that finds the rotation reaches it.

```bash
cd backend
python -m evaluation.run                          # quick corpus, minutes
python -m evaluation.run --full                   # every tilt, cell geometry and seed
python -m evaluation.run --with-reference         # add the brute-force yardstick
python -m evaluation.run --report results/*.json  # combine chunked runs
```

Results are written to `backend/evaluation/results/` as CSV and JSON, and are not tracked.

## Things I'd add next

- Solve the horizontal axis the way the vertical one is solved; it is still enumerated over a finite set of candidate offsets.
- Cache/memoize repeated packing requests for the same shape + cell size.
- Let obstacles be drawn directly on top of an uploaded image instead of only detected from it.
- Export the packed layout (cell centers, count) as CSV/DXF for use in CAD tools.
