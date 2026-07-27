# Grid0pt

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)
![Shapely](https://img.shields.io/badge/Shapely-geometry-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-image%20boundary-5C3EE8?logo=opencv&logoColor=white)

Give it an outline and a cell size, and it finds the grid placement that packs the most whole cells into that shape. Draw the outline, upload a floor plan, or hand-draw a sketch and photograph it. It tells you exactly where every cell lands.

The outline doesn't have to be a rectangle. L-shapes, irregular rooms, anything you can trace as a polygon works, and you can carve out obstacles (pillars, stairwells, existing fixtures) that the grid has to avoid. Sliding the grid by a fraction of a cell in any direction changes how many cells fall cleanly inside, so it sweeps every offset (and, optionally, every rotation) and keeps the placement with the most complete cells and the least waste.

## What it does

- **Packs a polygon with a regular grid** of `cell_width x cell_height` cells, maximizing complete (fully inside) cells and minimizing partial (clipped) ones.
- **Avoids obstacles.** Interior holes are subtracted from the shape before packing, so cells never land on blocked areas.
- **Sweeps offsets and rotation.** Every fractional-cell shift is tried; with rotation enabled it also sweeps angles across the grid's true period (90 deg for square cells, 180 deg for rectangular ones), since a naive 0-90 deg sweep misses genuinely better packings for non-square cells.
- **Reads shapes from images.** Upload a clean floor plan or a hand-drawn sketch and `image_boundary.py` extracts the outer boundary and interior obstacles automatically: Otsu thresholding, morphological closing to seal gaps in drawn lines, and contour-hierarchy analysis to tell a filled shape from an unfilled outline.
- **Draw shapes by hand** in the browser, or type in polygon coordinates directly, via the Draw tab's canvas.
- **Visualizes the result** (shape, obstacles, complete cells, and partial cells) synced between an image view and a canvas view.

## Project layout

```
grid/
├── backend/
│   ├── grid_packer.py       # GridPacker: the offset/rotation sweep and cell classification
│   ├── image_boundary.py    # image -> (shape, obstacles) via OpenCV contour detection
│   ├── packer_service.py    # pure packing logic shared by the API and tests
│   ├── server.py            # FastAPI app: /api/pack/polygon, /api/pack/image, serves ../frontend/dist
│   ├── demo.py               # standalone script demo of GridPacker
│   ├── requirements.txt
│   └── tests/                # pytest suite for the packer, image boundary, and API
└── frontend/
    ├── src/
    │   ├── App.jsx               # tabs, controls, run/error state
    │   ├── api.js                 # fetch wrappers for the packing endpoints
    │   └── components/
    │       ├── ImageInput.jsx     # image upload tab
    │       ├── DrawCanvas.jsx     # freehand + coordinate polygon input
    │       └── ResultView.jsx     # renders the packed result
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
python -m pytest -v
```

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
  "stats": { "complete": 42, "partial": 7, "coverage": 0.91, "dx": 3.5, "dy": 1.2, "angle": 15 }
}
```

## Things I'd add next

- Cache/memoize repeated packing requests for the same shape + cell size.
- Let obstacles be drawn directly on top of an uploaded image instead of only detected from it.
- Export the packed layout (cell centers, count) as CSV/DXF for use in CAD tools.
