# Grid Packer Web Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web UI (FastAPI + React) to run the grid packer via image upload or manual polygon entry (click-to-draw with a synced text panel), with cell size and rotation controls.

**Architecture:** `packer_service.py` wraps `GridPacker`/`image_boundary` into two pure, HTTP-independent functions; `server.py` is a thin FastAPI layer over them that also serves the built React app. The React app (`frontend/`, Vite) has shared controls (cell width/height, rotation), an Image tab, a Draw tab (SVG canvas + text panel, two-way synced), and a shared SVG result view.

**Tech Stack:** Python: FastAPI, uvicorn, python-multipart, shapely, opencv-python, numpy (all already used or newly added). Frontend: React 18, Vite, plain CSS (no UI framework).

## How to run (once all tasks are done)

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python server.py
# open http://localhost:8000
```

For frontend iteration: `cd frontend && npm run dev` (proxies `/api` to `:8000`) alongside `python server.py` in another terminal.

## Global Constraints

- Endpoints and JSON contract exactly as specified: `POST /api/pack/image` (multipart: `file`, `cell_width`, `cell_height`, `rotate`), `POST /api/pack/polygon` (JSON: `shape`, `obstacles`, `cell_width`, `cell_height`, `rotate`); both return `{shape, obstacles, complete_cells, partial_cells, stats: {complete, partial, coverage, dx, dy, angle}}`.
- Search parameters fixed server-side: `steps=10`; angles `(0.0,)` when `rotate` is false, `range(0, 90, 15)` when true.
- Cells/shape/obstacles returned as polygon coordinate lists (`[[x,y], ...]`), not bounding boxes.
- Validation errors → HTTP 400 with `{"detail": "<message>"}`.
- New backend dependencies: `fastapi`, `uvicorn`, `python-multipart` only.
- New frontend dependencies: `react`, `react-dom`, `vite`, `@vitejs/plugin-react` only — no additional UI framework or component library.
- No emoji anywhere in UI copy or code (user preference: line icons/text only, not emoji — emoji reads as low-effort).
- Draw mode: first closed polygon is the shape; every subsequent closed polygon is an obstacle. Text panel: one line per polygon, points as `x,y x,y x,y ...`, shape first line.
- Commit messages: plain, no Co-Authored-By trailers.

---

### Task 1: Backend packing service

**Files:**
- Create: `packer_service.py`
- Test: `tests/test_packer_service.py`

**Interfaces:**
- Consumes: `GridPacker(shape, obstacles=None, cell_width=1.0, cell_height=1.0)`, `GridPacker.optimize(steps, angles) -> (Placement, list[Placement])`, `GridPacker.from_image(image, *, cell_width, cell_height) -> GridPacker` (all existing, in `grid_packer.py`).
- Produces: `run_packing(shape_points: list[tuple[float,float]], obstacle_points: list[list[tuple[float,float]]], cell_width: float, cell_height: float, rotate: bool) -> dict` and `run_packing_from_image(image_bytes: bytes, cell_width: float, cell_height: float, rotate: bool) -> dict`. Both raise `ValueError` with a human-readable message on bad input. Task 2 imports both names from `packer_service`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_packer_service.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_packer_service.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'packer_service'`.

- [ ] **Step 3: Write the implementation**

Create `packer_service.py`:

```python
"""
packer_service.py
==================

Pure packing logic shared by the HTTP layer (server.py). Wraps GridPacker
and image_boundary so packing can be tested directly, without going through
FastAPI or HTTP.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon

from grid_packer import GridPacker

Point = Tuple[float, float]

DEFAULT_STEPS = 10
ROTATE_ANGLES = tuple(range(0, 90, 15))


def _polygon_coords(poly: Polygon) -> List[Point]:
    return [(float(x), float(y)) for x, y in poly.exterior.coords[:-1]]


def _placement_to_result(packer: GridPacker, best) -> dict:
    return {
        "shape": _polygon_coords(packer.shape),
        "obstacles": [_polygon_coords(o) for o in packer.obstacles],
        "complete_cells": [_polygon_coords(c) for c in best.complete_cells],
        "partial_cells": [_polygon_coords(c) for c in best.partial_cells],
        "stats": {
            "complete": best.complete,
            "partial": best.partial,
            "coverage": best.coverage,
            "dx": best.dx,
            "dy": best.dy,
            "angle": best.angle,
        },
    }


def _to_polygon(points: Sequence[Point], label: str) -> Polygon:
    if len(points) < 3:
        raise ValueError(f"{label} must have at least 3 points")
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        raise ValueError(f"{label} polygon is invalid or self-intersecting")
    return poly


def run_packing(
    shape_points: Sequence[Point],
    obstacle_points: Sequence[Sequence[Point]],
    cell_width: float,
    cell_height: float,
    rotate: bool,
) -> dict:
    """Pack a manually specified polygon. Raises ValueError on bad input."""
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("cell dimensions must be positive")

    shape = _to_polygon(shape_points, "shape")
    obstacles = [_to_polygon(pts, "obstacle") for pts in obstacle_points]

    packer = GridPacker(shape, obstacles, cell_width=cell_width, cell_height=cell_height)
    angles = ROTATE_ANGLES if rotate else (0.0,)
    best, _ = packer.optimize(steps=DEFAULT_STEPS, angles=angles)
    return _placement_to_result(packer, best)


def run_packing_from_image(
    image_bytes: bytes,
    cell_width: float,
    cell_height: float,
    rotate: bool,
) -> dict:
    """Pack the boundary detected in raw image bytes. Raises ValueError on bad input."""
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("cell dimensions must be positive")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not read image: unsupported or corrupt file")

    packer = GridPacker.from_image(img, cell_width=cell_width, cell_height=cell_height)
    angles = ROTATE_ANGLES if rotate else (0.0,)
    best, _ = packer.optimize(steps=DEFAULT_STEPS, angles=angles)
    return _placement_to_result(packer, best)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_packer_service.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add packer_service.py tests/test_packer_service.py
git commit -m "Add backend packing service"
```

---

### Task 2: FastAPI server

**Files:**
- Create: `server.py`
- Modify: `requirements.txt` (add fastapi, uvicorn, python-multipart)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `run_packing(shape_points, obstacle_points, cell_width, cell_height, rotate) -> dict` and `run_packing_from_image(image_bytes, cell_width, cell_height, rotate) -> dict` from `packer_service` (Task 1); both raise `ValueError` on bad input.
- Produces: HTTP app `server.app` (FastAPI instance) with routes `POST /api/pack/polygon` and `POST /api/pack/image`, and static-file serving of `frontend/dist` at `/` when that directory exists. Task 3/4 (frontend) call these routes; this task's tests use `fastapi.testclient.TestClient(app)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server.py`:

```python
"""Tests for server.py's HTTP endpoints."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def make_plan_image_bytes():
    img = np.zeros((200, 300), np.uint8)
    cv2.rectangle(img, (30, 20), (269, 179), 255, thickness=-1)
    cv2.rectangle(img, (100, 80), (139, 109), 0, thickness=-1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_pack_polygon_happy_path():
    resp = client.post("/api/pack/polygon", json={
        "shape": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "obstacles": [],
        "cell_width": 2,
        "cell_height": 2,
        "rotate": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["stats"]["complete"] == 25
    assert data["stats"]["partial"] == 0


def test_pack_polygon_invalid_shape_returns_400():
    resp = client.post("/api/pack/polygon", json={
        "shape": [[0, 0], [10, 0]],
        "obstacles": [],
        "cell_width": 2,
        "cell_height": 2,
        "rotate": False,
    })
    assert resp.status_code == 400
    assert "at least 3 points" in resp.json()["detail"]


def test_pack_image_happy_path():
    resp = client.post(
        "/api/pack/image",
        files={"file": ("plan.png", make_plan_image_bytes(), "image/png")},
        data={"cell_width": "20", "cell_height": "20", "rotate": "false"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["obstacles"]) == 1


def test_pack_image_bad_bytes_returns_400():
    resp = client.post(
        "/api/pack/image",
        files={"file": ("bad.png", b"not an image", "image/png")},
        data={"cell_width": "20", "cell_height": "20", "rotate": "false"},
    )
    assert resp.status_code == 400


def test_pack_image_blank_returns_400():
    blank = np.zeros((50, 50), np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    resp = client.post(
        "/api/pack/image",
        files={"file": ("blank.png", buf.tobytes(), "image/png")},
        data={"cell_width": "10", "cell_height": "10", "rotate": "false"},
    )
    assert resp.status_code == 400
    assert "no boundary detected" in resp.json()["detail"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pip install fastapi uvicorn python-multipart` then `python -m pytest tests/test_server.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'server'`.

- [ ] **Step 3: Write the implementation**

Create `server.py`:

```python
"""
server.py
=========

FastAPI app exposing the grid packer over HTTP, and serving the built React
frontend (frontend/dist) as static files.

Run: python server.py
Then open http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from packer_service import run_packing, run_packing_from_image

app = FastAPI(title="Grid Packer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PolygonRequest(BaseModel):
    shape: List[Tuple[float, float]]
    obstacles: List[List[Tuple[float, float]]] = []
    cell_width: float
    cell_height: float
    rotate: bool = False


@app.post("/api/pack/polygon")
def pack_polygon(req: PolygonRequest):
    try:
        return run_packing(
            req.shape, req.obstacles, req.cell_width, req.cell_height, req.rotate,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pack/image")
async def pack_image(
    file: UploadFile = File(...),
    cell_width: float = Form(...),
    cell_height: float = Form(...),
    rotate: bool = Form(False),
):
    image_bytes = await file.read()
    try:
        return run_packing_from_image(image_bytes, cell_width, cell_height, rotate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Modify `requirements.txt` — replace its full contents with:

```
numpy
shapely
matplotlib
opencv-python
fastapi
uvicorn
python-multipart
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_server.py -v`
Expected: all 5 tests PASS. Then run `python -m pytest tests/ -v` to confirm nothing else broke; expected: all 27 tests PASS (15 pre-existing + 7 from Task 1 + 5 here). If your actual total differs, note the drift and the exact numbers in your report.

- [ ] **Step 5: Commit**

```bash
git add server.py requirements.txt tests/test_server.py
git commit -m "Add FastAPI server with pack endpoints"
```

---

### Task 3: Frontend scaffold, Image tab, and result view

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.jsx`
- Create: `frontend/src/App.jsx`
- Create: `frontend/src/App.css`
- Create: `frontend/src/api.js`
- Create: `frontend/src/components/ImageInput.jsx`
- Create: `frontend/src/components/ResultView.jsx`

**Interfaces:**
- Consumes: `POST /api/pack/image` (multipart: `file`, `cell_width`, `cell_height`, `rotate`) from Task 2, returning `{shape, obstacles, complete_cells, partial_cells, stats: {complete, partial, coverage, dx, dy, angle}}`.
- Produces: `packImage(file, cellWidth, cellHeight, rotate) -> Promise<result>` and `packPolygon(shape, obstacles, cellWidth, cellHeight, rotate) -> Promise<result>` exported from `frontend/src/api.js` (Task 4 imports `packPolygon`). `ResultView` component exported from `frontend/src/components/ResultView.jsx`, accepting prop `result` shaped exactly like the API response or `null` (Task 4 reuses it unchanged).

- [ ] **Step 1: Scaffold the Vite project files**

Create `frontend/package.json`:

```json
{
  "name": "grid-packer-frontend",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.1",
    "vite": "^5.4.0"
  }
}
```

Create `frontend/vite.config.js`:

```javascript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

Create `frontend/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Grid Packer</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

Create `frontend/src/main.jsx`:

```jsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

- [ ] **Step 2: Add the API client**

Create `frontend/src/api.js`:

```javascript
const API_BASE = import.meta.env.VITE_API_BASE || "";

export async function packPolygon(shape, obstacles, cellWidth, cellHeight, rotate) {
  const res = await fetch(`${API_BASE}/api/pack/polygon`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      shape,
      obstacles,
      cell_width: cellWidth,
      cell_height: cellHeight,
      rotate,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Packing failed");
  }
  return data;
}

export async function packImage(file, cellWidth, cellHeight, rotate) {
  const form = new FormData();
  form.append("file", file);
  form.append("cell_width", String(cellWidth));
  form.append("cell_height", String(cellHeight));
  form.append("rotate", String(rotate));

  const res = await fetch(`${API_BASE}/api/pack/image`, {
    method: "POST",
    body: form,
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Packing failed");
  }
  return data;
}
```

- [ ] **Step 3: Add the Image input and Result view components**

Create `frontend/src/components/ImageInput.jsx`:

```jsx
import { useState } from "react";

export default function ImageInput({ onFileSelected }) {
  const [preview, setPreview] = useState(null);

  function handleFile(file) {
    if (!file) return;
    onFileSelected(file);
    setPreview(URL.createObjectURL(file));
  }

  return (
    <div className="image-input">
      <label className="dropzone">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {preview ? (
          <img src={preview} alt="Selected upload preview" className="image-preview" />
        ) : (
          <span>Click or drop an image</span>
        )}
      </label>
    </div>
  );
}
```

Create `frontend/src/components/ResultView.jsx`:

```jsx
export default function ResultView({ result }) {
  if (!result) return null;

  const { shape, obstacles, complete_cells, partial_cells, stats } = result;

  const allPoints = [...shape, ...obstacles.flat(), ...complete_cells.flat(), ...partial_cells.flat()];
  const xs = allPoints.map((p) => p[0]);
  const ys = allPoints.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = maxX - minX;
  const height = maxY - minY;
  const pad = Math.max(width, height) * 0.05 || 1;

  const flipY = (y) => minY + maxY - y;
  const toPoints = (pts) => pts.map(([x, y]) => `${x},${flipY(y)}`).join(" ");

  return (
    <div className="result-view">
      <svg
        viewBox={`${minX - pad} ${minY - pad} ${width + pad * 2} ${height + pad * 2}`}
        className="result-svg"
      >
        <polygon points={toPoints(shape)} className="shape-outline" />
        {obstacles.map((ob, i) => (
          <polygon key={`ob-${i}`} points={toPoints(ob)} className="obstacle" />
        ))}
        {partial_cells.map((c, i) => (
          <polygon key={`pc-${i}`} points={toPoints(c)} className="partial-cell" />
        ))}
        {complete_cells.map((c, i) => (
          <polygon key={`cc-${i}`} points={toPoints(c)} className="complete-cell" />
        ))}
      </svg>
      <div className="stats-strip">
        <span>Complete: {stats.complete}</span>
        <span>Partial: {stats.partial}</span>
        <span>Coverage: {(stats.coverage * 100).toFixed(1)}%</span>
        <span>Offset: ({stats.dx.toFixed(2)}, {stats.dy.toFixed(2)})</span>
        <span>Angle: {stats.angle.toFixed(0)}°</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Add the App shell (Image mode only) and styling**

Create `frontend/src/App.jsx`:

```jsx
import { useState } from "react";
import ImageInput from "./components/ImageInput";
import ResultView from "./components/ResultView";
import { packImage } from "./api";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [cellWidth, setCellWidth] = useState(40);
  const [cellHeight, setCellHeight] = useState(40);
  const [rotate, setRotate] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleRun() {
    setError(null);
    if (!file) {
      setError("Choose an image first.");
      return;
    }
    setLoading(true);
    try {
      const data = await packImage(file, Number(cellWidth), Number(cellHeight), rotate);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <h1>Grid Packer</h1>

      <div className="controls">
        <label>
          Cell width
          <input type="number" min="0.01" step="any" value={cellWidth}
                 onChange={(e) => setCellWidth(e.target.value)} />
        </label>
        <label>
          Cell height
          <input type="number" min="0.01" step="any" value={cellHeight}
                 onChange={(e) => setCellHeight(e.target.value)} />
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={rotate}
                 onChange={(e) => setRotate(e.target.checked)} />
          Allow rotation
        </label>
        <button onClick={handleRun} disabled={loading}>
          {loading ? "Running..." : "Run"}
        </button>
      </div>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button type="button" className="error-dismiss" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      <ImageInput onFileSelected={setFile} />

      <ResultView result={result} />
    </div>
  );
}
```

Create `frontend/src/App.css`:

```css
:root {
  --bg: #0f1115;
  --panel: #171a21;
  --border: #2a2e37;
  --text: #e6e8eb;
  --muted: #9aa1ac;
  --accent: #5cb85c;
  --accent-2: #f4a259;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
}

.app {
  max-width: 960px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
}

h1 {
  font-size: 1.4rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin-bottom: 1.5rem;
}

.controls {
  display: flex;
  align-items: end;
  gap: 1.25rem;
  flex-wrap: wrap;
  padding: 1rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--panel);
  margin-bottom: 1.5rem;
}

.controls label {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  font-size: 0.8rem;
  color: var(--muted);
}

.controls input[type="number"] {
  width: 6rem;
  padding: 0.4rem 0.5rem;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
}

.checkbox-label {
  flex-direction: row !important;
  align-items: center;
  gap: 0.5rem !important;
}

button {
  padding: 0.5rem 1.1rem;
  border-radius: 6px;
  border: none;
  background: var(--accent);
  color: #0f1115;
  font-weight: 600;
  cursor: pointer;
}

button:disabled {
  opacity: 0.6;
  cursor: default;
}

.error-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.75rem 1rem;
  border: 1px solid #b3413a;
  background: #2a1616;
  color: #ff9d94;
  border-radius: 8px;
  margin-bottom: 1rem;
}

.error-dismiss {
  background: transparent;
  color: #ff9d94;
  border: 1px solid #b3413a;
  padding: 0.2rem 0.6rem;
  font-size: 0.75rem;
  font-weight: 500;
  flex-shrink: 0;
}

.image-input .dropzone {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 10rem;
  border: 1px dashed var(--border);
  border-radius: 10px;
  color: var(--muted);
  cursor: pointer;
  margin-bottom: 1.5rem;
  overflow: hidden;
}

.image-input input[type="file"] {
  display: none;
}

.image-preview {
  max-width: 100%;
  max-height: 20rem;
  object-fit: contain;
}

.result-svg {
  width: 100%;
  max-height: 32rem;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
}

.shape-outline {
  fill: none;
  stroke: var(--text);
  stroke-width: 0.4%;
}

.obstacle {
  fill: #444;
  stroke: #000;
  opacity: 0.85;
}

.complete-cell {
  fill: var(--accent);
  opacity: 0.75;
  stroke: #2f6f2f;
  stroke-width: 0.15%;
}

.partial-cell {
  fill: var(--accent-2);
  opacity: 0.55;
  stroke: #b56a1e;
  stroke-width: 0.15%;
}

.stats-strip {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  margin-top: 1rem;
  font-size: 0.85rem;
  color: var(--muted);
}
```

- [ ] **Step 5: Install and build**

Run: `cd frontend && npm install && npm run build`
Expected: install completes without error; build produces `frontend/dist/` with no errors.

- [ ] **Step 6: Commit**

```bash
cd ..
git add frontend/
git commit -m "Add frontend scaffold, Image tab, and result view"
```

---

### Task 4: Draw tab (canvas + synced text panel)

**Files:**
- Create: `frontend/src/components/DrawCanvas.jsx`
- Modify: `frontend/src/App.jsx` (add tab switcher and Draw mode)
- Modify: `frontend/src/App.css` (append Draw-tab styles)

**Interfaces:**
- Consumes: `packPolygon(shape, obstacles, cellWidth, cellHeight, rotate)` from `frontend/src/api.js` (Task 3); `ResultView` from `frontend/src/components/ResultView.jsx` (Task 3, unchanged).
- Produces: `DrawCanvas` component exported from `frontend/src/components/DrawCanvas.jsx`, props `polygons: Array<Array<[number, number]>>` and `onPolygonsChange: (polygons) => void`. No later task consumes this directly (final task).

- [ ] **Step 1: Add the DrawCanvas component**

Create `frontend/src/components/DrawCanvas.jsx`:

```jsx
import { useRef, useState } from "react";

const VIEW_SIZE = 100;

function pointFromEvent(svgEl, evt) {
  const pt = svgEl.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const loc = pt.matrixTransform(svgEl.getScreenCTM().inverse());
  return [Math.round(loc.x * 10) / 10, Math.round(loc.y * 10) / 10];
}

function polygonsToText(polygons) {
  return polygons.map((pts) => pts.map(([x, y]) => `${x},${y}`).join(" ")).join("\n");
}

function textToPolygons(text) {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const polygons = [];
  for (const line of lines) {
    const pts = line.split(/\s+/).map((tok) => {
      const [xs, ys] = tok.split(",");
      const x = parseFloat(xs);
      const y = parseFloat(ys);
      if (Number.isNaN(x) || Number.isNaN(y)) {
        throw new Error(`invalid point "${tok}"`);
      }
      return [x, y];
    });
    if (pts.length < 3) {
      throw new Error("each polygon needs at least 3 points");
    }
    polygons.push(pts);
  }
  return polygons;
}

export default function DrawCanvas({ polygons, onPolygonsChange }) {
  const svgRef = useRef(null);
  const [current, setCurrent] = useState([]);
  const [text, setText] = useState(polygonsToText(polygons));
  const [textError, setTextError] = useState(null);

  function syncText(next) {
    setText(polygonsToText(next));
    setTextError(null);
  }

  function handleCanvasClick(evt) {
    const [x, y] = pointFromEvent(svgRef.current, evt);
    setCurrent((prev) => [...prev, [x, y]]);
  }

  function closePolygon() {
    if (current.length < 3) return;
    const next = [...polygons, current];
    onPolygonsChange(next);
    syncText(next);
    setCurrent([]);
  }

  function undoPoint() {
    setCurrent((prev) => prev.slice(0, -1));
  }

  function clearAll() {
    setCurrent([]);
    onPolygonsChange([]);
    syncText([]);
  }

  function handleTextChange(evt) {
    const value = evt.target.value;
    setText(value);
    try {
      const parsed = textToPolygons(value);
      setTextError(null);
      onPolygonsChange(parsed);
    } catch (err) {
      setTextError(err.message);
    }
  }

  const allDrawn = [...polygons, current].filter((p) => p.length > 0);
  const xs = allDrawn.flat().map((p) => p[0]);
  const ys = allDrawn.flat().map((p) => p[1]);
  const minX = Math.min(0, ...xs);
  const minY = Math.min(0, ...ys);
  const maxX = Math.max(VIEW_SIZE, ...xs);
  const maxY = Math.max(VIEW_SIZE, ...ys);

  return (
    <div className="draw-panel">
      <svg
        ref={svgRef}
        viewBox={`${minX} ${minY} ${maxX - minX} ${maxY - minY}`}
        className="draw-svg"
        onClick={handleCanvasClick}
      >
        {polygons.map((pts, i) => (
          <polygon
            key={i}
            points={pts.map((p) => p.join(",")).join(" ")}
            className={i === 0 ? "draw-shape" : "draw-obstacle"}
          />
        ))}
        {current.length > 0 && (
          <polyline
            points={current.map((p) => p.join(",")).join(" ")}
            className="draw-current"
          />
        )}
        {current.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={1} className="draw-point" />
        ))}
      </svg>

      <div className="draw-toolbar">
        <button type="button" onClick={closePolygon} disabled={current.length < 3}>
          Close polygon
        </button>
        <button type="button" onClick={undoPoint} disabled={current.length === 0}>
          Undo point
        </button>
        <button type="button" onClick={clearAll}>Clear</button>
        <span className="draw-hint">
          {polygons.length === 0
            ? "Draw the shape first"
            : `${polygons.length - 1} obstacle(s) drawn`}
        </span>
      </div>

      <textarea
        className="draw-text"
        value={text}
        onChange={handleTextChange}
        rows={4}
        placeholder={"shape: x,y x,y x,y ...\nobstacle: x,y x,y x,y ..."}
      />
      {textError && <div className="text-error">{textError}</div>}
    </div>
  );
}
```

- [ ] **Step 2: Wire the Draw tab into App**

Replace the full contents of `frontend/src/App.jsx` with:

```jsx
import { useState } from "react";
import ImageInput from "./components/ImageInput";
import DrawCanvas from "./components/DrawCanvas";
import ResultView from "./components/ResultView";
import { packImage, packPolygon } from "./api";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState("image");
  const [file, setFile] = useState(null);
  const [polygons, setPolygons] = useState([]);
  const [cellWidth, setCellWidth] = useState(40);
  const [cellHeight, setCellHeight] = useState(40);
  const [rotate, setRotate] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleRun() {
    setError(null);
    setLoading(true);
    try {
      let data;
      if (mode === "image") {
        if (!file) throw new Error("Choose an image first.");
        data = await packImage(file, Number(cellWidth), Number(cellHeight), rotate);
      } else {
        if (polygons.length === 0) throw new Error("Draw or enter a shape first.");
        const [shape, ...obstacles] = polygons;
        data = await packPolygon(shape, obstacles, Number(cellWidth), Number(cellHeight), rotate);
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <h1>Grid Packer</h1>

      <div className="tabs">
        <button
          type="button"
          className={mode === "image" ? "tab active" : "tab"}
          onClick={() => setMode("image")}
        >
          Image
        </button>
        <button
          type="button"
          className={mode === "draw" ? "tab active" : "tab"}
          onClick={() => setMode("draw")}
        >
          Draw
        </button>
      </div>

      <div className="controls">
        <label>
          Cell width
          <input type="number" min="0.01" step="any" value={cellWidth}
                 onChange={(e) => setCellWidth(e.target.value)} />
        </label>
        <label>
          Cell height
          <input type="number" min="0.01" step="any" value={cellHeight}
                 onChange={(e) => setCellHeight(e.target.value)} />
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={rotate}
                 onChange={(e) => setRotate(e.target.checked)} />
          Allow rotation
        </label>
        <button onClick={handleRun} disabled={loading}>
          {loading ? "Running..." : "Run"}
        </button>
      </div>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button type="button" className="error-dismiss" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {mode === "image" ? (
        <ImageInput onFileSelected={setFile} />
      ) : (
        <DrawCanvas polygons={polygons} onPolygonsChange={setPolygons} />
      )}

      <ResultView result={result} />
    </div>
  );
}
```

- [ ] **Step 3: Append Draw-tab styles**

Append to the end of `frontend/src/App.css`:

```css
.tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.tab {
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.4rem 1rem;
  font-weight: 500;
}

.tab.active {
  background: var(--panel);
  color: var(--text);
  border-color: var(--accent);
}

.draw-panel {
  margin-bottom: 1.5rem;
}

.draw-svg {
  width: 100%;
  height: 22rem;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  cursor: crosshair;
}

.draw-shape {
  fill: rgba(92, 184, 92, 0.15);
  stroke: var(--text);
  stroke-width: 0.3%;
}

.draw-obstacle {
  fill: rgba(244, 162, 89, 0.25);
  stroke: var(--accent-2);
  stroke-width: 0.3%;
}

.draw-current {
  fill: none;
  stroke: var(--muted);
  stroke-width: 0.3%;
  stroke-dasharray: 1 1;
}

.draw-point {
  fill: var(--text);
}

.draw-toolbar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 0.75rem 0;
}

.draw-hint {
  color: var(--muted);
  font-size: 0.85rem;
}

.draw-text {
  width: 100%;
  box-sizing: border-box;
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.6rem 0.75rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.85rem;
  resize: vertical;
}

.text-error {
  color: #ff9d94;
  font-size: 0.8rem;
  margin-top: 0.25rem;
}
```

- [ ] **Step 4: Build and verify**

Run: `cd frontend && npm run build`
Expected: build succeeds with no errors (this catches import/syntax mistakes in the new component and the modified `App.jsx`).

- [ ] **Step 5: Commit**

```bash
cd ..
git add frontend/src/components/DrawCanvas.jsx frontend/src/App.jsx frontend/src/App.css
git commit -m "Add Draw tab with canvas and synced text panel"
```
