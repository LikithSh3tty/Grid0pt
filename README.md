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
- **Says how good the answer is.** Every result carries a certificate bounding how far from optimal it could possibly be — and on request, a proof that no placement at *any* angle fits more cells, or leaves fewer clipped ones.
- **Reads shapes from images.** Upload a clean floor plan or a hand-drawn sketch and `image_boundary.py` extracts the outer boundary and interior obstacles automatically: Otsu thresholding, morphological closing to seal gaps in drawn lines, and contour-hierarchy analysis to tell a filled shape from an unfilled outline.
- **Draw shapes by hand** in the browser, or type in polygon coordinates directly, via the Draw tab's canvas.
- **Mark obstacles on an uploaded plan.** Tracing finds what the drawing shows as a hole; it cannot find a pillar the survey missed or an area someone has decided to keep clear. Click their corners on the image and the grid avoids them too.
- **Export the layout** as CSV (cell centres and corners) or DXF (a layered drawing), so it goes straight to a spreadsheet, a placement script, or CAD.
- **Visualizes the result** (shape, obstacles, complete cells, and partial cells) synced between an image view and a canvas view.

## How the placement is found

Sliding the grid by a fraction of a cell changes how many cells fall cleanly inside, so the obvious approach is to try a lot of offsets and angles. That is what this used to do, and it has two problems: a finite sweep can step straight over the best placement, and nothing tells you how good the placement you got actually is.

Neither axis is searched by sampling any more.

**Translation is solved, on both axes, for any shape.** Stop asking where the grid should go and ask where a single cell may sit. A cell is complete exactly when its corner lies in the region *eroded by the cell* — the set of points from which a whole cell still fits — and that set is computed once, exactly, with polygon operations. Since the grid's corners are a lattice, the complete-cell count is just "how many points of this lattice land in that fixed set". Fold the set modulo the lattice and every offset's count becomes the number of folded pieces stacked over a single point, so the best placement is the deepest overlap, and the deepest overlap sits at a corner of the pieces' outlines. Finitely many candidates, nothing sampled on either axis.

That last part is the whole claim. The overlap depth *is* the complete-cell count as a function of offset, over every offset at once, so its maximum bounds what any placement could achieve — reaching it proves the answer optimal rather than merely better than the sweep it beat. It also makes the search cost one evaluation per angle no matter how complicated the boundary is.

Earlier versions solved one axis at a time and are kept as ablations: enumerating the offsets where the count can change (exact for a rectilinear boundary) and then solving the vertical axis as a lattice-against-intervals problem while still enumerating the horizontal one. Both are exact only where the walls are square to the grid — a slanted wall flips a cell where a grid corner grazes it, at an offset no vertex can name — which on a plain trapezoid costs a cell: 59 found where 60 exist.

**Rotation is read off the shape.** Instead of scanning angles, the partial cells along the boundary vote: each obliquely-cut cell contributes its cut orientation, weighted by how much of the cell is inside and how long the cut is. The weighted circular mean is the orientation the walls want the grid to match, and the concentration of that vote decides whether rotating is worth doing at all — a room with a dominant wall gets turned onto it, a disc is left alone. Candidate angles are then solved exactly for translation, with the un-rotated placement kept in the running, so a bad vote costs a little time and never a worse answer.

Two details of that were settled by measurement rather than argument, both using the certificate below as the arbiter. The weight is `g(f) = f²`, which reaches the proven optimum on 67 of the 72 corpus instances against the linear weight's 65 and is worse on none. And for rectangular cells the quarter turn is solved unconditionally, because a rectilinear outline gives the vote no oblique chords to reason from at all — R comes out 0 and the grid would stay put, even though swapping a 2×3 cell to 3×2 is a different tiling and often the better one. Between them the pipeline went from missing the proven optimum on 7 of 72 instances to missing on 1.

**The angle can be proven, not just voted for.** The vote is evidence, not proof — nothing in it says a different angle wouldn't do better. Ask for `certify` and the API answers that too, by branch and bound over the angle. Turning by θ moves a point at radius r by exactly `r·θ`, so every angle within a window of some θ₀ sees a region contained in that one grown by `radius × half-window`; erosion is monotone, so the overlap depth of the grown region is an upper bound on the complete count for *every* angle in the window. A window whose bound can't beat the best placement found is discarded whole — without ever locating the angles where the count jumps, which is what makes this tractable when enumerating those jumps is not.

The vote changes job rather than being replaced: it supplies the starting placement, and the search supplies the proof that nothing beats it. Usually the vote was already right and the search only confirms it — 15 windows to prove that 108 cells is the most a 3×3 grid can ever fit in a 36×27 room tilted 23°. A curved boundary is the expensive case, since the count barely varies with angle and nothing prunes on quality. It costs tens of seconds against under one, so it is opt-in; a search that runs out of budget reports the gap it could not close rather than claiming optimality.

**The result is certified.** Cells clipped by a feature smaller than a cell can never be made complete by any placement, and boundary that must cross a cell's interior forces a partial cell wherever the grid sits. Together these bound the fewest partial cells any placement could have, and the API reports the gap between that floor and what was achieved. A gap of zero means no placement of that grid on that region does better. When the bound's assumption doesn't hold for a given shape, the response says so rather than quoting a number that doesn't apply.

## Project layout

```
grid/
├── backend/
│   ├── grid_packer.py        # GridPacker: cell classification, the translation
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

cd ../frontend
npm test
```

Run the Python ones from `backend/` — the modules import each other flatly, so the repository root is the wrong working directory.

The frontend tests cover what the components decide — which request goes out, which optimality verdict is shown — not how they paint. They were written after the components rather than before, so each was checked by breaking the code it covers and confirming it failed; a test that has never failed has not been shown to test anything.

## API

**`POST /api/pack/polygon`**

```json
{
  "shape": [[0, 0], [100, 0], [100, 60], [0, 60]],
  "obstacles": [[[40, 20], [60, 20], [60, 40], [40, 40]]],
  "cell_width": 10,
  "cell_height": 10,
  "rotate": false,
  "certify": false
}
```

`certify` proves the angle instead of voting for it, and only does anything alongside `rotate`. It costs tens of seconds against well under one, so leave it off unless you want the theorem.

**`POST /api/export/polygon`** takes the same body plus `"format": "csv"` or `"dxf"`, and returns the layout as a file rather than as JSON.

CSV is a row per cell — kind, centre, four corners — for a spreadsheet or a placement script. DXF is a layered drawing for a CAD package, with complete cells, partial cells, the outline and the obstacles on separate layers so any of them can be switched off. Partial cells are kept distinct in both: a partial cell is not somewhere anything can be installed, and merging the two would hand someone a layout that does not fit.

Repeated requests are served from a small cache keyed on the outline, cell size and flags, so exporting something already on screen costs the serialisation only.

**`POST /api/pack/image`** takes a multipart form with an image `file`, plus `cell_width`, `cell_height`, `rotate` and `certify` fields. The outer boundary becomes the shape; enclosed interior regions become obstacles.

An optional `obstacles` field carries areas marked by hand, as a JSON list of rings, added to whatever tracing found. They are given in the same coordinate space the response's `shape` comes back in — image pixels, y up — because the tracer flips the y axis, and a ring passed in image-row order would land mirrored: wrong, and plausibly enough to be believed.

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

    "resultant": 0.98, "rotated": true, "evaluations": 38,

    "rotation_bound": 42, "rotation_gap": 0, "rotation_optimal": true,
    "rotation_exhausted": true, "rotation_nodes": 15
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
| `rotation_bound` | the most complete cells any placement could hold, **at any angle** |
| `rotation_gap` | that bound minus what was achieved; `0` proves the result globally optimal |
| `rotation_optimal` | `true` only when the gap is 0 *and* the search closed the whole angle range |
| `rotation_exhausted` | `false` when the search hit its node budget — the bound still holds, it just isn't closed |
| `rotation_nodes` | angular windows examined, the cost measure for the proof |
| `angle_partial_floor` | fewest partial cells any offset could leave **at this angle** — computed, not argued, so it never declines |
| `angle_partial_gap` | partials above that floor; `0` means no offset at this angle does better |

`resultant`, `rotated` and `evaluations` appear only when rotation was requested. The `rotation_*` block appears only when `certify` was too.

The certificates are independent statements and are deliberately never merged. `optimality_gap` bounds the *partial* count from below with a covering argument that holds at every angle but doesn't always apply, and says so via `certified`. `angle_partial_gap` bounds the same count with an argument that always applies but only at the angle in the result. `rotation_gap` bounds the *complete* count from above, for any shape, at any angle. Combining any two of them would take the weaker scope from one and the weaker guarantee from the other.

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

- **Tighten the computed floor on curved boundaries.** Re-measured against attainable counts rather than a coarse sweep, it is exactly tight on five of seven fixed-angle cases; the exceptions are both discs, one cell under what any offset at that angle achieves. Over all angles even those close, because the value is attained at a different angle — so the remaining looseness costs nothing to the all-angles claim and only shows up if you ask about one angle in particular.
- **The one instance that still misses.** `traced-l23-2x3` comes back one cell under its proven 83. The certificate names the placement the vote should have found, so what is missing is an explanation of why the vote does not name that angle — not a way to discover it.
- **A cheaper certificate for an asymmetric shape with no dominant wall.** Symmetry handles the common case — a polygonised disc repeats every 7.5°, so the search does too. A shape that is flat in the angle *without* being symmetric would still split to the tolerance, and nothing here prunes it. No instance in the corpus does that, which is why it is a note rather than a measurement.
