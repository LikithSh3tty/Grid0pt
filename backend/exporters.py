"""
exporters.py
============

Turn a packing result into something another tool can open: CSV for a
spreadsheet or a script, DXF for a drawing package.

Both take the dict `packer_service` returns, rather than a `GridPacker` or a
`Placement`, so what is exported is exactly what the API served. An exporter
reaching back into the solver could disagree with the response the caller is
looking at, which is the one thing a file handed to a builder must not do.

COMPLETE AND PARTIAL CELLS ARE KEPT APART in both formats. A partial cell is
not a cell anything can be installed in -- it is the boundary's problem showing
through -- so merging the two would hand someone a layout that does not fit.
CSV says so in a column, DXF in a layer, because a layer is what a CAD user can
switch off.
"""
from __future__ import annotations

import csv
import io
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]

#: Layer names in the DXF. Uppercase because that is the convention in the
#: format, and short because some older CAD packages truncate.
LAYER_COMPLETE = "GRID_COMPLETE"
LAYER_PARTIAL = "GRID_PARTIAL"
LAYER_SHAPE = "GRID_SHAPE"
LAYER_OBSTACLE = "GRID_OBSTACLE"

#: Colour indices, in the DXF's own palette: green for cells that fit, amber
#: for cells that do not, white for the outline, red for obstacles. The same
#: reading the app gives, so a drawing does not have to be re-learned.
_COLOURS = {
    LAYER_COMPLETE: 3,
    LAYER_PARTIAL: 30,
    LAYER_SHAPE: 7,
    LAYER_OBSTACLE: 1,
}


def _centre(cell: Sequence[Point]) -> Point:
    xs = [p[0] for p in cell]
    ys = [p[1] for p in cell]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def to_csv(result: dict) -> str:
    """One row per cell: its kind, its centre, and its four corners.

    The centre is what a placement script actually consumes, and the corners
    are what makes the row checkable against a drawing. Both are written rather
    than one derived from the other, so a reader never has to reconstruct the
    cell to know where it is.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["index", "kind", "cx", "cy",
                     "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"])

    index = 0
    for kind, cells in (("complete", result.get("complete_cells", [])),
                        ("partial", result.get("partial_cells", []))):
        for cell in cells:
            corners = list(cell)[:4]
            if len(corners) < 4:                # pragma: no cover - not emitted
                continue
            cx, cy = _centre(corners)
            writer.writerow([index, kind, f"{cx:.6f}", f"{cy:.6f}"]
                            + [f"{value:.6f}" for point in corners
                               for value in point])
            index += 1

    return buffer.getvalue()


def _dxf_polyline(points: Sequence[Point], layer: str) -> List[str]:
    """A closed LWPOLYLINE.

    LWPOLYLINE rather than POLYLINE/VERTEX/SEQEND: it is one entity per cell
    instead of three-plus, which on a fine grid is the difference between a
    file a CAD package opens promptly and one it labours over.
    """
    out = [
        "0", "LWPOLYLINE",
        "8", layer,
        "90", str(len(points)),
        "70", "1",                              # closed
    ]
    for x, y in points:
        out += ["10", f"{x:.6f}", "20", f"{y:.6f}"]
    return out


def _dxf_layer_table(layers: Iterable[str]) -> List[str]:
    out = ["0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LAYER"]
    for name in layers:
        out += [
            "0", "LAYER",
            "2", name,
            "70", "0",
            "62", str(_COLOURS.get(name, 7)),
            "6", "CONTINUOUS",
        ]
    out += ["0", "ENDTAB", "0", "ENDSEC"]
    return out


def to_dxf(result: dict) -> str:
    """The layout as a minimal but complete R12-style DXF.

    Written by hand rather than with a DXF library because what is needed is
    small and fixed -- closed polylines on four layers -- and a dependency
    whose surface is a hundred entity types would be carried for none of them.

    The obstacles are included. A drawing showing only the cells makes the
    empty space where a pillar stands look like space that can be used, which
    is precisely the mistake this tool exists to prevent.
    """
    entities: List[str] = ["0", "SECTION", "2", "ENTITIES"]

    shape = result.get("shape") or []
    if shape:
        entities += _dxf_polyline(shape, LAYER_SHAPE)
    for obstacle in result.get("obstacles", []):
        entities += _dxf_polyline(obstacle, LAYER_OBSTACLE)
    for cell in result.get("complete_cells", []):
        entities += _dxf_polyline(cell, LAYER_COMPLETE)
    for cell in result.get("partial_cells", []):
        entities += _dxf_polyline(cell, LAYER_PARTIAL)
    entities += ["0", "ENDSEC"]

    header = ["0", "SECTION", "2", "HEADER",
              "9", "$ACADVER", "1", "AC1009",
              "0", "ENDSEC"]
    tables = _dxf_layer_table(
        [LAYER_SHAPE, LAYER_OBSTACLE, LAYER_COMPLETE, LAYER_PARTIAL])

    return "\n".join(header + tables + entities + ["0", "EOF"]) + "\n"
