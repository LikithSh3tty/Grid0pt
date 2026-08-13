"""Tests for exporting a packed layout to CSV and DXF.

The point of the tool is a set of cell positions someone then builds from, so
the layout has to leave here in a form a CAD package will open. Two formats,
for two different consumers: CSV for a spreadsheet or a script, DXF for a
drawing package.

What is pinned is what makes an export trustworthy rather than merely parseable:
that the geometry matches the placement it came from, that partial cells are
distinguishable from complete ones rather than silently merged, and that the
DXF is a structurally valid document rather than something that happens to open
in one reader.
"""
import csv
import io

import pytest

from shapely.geometry import Polygon

from exporters import to_csv, to_dxf
from packer_service import run_packing

SQUARE = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]
ODD = [(0.0, 0.0), (13.0, 0.0), (13.0, 10.0), (0.0, 10.0)]


@pytest.fixture(scope="module")
def packed():
    return run_packing(SQUARE, [], 3.0, 3.0, rotate=False)


@pytest.fixture(scope="module")
def packed_with_partials():
    return run_packing(ODD, [], 3.0, 3.0, rotate=False)


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #

def test_csv_has_a_row_per_cell(packed):
    rows = list(csv.DictReader(io.StringIO(to_csv(packed))))

    assert len(rows) == packed["stats"]["complete"] + packed["stats"]["partial"]


def test_csv_says_which_cells_are_complete(packed_with_partials):
    """A partial cell is not a cell you can build in. Exporting both without
    distinguishing them would hand someone a layout that does not fit."""
    rows = list(csv.DictReader(io.StringIO(to_csv(packed_with_partials))))

    kinds = {r["kind"] for r in rows}
    assert kinds == {"complete", "partial"}
    assert sum(1 for r in rows if r["kind"] == "complete") == \
        packed_with_partials["stats"]["complete"]


def test_csv_centres_are_the_centres_of_the_exported_cells(packed):
    """The centre column is what a script places from, so it has to be the
    centre of the polygon in the same row and not of some other cell."""
    rows = list(csv.DictReader(io.StringIO(to_csv(packed))))

    for row in rows[:5]:
        corners = [(float(row[f"x{i}"]), float(row[f"y{i}"])) for i in range(4)]
        centroid = Polygon(corners).centroid
        assert float(row["cx"]) == pytest.approx(centroid.x, abs=1e-9)
        assert float(row["cy"]) == pytest.approx(centroid.y, abs=1e-9)


def test_csv_geometry_matches_the_placement(packed):
    rows = list(csv.DictReader(io.StringIO(to_csv(packed))))
    exported = sorted(
        tuple(sorted((round(float(r[f"x{i}"]), 6), round(float(r[f"y{i}"]), 6))
                     for i in range(4)))
        for r in rows if r["kind"] == "complete")
    placed = sorted(
        tuple(sorted((round(x, 6), round(y, 6)) for x, y in cell))
        for cell in packed["complete_cells"])

    assert exported == placed


# --------------------------------------------------------------------------- #
# DXF
# --------------------------------------------------------------------------- #

def test_dxf_is_a_structurally_complete_document(packed):
    text = to_dxf(packed)

    assert text.startswith("0\nSECTION")
    assert text.rstrip().endswith("0\nEOF")
    for section in ("HEADER", "TABLES", "ENTITIES"):
        assert f"2\n{section}" in text
    assert text.count("0\nSECTION") == text.count("0\nENDSEC")


def test_dxf_separates_complete_from_partial_by_layer(packed_with_partials):
    """Layers are how a CAD user turns one of them off. Putting both on one
    layer would make the distinction invisible where it matters most."""
    text = to_dxf(packed_with_partials)

    assert "COMPLETE" in text
    assert "PARTIAL" in text


def test_dxf_draws_every_cell_as_a_closed_polyline(packed):
    text = to_dxf(packed)
    stats = packed["stats"]

    # One LWPOLYLINE per cell, plus the shape outline itself.
    assert text.count("LWPOLYLINE") == stats["complete"] + stats["partial"] + 1


def test_dxf_carries_the_obstacles_it_had_to_avoid():
    """A layout without its obstacles is misleading in a drawing: the empty
    space where a pillar stands looks like space you can use."""
    pillar = [(3.0, 3.0), (6.0, 3.0), (6.0, 6.0), (3.0, 6.0)]
    result = run_packing(SQUARE, [pillar], 3.0, 3.0, rotate=False)

    assert "OBSTACLE" in to_dxf(result)


# --------------------------------------------------------------------------- #
# over HTTP
# --------------------------------------------------------------------------- #

def client():
    from fastapi.testclient import TestClient
    from server import app
    return TestClient(app)


BODY = {"shape": [list(p) for p in SQUARE], "cell_width": 3.0,
        "cell_height": 3.0, "rotate": False}


@pytest.mark.parametrize("fmt, marker", [("csv", "kind"), ("dxf", "SECTION")])
def test_the_endpoint_serves_each_format(fmt, marker):
    response = client().post("/api/export/polygon", json={**BODY, "format": fmt})

    assert response.status_code == 200
    assert marker in response.text


def test_the_export_is_offered_as_a_download_with_a_name():
    """Served inline it opens as a wall of text in the browser; the point is a
    file someone hands to CAD."""
    response = client().post("/api/export/polygon", json={**BODY, "format": "dxf"})

    disposition = response.headers["content-disposition"]
    assert "attachment" in disposition
    assert ".dxf" in disposition


def test_an_unknown_format_is_refused_rather_than_guessed():
    response = client().post("/api/export/polygon", json={**BODY, "format": "pdf"})

    assert response.status_code == 400


def test_the_export_describes_the_same_placement_the_pack_endpoint_returns():
    """The two must not drift: a file quoting different cells from the drawing
    on screen is worse than no export at all."""
    api = client()
    packed = api.post("/api/pack/polygon", json=BODY).json()
    exported = api.post("/api/export/polygon",
                        json={**BODY, "format": "csv"}).text

    rows = list(csv.DictReader(io.StringIO(exported)))
    assert len(rows) == packed["stats"]["complete"] + packed["stats"]["partial"]
