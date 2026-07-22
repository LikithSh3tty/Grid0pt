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
