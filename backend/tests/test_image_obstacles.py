"""Tests for obstacles drawn onto an uploaded image.

Detection finds what is drawn as a hole in the plan. It cannot find what is not
drawn: a pillar the survey missed, a stairwell added since, an area someone has
decided to keep clear. Those have to be added by hand, on top of the image.

The coordinate space is the thing to get right and the easy thing to get wrong.
`image_boundary` flips the y axis when tracing, so the shape that comes back is
not in image-row order -- and an obstacle passed in raw image rows would land
mirrored, plausibly, and in the wrong place. So obstacles arrive in the SAME
space as the returned shape, which is the space the caller is looking at.
"""
import cv2
import numpy as np
import pytest

from packer_service import run_packing_from_image


def plan_bytes():
    """A 240x160 room, no detected holes: whatever obstacles appear are the
    ones passed in."""
    img = np.zeros((200, 300), np.uint8)
    cv2.rectangle(img, (30, 20), (269, 179), 255, thickness=-1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_a_plain_run_finds_no_obstacles():
    result = run_packing_from_image(plan_bytes(), 20, 20, rotate=False)

    assert result["obstacles"] == []


def test_a_drawn_obstacle_is_honoured():
    drawn = [[(100.0, 60.0), (160.0, 60.0), (160.0, 120.0), (100.0, 120.0)]]

    plain = run_packing_from_image(plan_bytes(), 20, 20, rotate=False)
    marked = run_packing_from_image(plan_bytes(), 20, 20, rotate=False,
                                    obstacle_points=drawn)

    assert len(marked["obstacles"]) == 1
    assert marked["stats"]["complete"] < plain["stats"]["complete"]


def test_no_cell_lands_on_a_drawn_obstacle():
    """The point of marking it. A cell overlapping the blocked area would make
    the drawing a suggestion rather than a constraint."""
    from shapely.geometry import Polygon

    drawn = [[(100.0, 60.0), (160.0, 60.0), (160.0, 120.0), (100.0, 120.0)]]
    result = run_packing_from_image(plan_bytes(), 20, 20, rotate=False,
                                    obstacle_points=drawn)

    blocked = Polygon(drawn[0])
    for cell in result["complete_cells"]:
        assert Polygon(cell).intersection(blocked).area == pytest.approx(0.0, abs=1e-9)


def test_drawn_obstacles_are_in_the_same_space_as_the_returned_shape():
    """The mirroring trap: an obstacle in the top half of the returned shape
    must block cells in the top half, not the bottom."""
    from shapely.geometry import Polygon

    result = run_packing_from_image(plan_bytes(), 20, 20, rotate=False)
    minx, miny, maxx, maxy = Polygon(result["shape"]).bounds
    midy = (miny + maxy) / 2.0
    upper = [[(minx + 20, midy + 10), (maxx - 20, midy + 10),
              (maxx - 20, maxy - 10), (minx + 20, maxy - 10)]]

    marked = run_packing_from_image(plan_bytes(), 20, 20, rotate=False,
                                    obstacle_points=upper)

    blocked = Polygon(upper[0])
    above = [c for c in marked["complete_cells"]
             if Polygon(c).centroid.y > midy]
    for cell in above:
        assert Polygon(cell).intersection(blocked).area == pytest.approx(0.0, abs=1e-9)


def test_a_malformed_obstacle_is_refused():
    with pytest.raises(ValueError):
        run_packing_from_image(plan_bytes(), 20, 20, rotate=False,
                               obstacle_points=[[(0.0, 0.0), (1.0, 1.0)]])


def test_an_obstacle_covering_everything_is_refused_clearly():
    """Better a stated error than an empty result the caller has to interpret."""
    huge = [[(-1e4, -1e4), (1e4, -1e4), (1e4, 1e4), (-1e4, 1e4)]]

    with pytest.raises(ValueError):
        run_packing_from_image(plan_bytes(), 20, 20, rotate=False,
                               obstacle_points=huge)


# --------------------------------------------------------------------------- #
# over HTTP
# --------------------------------------------------------------------------- #

def post(client, **extra):
    return client.post(
        "/api/pack/image",
        files={"file": ("plan.png", plan_bytes(), "image/png")},
        data={"cell_width": "20", "cell_height": "20", "rotate": "false", **extra},
    )


def api():
    from fastapi.testclient import TestClient
    from server import app
    return TestClient(app)


def test_the_endpoint_takes_obstacles_marked_on_the_image():
    import json

    client = api()
    plain = post(client).json()
    marked = post(client, obstacles=json.dumps(
        [[[100, 60], [160, 60], [160, 120], [100, 120]]])).json()

    assert len(marked["obstacles"]) == 1
    assert marked["stats"]["complete"] < plain["stats"]["complete"]


def test_a_client_that_sends_none_behaves_as_before():
    """Every existing client omits the field entirely."""
    assert post(api()).status_code == 200


def test_obstacles_that_are_not_json_are_refused_with_a_reason():
    response = post(api(), obstacles="not json")

    assert response.status_code == 400
    assert "JSON" in response.json()["detail"]
