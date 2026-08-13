/**
 * Tests for the parts of the UI that carry the solver's claims.
 *
 * These components were built and shipped verified by `npm run build` plus one
 * manual pass in a browser, which proves they compile and that one path works.
 * What they say about optimality is the strongest statement this app makes, and
 * "it rendered once when I looked at it" is not a standard to hold that to --
 * so the pieces that decide WHICH claim appears are pinned here.
 *
 * Deliberately not covered: that the backend agrees. The verdicts are computed
 * from stats the API returns, and a test that invents those stats can only
 * check the reading of them. Whether the API produces them is a backend test,
 * and whether the two meet is the manual browser pass.
 */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import ResultView from "../components/ResultView";
import DrawCanvas from "../components/DrawCanvas";
import { packImage, packPolygon } from "../api";

const SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]];

function result(stats) {
  return {
    shape: SQUARE,
    obstacles: [],
    complete_cells: [],
    partial_cells: [],
    stats: {
      complete: 12, partial: 4, coverage: 0.9, dx: 0, dy: 0, angle: 23,
      ...stats,
    },
  };
}

describe("the verdicts", () => {
  test("says a result is proven only when the search actually closed", () => {
    render(<ResultView result={result({
      rotation_bound: 12, rotation_gap: 0,
      rotation_optimal: true, rotation_exhausted: true, rotation_nodes: 15,
    })} />);

    expect(screen.getByText("Proven optimal")).toBeInTheDocument();
    expect(screen.getByText(/at any angle/)).toBeInTheDocument();
  });

  test("a search that ran out of budget is reported as not proven", () => {
    // The honest failure: a valid bound, no proof, and it must not read as one.
    render(<ResultView result={result({
      complete: 45,
      rotation_bound: 46, rotation_gap: 1,
      rotation_optimal: false, rotation_exhausted: false, rotation_nodes: 600,
    })} />);

    expect(screen.getByText("Not proven")).toBeInTheDocument();
    expect(screen.queryByText("Proven optimal")).not.toBeInTheDocument();
  });

  test("a result with no proof says so rather than staying silent", () => {
    // Silence reads as success. Translation is exact on every request, so the
    // gap a reader needs told about is the angle.
    render(<ResultView result={result({})} />);

    expect(screen.queryByText("Proven optimal")).not.toBeInTheDocument();
    expect(screen.getByText(/Prove optimal/)).toBeInTheDocument();
  });

  test("the partial-cell floor is shown as its own separate claim", () => {
    // Never merged with the rotation certificate: one bounds completes from
    // above over all angles, the other bounds partials from below.
    render(<ResultView result={result({
      partial: 11, irreducible: 5, partial_floor: 5,
      optimality_gap: 6, certified: true, recoverable_area: 0,
    })} />);

    expect(screen.getByText(/Within 6 partial cells of the floor/)).toBeInTheDocument();
  });

  test("a floor whose assumption failed is declined, not quoted", () => {
    render(<ResultView result={result({
      partial: 11, irreducible: 5, partial_floor: 5,
      optimality_gap: 6, certified: false, recoverable_area: 0,
    })} />);

    expect(screen.getByText(/not bounded here/)).toBeInTheDocument();
    expect(screen.queryByText(/Within 6 partial cells/)).not.toBeInTheDocument();
  });
});

describe("the request", () => {
  beforeEach(() => {
    global.fetch = vi.fn(async () => ({ ok: true, json: async () => ({}) }));
  });
  afterEach(() => vi.restoreAllMocks());

  test("polygon requests carry the certify flag", async () => {
    await packPolygon(SQUARE, [], 3, 3, true, true);

    const body = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(body.certify).toBe(true);
    expect(body.rotate).toBe(true);
  });

  test("certify defaults off, so an ordinary request pays nothing", async () => {
    await packPolygon(SQUARE, [], 3, 3, false);

    expect(JSON.parse(global.fetch.mock.calls[0][1].body).certify).toBe(false);
  });

  test("image requests carry it too", async () => {
    await packImage(new File(["x"], "plan.png"), 3, 3, true, true);

    const form = global.fetch.mock.calls[0][1].body;
    expect(form.get("certify")).toBe("true");
  });
});

describe("the coordinate box", () => {
  test("accepts the labelled form the placeholder demonstrates", async () => {
    // Regression: typing exactly what the placeholder showed used to fail with
    // `invalid point "shape:"`.
    const onPolygonsChange = vi.fn();
    render(<DrawCanvas polygons={[]} onPolygonsChange={onPolygonsChange} />);

    await userEvent.type(
      screen.getByPlaceholderText(/shape:/),
      "shape: 0,0 10,0 10,10 0,10");

    expect(onPolygonsChange).toHaveBeenLastCalledWith([
      [[0, 0], [10, 0], [10, 10], [0, 10]],
    ]);
  });

  test("still rejects something that is not a coordinate", async () => {
    const onPolygonsChange = vi.fn();
    render(<DrawCanvas polygons={[]} onPolygonsChange={onPolygonsChange} />);

    await userEvent.type(screen.getByPlaceholderText(/shape:/), "not a point");

    expect(screen.getByText(/invalid point/)).toBeInTheDocument();
  });
});
