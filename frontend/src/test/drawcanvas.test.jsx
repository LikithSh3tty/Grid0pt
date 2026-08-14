/**
 * Tests for the drawing surface's coordinate space.
 *
 * The canvas maps clicks through its viewBox. When that box is square and the
 * element is not, the browser letterboxes: the drawable square sits centred and
 * clicks in the side margins map OUTSIDE the nominal range. Measured in a real
 * browser, clicking inside the canvas produced x from -36 to 124 against a
 * nominal 0 to 100.
 *
 * Nothing broke visually, because the viewBox then grew to include the stray
 * points -- which is exactly what made it worth fixing rather than ignoring:
 * the scale a person draws in silently depended on the window's aspect ratio,
 * so the same drawn rectangle meant different coordinates, and therefore a
 * different number of cells, at different window widths.
 */
import { render } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

import DrawCanvas from "../components/DrawCanvas";

function viewBoxOf(polygons) {
  render(<DrawCanvas polygons={polygons} onPolygonsChange={vi.fn()} />);
  return document.querySelector(".draw-svg")
    .getAttribute("viewBox").split(/\s+/).map(Number);
}

describe("the drawing surface", () => {
  test("is square before anything is drawn", () => {
    const [x, y, w, h] = viewBoxOf([]);

    expect([x, y]).toEqual([0, 0]);
    expect(w).toBe(h);
  });

  test("stays square when a shape runs past the nominal extent", () => {
    // Typed coordinates can exceed the default box. If the box grows on one
    // axis only it stops matching the element and the letterboxing returns,
    // which is the whole defect.
    const [, , w, h] = viewBoxOf([[[0, 0], [240, 0], [240, 30], [0, 30]]]);

    expect(w).toBe(h);
    expect(w).toBeGreaterThanOrEqual(240);
  });

  test("stays square around a shape drawn at negative coordinates", () => {
    const [x, y, w, h] = viewBoxOf([[[-40, -10], [60, -10], [60, 50], [-40, 50]]]);

    expect(w).toBe(h);
    expect(x).toBeLessThanOrEqual(-40);
    expect(y).toBeLessThanOrEqual(-10);
  });
});
