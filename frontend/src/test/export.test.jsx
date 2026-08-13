/**
 * Tests for the export controls.
 *
 * The interesting part is not the fetch, it is that the buttons only appear
 * when there is something to export and that they ask for the layout actually
 * on screen. An export describing a different placement from the drawing beside
 * it is worse than no export.
 */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import ExportButtons from "../components/ExportButtons";

const REQUEST = {
  shape: [[0, 0], [12, 0], [12, 9], [0, 9]],
  obstacles: [],
  cell_width: 3,
  cell_height: 3,
  rotate: false,
  certify: false,
};

describe("the export controls", () => {
  beforeEach(() => {
    global.fetch = vi.fn(async () => ({
      ok: true,
      blob: async () => new Blob(["x"]),
      headers: new Headers(),
    }));
    global.URL.createObjectURL = vi.fn(() => "blob:x");
    global.URL.revokeObjectURL = vi.fn();
  });
  afterEach(() => vi.restoreAllMocks());

  test("nothing is offered before there is a result", () => {
    render(<ExportButtons request={null} />);

    expect(screen.queryByRole("button", { name: /csv/i })).not.toBeInTheDocument();
  });

  test("asks for the layout that is on screen, not a fresh one", async () => {
    render(<ExportButtons request={REQUEST} />);

    await userEvent.click(screen.getByRole("button", { name: /csv/i }));

    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toContain("/api/export/polygon");
    expect(JSON.parse(options.body)).toMatchObject({ ...REQUEST, format: "csv" });
  });

  test("each format asks for itself", async () => {
    render(<ExportButtons request={REQUEST} />);

    await userEvent.click(screen.getByRole("button", { name: /dxf/i }));

    expect(JSON.parse(global.fetch.mock.calls[0][1].body).format).toBe("dxf");
  });

  test("a failed export says so instead of downloading nothing", async () => {
    global.fetch = vi.fn(async () => ({
      ok: false,
      json: async () => ({ detail: "unknown export format" }),
      headers: new Headers(),
    }));
    render(<ExportButtons request={REQUEST} />);

    await userEvent.click(screen.getByRole("button", { name: /csv/i }));

    expect(await screen.findByText(/unknown export format/)).toBeInTheDocument();
  });
});

describe("marking obstacles on an uploaded plan", () => {
  test("the image request carries them, as JSON", async () => {
    global.fetch = vi.fn(async () => ({ ok: true, json: async () => ({}) }));
    const { packImage } = await import("../api");
    const rings = [[[10, 10], [20, 10], [20, 20]]];

    await packImage(new File(["x"], "plan.png"), 3, 3, false, false, rings);

    const form = global.fetch.mock.calls[0][1].body;
    expect(JSON.parse(form.get("obstacles"))).toEqual(rings);
  });

  test("sending none is an empty list, not a missing field", async () => {
    global.fetch = vi.fn(async () => ({ ok: true, json: async () => ({}) }));
    const { packImage } = await import("../api");

    await packImage(new File(["x"], "plan.png"), 3, 3, false);

    expect(global.fetch.mock.calls[0][1].body.get("obstacles")).toBe("[]");
  });
});
