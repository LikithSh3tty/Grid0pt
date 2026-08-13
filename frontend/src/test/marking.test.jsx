/**
 * Tests for marking obstacles on an uploaded plan.
 *
 * Clicking corners is easy to get wrong by one point, and until now the only
 * remedies were to close a wrong shape or clear everything and start again.
 * Both are the same complaint: the work is destructive to correct.
 *
 * The coordinate maths is exercised here too, because it is the part that has
 * to survive being wrong silently -- a mirrored obstacle blocks the wrong half
 * of the plan and still looks plausible.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

import ImageInput from "../components/ImageInput";

function uploadAPlan() {
  const onObstaclesChange = vi.fn();
  render(<ImageInput onFileSelected={() => {}}
                     onObstaclesChange={onObstaclesChange} />);

  const input = document.querySelector('input[type="file"]');
  fireEvent.change(input, {
    target: { files: [new File(["x"], "plan.png", { type: "image/png" })] },
  });

  // jsdom lays nothing out, so the image's geometry is stated rather than
  // measured: a 200x100 plan displayed at 400x200, twice its natural size.
  const img = document.querySelector(".image-preview");
  Object.defineProperty(img, "naturalWidth", { value: 200 });
  Object.defineProperty(img, "naturalHeight", { value: 100 });
  img.getBoundingClientRect = () => ({ left: 0, top: 0, width: 400, height: 200 });
  return { img, onObstaclesChange };
}

beforeEach(() => {
  global.URL.createObjectURL = vi.fn(() => "blob:plan");
});

async function markSquare(img) {
  for (const [x, y] of [[40, 40], [80, 40], [80, 80], [40, 80]]) {
    fireEvent.click(img, { clientX: x, clientY: y });
  }
}

describe("marking", () => {
  test("a click lands where it was made, with the y axis flipped once", async () => {
    // Display (40,40) on a 2x-scaled preview is natural (20,20); the tracer
    // flips y, so it must arrive as 100-20 = 80.
    const { img, onObstaclesChange } = uploadAPlan();
    await userEvent.click(screen.getByRole("button", { name: /mark obstacles/i }));

    fireEvent.click(img, { clientX: 40, clientY: 40 });
    fireEvent.click(img, { clientX: 80, clientY: 40 });
    fireEvent.click(img, { clientX: 80, clientY: 80 });
    await userEvent.click(screen.getByRole("button", { name: /close obstacle/i }));

    expect(onObstaclesChange).toHaveBeenLastCalledWith([
      [[20, 80], [40, 80], [40, 60]],
    ]);
  });

  test("a mis-clicked point can be taken back without losing the rest", async () => {
    const { img, onObstaclesChange } = uploadAPlan();
    await userEvent.click(screen.getByRole("button", { name: /mark obstacles/i }));

    fireEvent.click(img, { clientX: 40, clientY: 40 });
    fireEvent.click(img, { clientX: 80, clientY: 40 });
    fireEvent.click(img, { clientX: 300, clientY: 180 });   // the slip
    await userEvent.click(screen.getByRole("button", { name: /undo point/i }));
    fireEvent.click(img, { clientX: 80, clientY: 80 });
    await userEvent.click(screen.getByRole("button", { name: /close obstacle/i }));

    expect(onObstaclesChange).toHaveBeenLastCalledWith([
      [[20, 80], [40, 80], [40, 60]],
    ]);
  });

  test("undo does nothing when there is nothing to take back", async () => {
    uploadAPlan();
    await userEvent.click(screen.getByRole("button", { name: /mark obstacles/i }));

    expect(screen.getByRole("button", { name: /undo point/i })).toBeDisabled();
  });

  test("a closed obstacle can be removed without clearing the others", async () => {
    const { img, onObstaclesChange } = uploadAPlan();
    await userEvent.click(screen.getByRole("button", { name: /mark obstacles/i }));

    await markSquare(img);
    await userEvent.click(screen.getByRole("button", { name: /close obstacle/i }));
    fireEvent.click(img, { clientX: 120, clientY: 40 });
    fireEvent.click(img, { clientX: 160, clientY: 40 });
    fireEvent.click(img, { clientX: 160, clientY: 80 });
    await userEvent.click(screen.getByRole("button", { name: /close obstacle/i }));
    expect(onObstaclesChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ length: 2 }));

    await userEvent.click(screen.getByRole("button", { name: /remove last/i }));

    expect(onObstaclesChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ length: 1 }));
  });

  test("removing is offered only once something is there to remove", async () => {
    uploadAPlan();
    await userEvent.click(screen.getByRole("button", { name: /mark obstacles/i }));

    expect(screen.getByRole("button", { name: /remove last/i })).toBeDisabled();
  });
});
