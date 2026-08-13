"""
demo.py — runnable examples for grid_packer.GridPacker

Each example runs what the packer actually does now, not what it did first.
`optimize` is still here as the ORIGINAL uniform sweep and is what the
evaluation's baselines call, but nothing serves requests through it any more:
translation is solved on both axes by `optimize_erosion`, and the angle comes
from the fringe vote in `optimize_guided`. A demo showing the sweep would be
demonstrating the thing this package replaced.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from grid_packer import GridPacker


def rectangle_example():
    # A 20x12 rectangle whose corner is NOT at the origin. A grid pinned to
    # (0,0) slices every edge cell; sliding the grid by the offset realigns it.
    ox, oy = 0.9, 0.6
    shape = Polygon([(ox, oy), (ox + 20, oy), (ox + 20, oy + 12), (ox, oy + 12)])
    obstacles = [
        Polygon([(ox + 5, oy + 4), (ox + 9, oy + 4),
                 (ox + 9, oy + 8), (ox + 5, oy + 8)]),
        Polygon([(ox + 14, oy + 1), (ox + 18, oy + 1),
                 (ox + 18, oy + 3), (ox + 14, oy + 3)]),
    ]
    packer = GridPacker(shape, obstacles, cell_width=2.0, cell_height=2.0)

    # baseline: grid pinned to origin, no search
    baseline = packer.evaluate(0, 0)
    # solved: the offsets are not searched at all, they are read off the
    # deepest overlap of the region eroded by the cell -- one evaluation.
    best, _ = packer.optimize_erosion()

    print("RECTANGLE")
    print("  baseline :", baseline)
    print("  optimized:", best)
    print(f"  gain     : +{best.complete - baseline.complete} complete cells")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    packer.plot(baseline, ax=a1, title=f"Baseline (grid at origin)\n{baseline!r}")
    packer.plot(best, ax=a2, title=f"Optimized placement\n{best!r}")
    fig.tight_layout()
    fig.savefig("result_rectangle.png", dpi=110)
    plt.close(fig)


def l_shape_example():
    # an L-shaped boundary to prove "any shape" works
    shape = Polygon([(0, 0), (16, 0), (16, 6), (7, 6), (7, 14), (0, 14)])
    obstacles = [Polygon([(2, 2), (5, 2), (5, 5), (2, 5)])]
    packer = GridPacker(shape, obstacles, cell_width=1.5, cell_height=1.5)

    # also let the grid rotate this time: the angle comes from the fringe's own
    # orientation vote rather than from a ladder of candidate angles.
    best, _ = packer.optimize_guided()
    print("\nL-SHAPE")
    print("  solved   :", best)
    print(f"  vote     : R={best.rotation_vote.resultant:.3f}, "
          f"turned={best.rotation_vote.confident()}")

    fig, ax = plt.subplots(figsize=(8, 8))
    packer.plot(best, ax=ax)
    fig.tight_layout()
    fig.savefig("result_lshape.png", dpi=110)
    plt.close(fig)


def image_example():
    # synthesize a "floor plan": white L-shaped room with two dark obstacles,
    # then let from_image() detect the boundary and run the optimizer on it
    import cv2
    import numpy as np

    img = np.zeros((400, 600), np.uint8)
    cv2.fillPoly(img, [np.array([(40, 40), (560, 40), (560, 220),
                                 (300, 220), (300, 360), (40, 360)])], 255)
    cv2.rectangle(img, (120, 120), (200, 180), 0, thickness=-1)
    cv2.circle(img, (450, 130), 35, 0, thickness=-1)
    cv2.imwrite("demo_plan.png", img)

    packer = GridPacker.from_image("demo_plan.png", cell_width=40, cell_height=40)
    best, _ = packer.optimize_erosion()
    certificate = packer.certificate(
        packer.evaluate(best.dx, best.dy, best.angle, classify=True))

    print("\nIMAGE")
    print(f"  detected : shape area={packer.shape.area:.0f}px², "
          f"{len(packer.obstacles)} obstacle(s)")
    print("  solved   :", best)
    print(f"  certified: {certificate!r}")

    fig, ax = plt.subplots(figsize=(10, 7))
    packer.plot(best, ax=ax)
    fig.tight_layout()
    fig.savefig("result_image.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    rectangle_example()
    l_shape_example()
    image_example()
    print("\nSaved: result_rectangle.png, result_lshape.png, result_image.png")
