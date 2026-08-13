"""Every solver under test, behind one interface (design note section 11).

Three groups, and the distinction is what makes the table readable:

  baseline  -- what the repository did before this work: the uniform offset
               sweep at several step counts, with and without edge snapping, and
               the fixed 15-degree rotation ladder the service used to serve.
               These are the numbers the contribution is measured against, so
               they are called through the ORIGINAL `optimize`, unchanged.
  grid0pt   -- the method: exact translation, and the guided pipeline.
  ablation  -- the method with exactly one component switched off, so a column
               of the results table attributes the difference to that component
               and nothing else.
  reference -- a very dense sweep, used as a stand-in for the true optimum on
               instances where no optimum is known by construction. Expensive by
               design and therefore opt-in.

Every method is a function of a packer and returns a Placement, so the driver
times them and counts their evaluations identically. None of them may classify
during the search: the taxonomy is a read-back the driver performs once on the
winner, and letting a method pay for it inside the search would make its cost
incomparable with the baselines'.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from grid_packer import GridPacker, Placement


@dataclass(frozen=True)
class Method:
    name: str
    group: str
    run: Callable[[GridPacker], Placement]
    description: str


def _rotation_ladder(packer: GridPacker) -> Tuple[float, ...]:
    """The fixed angle ladder the service shipped: 15-degree steps over the
    placement period (90 for square cells, 180 for rectangular -- a quarter turn
    of a rectangular cell swaps cw and ch and is a genuinely different packing).
    Reproduced here rather than imported so the baseline stays pinned even if
    the service's own helper is retired."""
    period = 90 if packer.cw == packer.ch else 180
    return tuple(float(a) for a in range(0, period, 15))


# --------------------------------------------------------------------------- #
# baselines: the code as it was
# --------------------------------------------------------------------------- #

def _uniform(steps: int, snap: bool) -> Callable[[GridPacker], Placement]:
    def run(packer: GridPacker) -> Placement:
        best, _ = packer.optimize(steps=steps, snap_to_edges=snap)
        return best
    return run


def _fixed_ladder(steps: int) -> Callable[[GridPacker], Placement]:
    def run(packer: GridPacker) -> Placement:
        best, _ = packer.optimize(steps=steps, angles=_rotation_ladder(packer))
        return best
    return run


def _exact(packer: GridPacker) -> Placement:
    best, _ = packer.optimize_exact(angles=(0.0,))
    return best


def _columns(packer: GridPacker) -> Placement:
    best, _ = packer.optimize_columns(angles=(0.0,))
    return best


def _erosion(packer: GridPacker) -> Placement:
    best, _ = packer.optimize_erosion(angles=(0.0,))
    return best


def _guided(**kwargs) -> Callable[[GridPacker], Placement]:
    def run(packer: GridPacker) -> Placement:
        best, _ = packer.optimize_guided(**kwargs)
        return best
    return run


def _note_vote_period(packer: GridPacker) -> Placement:
    best, _ = packer.optimize_guided(vote_period=packer.rotation_period)
    return best


def _dense_reference(offset_steps: int, angle_step: float
                     ) -> Callable[[GridPacker], Placement]:
    """Ground truth by brute force: a fine offset sweep at every fine angle.

    This is not a method anyone would run; it is the yardstick. Note that on
    instances with a known optimum it is not needed at all, and where it IS used
    it is still only a lower bound on the true optimum -- a sweep can step over
    a sharp maximum, which is the very weakness section 3 raises. It is reported
    as "the best anyone found", never as "the optimum".
    """
    def run(packer: GridPacker) -> Placement:
        angles = [a * angle_step for a in range(int(round(90.0 / angle_step)))]
        if packer.cw != packer.ch:
            angles += [a + 90.0 for a in angles]
        best, _ = packer.optimize(steps=offset_steps, angles=angles)
        return best
    return run


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #

def build(quick: bool = True, with_reference: bool = False) -> List[Method]:
    """The methods to run. `quick` drops the redundant step counts."""
    steps = (10,) if quick else (6, 10, 16)

    methods: List[Method] = []

    for n in steps:
        methods.append(Method(
            f"uniform-s{n}", "baseline", _uniform(n, snap=True),
            f"uniform {n}x{n} offset sweep with edge snapping (the shipped default)"))
    methods.append(Method(
        "uniform-nosnap-s10", "baseline", _uniform(10, snap=False),
        "uniform sweep with snapping off: what a plain grid search finds"))
    methods.append(Method(
        "fixed15-s10", "baseline", _fixed_ladder(10),
        "uniform sweep at every 15 degrees: the rotation path the service shipped"))

    methods.append(Method(
        "exact", "grid0pt", _exact,
        "translation by enumerating the critical-offset arrangement, no rotation"))
    methods.append(Method(
        "columns", "grid0pt", _columns,
        "translation with the dy axis solved rather than enumerated, no rotation"))
    methods.append(Method(
        "erosion", "grid0pt", _erosion,
        "translation with BOTH axes solved: exact on any boundary, no rotation"))
    methods.append(Method(
        "guided", "grid0pt", _guided(),
        "the full pipeline: exact translation, fringe vote, refine, area stop"))

    methods += [
        Method("abl-norefine", "ablation", _guided(refine="none"),
               "no local refine: candidates only"),
        Method("abl-golden", "ablation", _guided(refine="golden"),
               "the note's golden-section refine instead of uniform probes"),
        Method("abl-certified", "ablation", _guided(refine="certified"),
               "refine by branch and bound rather than probing: exact in the window"),
        Method("abl-nostop", "ablation", _guided(recover_min=0.0),
               "no recoverable-area stop: always spend the refine budget"),
        Method("abl-columns", "ablation", _guided(translation="columns"),
               "solve the dy axis but enumerate dx: exact only where the boundary is"),
        Method("abl-enumerate", "ablation", _guided(translation="exact"),
               "enumerate the offset arrangement on both axes instead of solving them"),
        Method("abl-nogate", "ablation", _guided(r_min=0.0),
               "no R gate: rotate even when the fringe has no dominant direction"),
        Method("abl-weight0", "ablation", _guided(weight_power=0.0),
               "vote weight g(f) = 1: chord length only, near-complete cells not favoured"),
        # The note's linear weight, which was the default until certifying the
        # full corpus showed g(f) = f^2 reaching the proven optimum on two more
        # instances and fewer on none. Retained as the ablation that measures
        # the change.
        Method("abl-weight1", "ablation", _guided(weight_power=1.0),
               "vote weight g(f) = f: the note's linear weight, the previous default"),
        # The note's literal vote circle is the PLACEMENT period (m = 360/P);
        # this implementation votes on the alignment period, always 90. The
        # period depends on the packer, so this one reads it at call time.
        Method("abl-noteperiod", "ablation", _note_vote_period,
               "the note's literal vote circle m = 360/P instead of the alignment period"),
    ]

    if with_reference:
        methods.append(Method(
            "reference-dense", "reference", _dense_reference(40, 1.0),
            "40x40 offsets at every 1 degree: the best anyone found, by brute force"))

    return methods
