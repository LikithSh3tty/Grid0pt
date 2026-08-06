"""
grid_packer.py
==============

Lay a regular grid over an arbitrary shape that contains obstacles, and find the
grid placement that yields the MOST complete cells (and the fewest partial ones).

Definitions
-----------
- shape      : the outer boundary you are allowed to fill (any polygon, not just
               a rectangle: L-shapes, circles approximated as polygons, etc.)
- obstacles  : holes / blocked areas inside the shape that the grid must avoid.
- usable     : shape minus obstacles. This is the region a cell must sit inside.
- complete   : a grid cell lying ENTIRELY inside the usable region.
- partial    : a grid cell that overlaps the usable region but pokes outside it
               (crosses the boundary) or clips an obstacle.
- outside    : a cell with no usable AREA at all -> ignored. A cell that merely
               touches the region along a line or a point has zero area in
               common with it and is outside, not partial.

The grid can slide in every direction (the "up / down / left / right / center"
shift) and optionally rotate, because moving the grid by a fraction of a cell
changes how many cells fall cleanly inside the shape. We sweep those placements
and keep the best one.

Requires: shapely, matplotlib, numpy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep


# --------------------------------------------------------------------------- #
# coordinate resolution -- ONE quantity, used in two places
# --------------------------------------------------------------------------- #
# `_wrap` reduces a critical offset modulo the cell period and rounds it to
# _COORD_DECIMALS decimals, so the grid line it produces can sit up to
# 0.5 * 10**-_COORD_DECIMALS away from the wall it was derived from. The
# rotate-by-(-angle)-then-back trick in `evaluate` adds a few ulp of its own
# drift on top: a wall computed as 0.4 comes back as 0.40000000000000036.
#
# `evaluate`'s containment test must carry that SAME resolution, or a cell that
# tiles the region exactly is a few ulp short of the wall, `contains` is
# strictly false, and the flush placement -- precisely the optimum the search
# exists to find, and precisely what the rotated analysis produces -- is
# misreported as partial. The rounding below and the tolerance below it are the
# same quantity seen twice; they are defined together so they cannot drift
# apart.
#
# _GEOM_TOL is 20x the worst-case rounding granularity (so it strictly covers
# it) and still eight or more orders of magnitude below any cell dimension in
# use, so it can never absorb a geometrically real overlap.
_COORD_DECIMALS = 9
_GEOM_TOL = 10.0 ** -(_COORD_DECIMALS - 1)      # 1e-8 vs 5e-10 of rounding


# --------------------------------------------------------------------------- #
# area resolution -- the OTHER half of the classification, and a DIFFERENT
# quantity from the two above
# --------------------------------------------------------------------------- #
# The three classes are defined by area: a cell C is complete when C is a subset
# of U, partial when 0 < area(C n U) < area(C), and outside when
# area(C n U) = 0. `_GEOM_TOL` above is a LENGTH slack on the complete/partial
# side; it is deliberately coarse (1e-8) because it has to absorb `_wrap`'s
# rounding of the offsets. This constant is the partial/outside side, and it
# must NOT be derived from `_GEOM_TOL`: nothing rounds an area, so the only
# thing to discount here is floating-point noise, which sits ~1e-15.
#
# `_generate_cells` tiles `region.bounds`, and the rotate-by-(-angle) round trip
# leaves a bound such as maxx = 12.000000000000002. The extra row/column that
# emits therefore overlaps the region in a sliver one coordinate-ulp wide
# (~eps * L, for a region of extent L) running along a cell edge (length c), so
# its inside-fraction is ~eps * L / c -- 9e-16 for the 12x9-at-3x3 instance,
# where `intersects()` is nevertheless True. Being a FRACTION of the cell area
# rather than an absolute area, the bound below is scale-free: it holds at any
# cell size, and its headroom is spent on the grid's aspect L/c, tolerating
# L/c up to ~1e-12 / eps ~= 4.5e3 cells across (2e7 cells) before the noise
# floor could reach it.
#
# It must stay FAR below any real inside-fraction: the taxonomy's class F
# ("grazing sliver", f ~ 0, the boundary just clips a corner) is a genuine
# partial and has to keep being counted. 1e-12 sits ~1000x above the noise it
# removes and ~1e6x below the smallest fraction any test calls real, so it can
# only ever delete a numerically-zero overlap. Widening it toward a
# "drop small partials" cutoff would silently destroy class F.
_AREA_TOL_REL = 1e-12


# --------------------------------------------------------------------------- #
# taxonomy resolution -- the ONE quantity the existing constants cannot supply
# --------------------------------------------------------------------------- #
# The partial-cell taxonomy separates class F ("grazing sliver", f ~ 0, the cell
# is essentially outside, no move recovers it) from class B3 ("corner triangle,
# f small, marginal but real"). That split needs a threshold on the inside
# fraction f, and NEITHER existing constant can serve as it:
#
#   * `_AREA_TOL_REL` (1e-12) is the partial/outside boundary -- the level below
#     which an overlap is floating-point noise rather than geometry. Reusing it
#     here would make class F unreachable: every cell above the noise floor is
#     already counted as partial, so F would be the empty class. The comment on
#     `_AREA_TOL_REL` says as much ("widening it toward a 'drop small partials'
#     cutoff would silently destroy class F") -- the two are opposite ends of
#     the same axis and must not be conflated.
#   * `_GEOM_TOL` (1e-8) is a LENGTH slack on coordinates, not an area fraction;
#     a cell whose clip is 1e-8 of its area is not a coordinate rounding
#     artefact, it is a real (if useless) sliver.
#
# So F is a MODELLING threshold, not a numerical one: it states how little of a
# cell has to be inside before the cell is declared unrecoverable. 1e-3 means
# "less than a tenth of a percent of the cell" -- nine orders of magnitude above
# the noise floor it must never touch, and two to three orders below any inside
# fraction the taxonomy calls small (a B3 corner triangle with legs a tenth of a
# cell already has f = 5e-3). It is exposed as the `sliver_fraction` argument of
# `_classify_partial` precisely because it is a modelling choice the paper
# should report a sensitivity study for, not a constant of the geometry.
_SLIVER_FRACTION = 1e-3


# --------------------------------------------------------------------------- #
# rotation-vote modelling constants (design note section 8)
# --------------------------------------------------------------------------- #
# Every constant below is a MODELLING choice, not a property of the geometry.
# Each is named, defaulted here and overridable per call, because the paper runs
# an ablation over each one. The trade-off each controls is stated with it.

#: The circle the fringe orientations are averaged on, in degrees.
#:
#: The note writes the multiplier as m = 360/P with P the ROTATION PERIOD of the
#: placement (90 for square cells, 180 for rectangular). That conflates two
#: different periodicities and is wrong for rectangular cells:
#:
#:   * The PLACEMENT period P is 90 or 180: turning a rectangular cell by 90
#:     degrees swaps its width and height, so theta = 0 and theta = 90 are
#:     genuinely different tilings and theta must range over [0, 180).
#:   * ALIGNMENT, which is what the vote is actually about, always has period
#:     90: a wall is flush with a grid line exactly when its orientation is
#:     0 mod 90, whatever the cell's aspect ratio, because the grid has both
#:     horizontal and vertical lines.
#:
#: Averaging on the placement circle with m = 2 makes the two perpendicular wall
#: families of any rectangular room ANTIPODAL, so they cancel: measured on a
#: 12x9 rectangle tilted 23 degrees with 3x2 cells, m = 2 gives R = 0.11 (below
#: any sensible gate, so the method declines to rotate and keeps 10 complete
#: cells) where m = 4 gives R = 1.00 and a candidate that reaches 18. The vote
#: is therefore taken on the alignment circle always, and the placement period
#: enters where it belongs -- in the RANGE of candidate angles, which is why
#: `RotationVote` carries both numbers. For square cells the two agree exactly
#: and this is the note's formula unchanged.
#:
#: `_dominant_orientations(vote_period=...)` overrides it, so the note's literal
#: m = 360/P remains runnable as an ablation.
_ALIGNMENT_PERIOD = 90.0

#: Vote-confidence gate: below this resultant length the fringe has no dominant
#: direction and the grid stays put.
#:
#: Trade-off. R is the weighted concentration of the fringe orientations on the
#: alignment circle; it is 1 when every oblique chord points the same way and 0
#: when they are spread uniformly. Too LOW a gate spends candidate solves on
#: shapes with no wall to align to (a disc: measured R = 0.02-0.08); too HIGH a
#: gate refuses to turn for a shape with a real but noisy dominant wall. 0.35
#: sits an order of magnitude above the disc measurements and well below the
#: R = 1.0 a clean tilted room produces, so both ends have wide margin. Note
#: that the gate can only cost QUALITY, never correctness: `optimize_guided`
#: compares every candidate against the un-rotated base and keeps the best.
R_MIN = 0.35

#: Width of the orientation bins the candidate list is clustered into, degrees.
#:
#: The note bins with `round(phi)`, i.e. 1-degree bins centred on the integers.
#: Two things are made deliberate here. (i) The bins are cyclic and are offset
#: by half a width so that a bin is CENTRED on 0 -- the alignment direction,
#: where wall orientations pile up -- instead of split across a bin edge.
#: (ii) The angle reported for a bin is the weighted circular MEAN of its
#: members, not the bin's centre, so the bin width sets the clustering
#: resolution and NOT the angular accuracy of the candidate: the headline
#: instances recover 12 / 23 / 31 degrees exactly at any bin width.
#:
#: Trade-off. Narrower than the scatter of a real wall (a boundary decimated by
#: Douglas-Peucker scatters its chord orientations by 1-2 degrees) fragments one
#: wall into several candidates and wastes the candidate budget; wider than
#: about 5 degrees merges genuinely distinct wall families -- at 5 degrees a
#: 3-unit chord drifts 0.26 units transversely, a tenth of a cell, which is a
#: real difference in the count. 2 degrees sits between the two.
VOTE_BIN_DEG = 2.0

#: How many orientation CLUSTERS become candidate angles (the note: "usually
#: 1-4"). Each candidate costs one exact translation solve, so this is the
#: quality/cost dial of the whole method. Clusters are ranked by accumulated
#: vote weight, so the cap only ever drops the weakest walls. With rectangular
#: cells each cluster yields two candidate angles (phi and phi+90, see
#: `_ALIGNMENT_PERIOD`), so the emitted list can be up to twice this long.
MAX_CANDIDATES = 4

#: Half-width, in degrees, of the local refine window around each candidate.
#: The note prescribes "+/- a few degrees". Wider windows drift toward being a
#: scan again (the thing the method exists to remove); narrower ones cannot
#: reach the compromise angle between two wall families that the refine is for.
REFINE_HALF_WINDOW = 5.0

#: Number of exact-translation solves the refine may spend inside that window.
REFINE_PROBES = 8

#: Inside fraction above which a partial cell counts as worth reclaiming, used
#: by the certificate's `recoverable_area` (design note section 8.4's stop
#: criterion, "sum over f > tau of (1 - f) * area(C)").
#:
#: Trade-off. The quantity answers "how much is still on the table?", so it must
#: not count cells that are mostly outside: a B3 corner triangle at f = 0.05 has
#: 0.95 of a cell outside the region, and summing those would report a large
#: upside that no rotation can collect. 0.5 counts a cell only when more of it
#: is in than out, which is also the point above which the taxonomy calls a
#: partial recoverable in practice (A slabs, B1 pentagons, C2 reflex corners).
RECOVERABLE_FRACTION_MIN = 0.5

#: The note's section 8.4 stop criterion, in CELL AREAS: once the fringe of the
#: best placement found so far holds less recoverable area than this, the local
#: refine is skipped -- "no upside left to turn for".
#:
#: Placement. The note reads this at the end of the pipeline; the tempting place
#: is the entrance, as a should-we-rotate gate. Measured, the entrance separates
#: nothing: a 24x18 room tilted 12 degrees, where rotating is worth +14 complete
#: cells, carries 2.89 cells of recoverable area before rotating, and a disc,
#: where rotating is worth nothing, carries 3.10. Any threshold silencing the
#: disc silences the headline. That question is R's, and R answers it (0.029 for
#: the disc against 1.000 for the tilted room). What the fringe area DOES answer
#: is the one asked a stage later: after the vote's angle has been solved, is
#: there anything left for the refine to find?
#:
#: Trade-off. Measured after the candidate solves, an instance whose walls came
#: out flush holds exactly 0.00 cells; one still compromising between two wall
#: families holds 1.2-1.7. Anything in between works; 1.0 is stated in the unit
#: that makes the argument, since under one cell of reclaimable area there is
#: not enough on the table to complete even one more cell. The stop costs one
#: classifying evaluation and saves `REFINE_PROBES` exact translation solves --
#: 105 evaluations against 905 on the tilted room, for the same count. Set to 0
#: to run the note's un-gated pipeline (an ablation axis of section 11).
RECOVERABLE_AREA_MIN = 1.0

#: Which local refine to run: "none", "grid" or "golden".
#:
#: The note prescribes a golden-section search. Golden-section assumes a
#: unimodal CONTINUOUS objective; N_complete(theta) is integer-valued and
#: piecewise-constant, so on a plateau -- which is most of any window -- the
#: bracket contracts on a comparison between two equal values and the search
#: converges to an arbitrary interior point rather than to a maximum. Measured
#: over the instances in `tests/test_guided_rotation.py` at an equal probe
#: budget, golden-section never beat the un-refined candidate, and never beat
#: plain uniform sampling of the same window. The default is therefore "grid",
#: which is what the evidence supports; "golden" is kept runnable so the paper's
#: ablation can report the comparison rather than assert it.
REFINE_METHOD = "grid"


class PartialClass(str, Enum):
    """The nine partial-cell morphologies (design note section 6.1).

    Subclasses `str` so a label compares equal to its own name ("A", "B1", ...)
    and serialises straight into JSON stats without a conversion step.
    """

    A = "A"      # axis-aligned slab: one cut parallel to a grid axis, K a rectangle
    B1 = "B1"    # oblique cut, pentagon: a corner sliced off, f ~ 1
    B2 = "B2"    # oblique cut, trapezoid: cut across opposite cell edges
    B3 = "B3"    # oblique cut, triangle: only a corner kept, f small
    C1 = "C1"    # convex region vertex inside the cell: K a wedge, f small
    C2 = "C2"    # reflex region vertex inside the cell: K an L-shape, f large
    D = "D"      # obstacle bite: an interior hole intrudes, K = cell - bite
    E = "E"      # sub-cell feature / neck: two or more cuts, K a band or split
    F = "F"      # grazing sliver: f ~ 0, the cell is essentially outside

    def __str__(self) -> str:                      # pragma: no cover - display
        return self.value


#: Classes whose cut is oblique, i.e. the ones that vote in the rotation
#: step (design note section 7.1: "oblique cut -> contribute a rotation vote").
#: C2/D are included only when their chords are actually oblique, which is a
#: per-cell property, so they are not listed here -- use `oblique_chords`.
OBLIQUE_CLASSES = frozenset({PartialClass.B1, PartialClass.B2, PartialClass.B3})

#: Classes resolved exactly by translation (design note section 9, regime T).
TRANSLATION_CLASSES = frozenset(
    {PartialClass.A, PartialClass.C1, PartialClass.C2, PartialClass.D})

#: Classes that no placement can recover (design note section 9, regime X).
IRREDUCIBLE_CLASSES = frozenset({PartialClass.E, PartialClass.F})


@dataclass(frozen=True)
class Chord:
    """One straight piece of a clip's boundary that lies on the REGION boundary.

    A clip K = C n U is bounded partly by cell edges (where the grid cut it) and
    partly by dU (where the region cut it). Only the latter carries information
    about how the boundary crosses the cell, and only the latter is a chord.
    That distinction is the whole basis of the taxonomy.

    All quantities are in the ANALYSIS frame -- the frame in which the grid is
    axis-aligned. `angle_deg` is therefore an orientation relative to the GRID
    axes, which is exactly what the rotation vote (design note section 8) needs.

    length          : chord length.
    angle_deg       : orientation in [0, 180). A chord has no direction, so
                      pointing "up-right" and "down-left" are the same chord and
                      must report the same angle; the reduction modulo 180
                      enforces that.
    axis_aligned    : the chord is parallel to a grid axis to within `_GEOM_TOL`
                      of TRANSVERSE deviation (see `_make_chord`).
    inside_fraction : f of the cell this chord came from. Carried on the chord
                      so the rotation vote's weight w = L * g(f) is a pure
                      function of the chord (design note section 8.1).
    on_hole         : the chord lies on an INTERIOR ring of the usable region,
                      i.e. it is an obstacle edge rather than an outer wall.
    cut_index       : which connected run of dU-boundary this chord belongs to.
                      Chords sharing a cut_index meet at a region vertex.
    geometry        : the chord itself, in the analysis frame.
    """

    length: float
    angle_deg: float
    axis_aligned: bool
    inside_fraction: float
    on_hole: bool
    cut_index: int
    geometry: LineString


@dataclass(frozen=True)
class PartialClassification:
    """The taxonomy read-off for one partial cell (design note sections 6-7).

    label                  : the A-F class.
    inside_fraction        : f = area(K) / area(C).
    chords                 : every straight dU piece of dK, in the analysis
                             frame, in ring order.
    cut_count              : number of CONNECTED runs of dU along dK. This is
                             the note's "number of boundary cuts through the
                             cell" -- a cut that turns at a region vertex inside
                             the cell is ONE cut made of two chords, which is
                             what keeps class C from colliding with class E.
    clip_vertex_count      : corners of K's largest component (collinear points
                             removed). 3 / 4 / 5 is the B3 / B2 / B1 split.
    interior_vertex_count  : region vertices strictly inside the cell.
    interior_convex_count  : how many of those are convex (shape pokes in).
    interior_reflex_count  : how many are reflex (concave corner).
    on_hole                : any chord lies on an obstacle ring.
    disconnected           : K has more than one connected component.
    """

    label: PartialClass
    inside_fraction: float
    chords: Tuple[Chord, ...]
    cut_count: int
    clip_vertex_count: int
    interior_vertex_count: int
    interior_convex_count: int
    interior_reflex_count: int
    on_hole: bool
    disconnected: bool

    @property
    def oblique_chords(self) -> Tuple[Chord, ...]:
        """The chords that vote in the rotation step (section 8.1)."""
        return tuple(c for c in self.chords if not c.axis_aligned)

    @property
    def axis_aligned_chords(self) -> Tuple[Chord, ...]:
        return tuple(c for c in self.chords if c.axis_aligned)

    @property
    def chord_length(self) -> float:
        """Total length of region boundary crossing this cell."""
        return sum(c.length for c in self.chords)

    @property
    def recoverable(self) -> bool:
        """False for regime-X cells (E, F): no placement makes them complete."""
        return self.label not in IRREDUCIBLE_CLASSES

    def __repr__(self) -> str:                     # pragma: no cover - display
        return (f"PartialClassification({self.label.value}, "
                f"f={self.inside_fraction:.3f}, cuts={self.cut_count}, "
                f"chords={len(self.chords)}, "
                f"interior_vertices={self.interior_vertex_count})")


@dataclass(frozen=True)
class OrientationCandidate:
    """One dominant wall direction the fringe voted for.

    angle     : the ABSOLUTE grid angle to try, in [0, period). This is the base
                placement's own angle plus the cluster's orientation, not the
                orientation itself -- chord angles are measured against the grid
                (see `Chord`), so at a base angle theta a chord reading phi
                belongs to a wall whose absolute orientation is theta + phi.
    weight     : accumulated vote weight of the cluster, sum of L * g(f).
    residual   : signed turn from the base angle onto this candidate, reduced
                 into (-vote_period/2, +vote_period/2]. Positive is
                 counter-clockwise, negative clockwise (design note section 8.2);
                 |residual| is the exact angle that makes this wall family
                 parallel to a grid axis.
    resultant  : the cluster's OWN concentration in [0, 1] -- how tightly its
                 members agree. A cluster of one chord has resultant 1.
    chord_count: how many oblique chords voted for it.
    """

    angle: float
    weight: float
    residual: float
    resultant: float
    chord_count: int


@dataclass(frozen=True)
class RotationVote:
    """The rotation read-off from one classified placement (design note 8.1).

    base_angle   : the grid angle of the placement the vote was taken from. All
                   chord orientations are relative to it.
    period       : the PLACEMENT period, 90 for square cells and 180 for
                   rectangular ones. Candidate angles live in [0, period).
    vote_period  : the circle the circular mean was taken on. See
                   `_ALIGNMENT_PERIOD` for why this is 90 regardless of `period`
                   by default, and how to run the note's literal m = 360/period.
    multiplier   : m = 360 / vote_period, the note's symmetry multiplier.
    phi_star     : the dominant chord orientation RELATIVE to `base_angle`, in
                   [0, vote_period).
    angle        : phi_star as an absolute grid angle, (base_angle + phi_star)
                   mod period -- the note's "the angle the walls want the grid
                   to match".
    delta        : signed residual onto `angle`, in (-vote_period/2,
                   vote_period/2]. The turn direction and magnitude of 8.2.
    resultant    : R = |sum w e^(i m phi)| / sum w, in [0, 1]. The built-in
                   should-we-rotate confidence. 0 when nothing voted.
    total_weight : sum of w = L * g(f) over every oblique chord.
    chord_count  : how many oblique chords voted.
    candidates   : the distinct dominant orientations, ranked by weight.
    evaluations  : placements evaluated by the guided search that produced this
                   vote. 0 when `_dominant_orientations` was called directly --
                   the vote itself evaluates nothing.
    """

    base_angle: float
    period: float
    vote_period: float
    multiplier: float
    phi_star: float
    angle: float
    delta: float
    resultant: float
    total_weight: float
    chord_count: int
    candidates: Tuple[OrientationCandidate, ...] = ()
    evaluations: int = 0

    @property
    def candidate_angles(self) -> Tuple[float, ...]:
        """Just the absolute angles, in rank order -- what the search sweeps."""
        return tuple(c.angle for c in self.candidates)

    def confident(self, r_min: float = R_MIN) -> bool:
        """True when the fringe has a dominant direction worth turning for."""
        return self.chord_count > 0 and self.resultant >= r_min

    def __repr__(self) -> str:                     # pragma: no cover - display
        return (f"RotationVote(phi*={self.phi_star:.2f}deg, "
                f"angle={self.angle:.2f}deg, delta={self.delta:+.2f}deg, "
                f"R={self.resultant:.3f}, chords={self.chord_count}, "
                f"candidates={[round(a, 2) for a in self.candidate_angles]})")


@dataclass(frozen=True)
class Certificate:
    """How close to optimal a placement is guaranteed to be (note section 9).

    partial            : partial cells at this placement.
    irreducible        : |X| -- regime-X cells (classes E and F). Partial under
                         EVERY placement, so a hard floor.
    boundary_length    : L, the perimeter of the usable region including holes.
    forced_length      : the part of L that no grid angle can lay along a grid
                         line, and which therefore must cross cell interiors.
                         This, not L, is what bounds the partial count -- see
                         `GridPacker.certificate`.
    best_direction     : the orientation, in [0, 90), carrying the most boundary
                         length. The angle the floor assumes a grid aligns to.
    cell_diagonal      : the longest straight boundary piece one cell can hold.
    boundary_floor     : forced_length / cell_diagonal, the covering bound.
    observed_max_chord : the most boundary any one partial cell actually holds
                         here. The boundary floor assumes this stays under the
                         diagonal; this is the measurement of that assumption.
    floor              : max(irreducible, ceil(boundary_floor)) -- the fewest
                         partials any placement of this grid on this region
                         could have.
    recoverable_area   : area still outside the region across partials worth
                         reclaiming; section 8.4's stop criterion. 0 means there
                         is no upside left to turn for.
    recover_threshold  : the inside fraction above which a partial counts toward
                         `recoverable_area`.
    """

    partial: int
    irreducible: int
    boundary_length: float
    forced_length: float
    best_direction: float
    cell_diagonal: float
    boundary_floor: float
    observed_max_chord: float
    floor: int
    recoverable_area: float
    recover_threshold: float

    @property
    def gap(self) -> int:
        """Partials above the floor. 0 certifies the placement optimal in the
        partial count; a small value bounds how much better anyone could do."""
        return self.partial - self.floor

    @property
    def certified(self) -> bool:
        """False when the boundary floor's assumption does not hold here.

        A cell holding more boundary than its own diagonal means the boundary
        wiggles inside a cell, so L / diagonal over-counts the cells needed and
        the floor -- and therefore the gap -- cannot be relied on.
        """
        return self.observed_max_chord <= self.cell_diagonal + _GEOM_TOL

    def __repr__(self) -> str:                     # pragma: no cover - display
        return (f"Certificate(partial={self.partial}, floor={self.floor}, "
                f"gap={self.gap}, irreducible={self.irreducible}, "
                f"certified={self.certified})")


@dataclass
class Placement:
    """Result of evaluating one grid placement (offset + angle)."""
    dx: float
    dy: float
    angle: float
    complete: int
    partial: int
    complete_cells: List[Polygon] = field(default_factory=list)
    partial_cells: List[Polygon] = field(default_factory=list)
    usable_area: float = 0.0
    #: Taxonomy read-off, INDEX-ALIGNED with `partial_cells`: the class of
    #: `partial_cells[i]` is `partial_classes[i]`. Empty unless `evaluate` was
    #: called with `classify=True` -- classification is opt-in because
    #: `evaluate` is the hot primitive inside the search sweeps and must not
    #: pay for it. `classified` says which of the two states this is in.
    partial_classes: List["PartialClassification"] = field(default_factory=list)
    #: The rotation read-off that produced this placement, when it came out of
    #: `optimize_guided`. None for a placement that was merely evaluated. Step 4
    #: surfaces phi*, R and the candidate weights from here into the API stats,
    #: and the paper's figures read the same object.
    rotation_vote: Optional["RotationVote"] = None

    @property
    def classified(self) -> bool:
        """True when there is a class for every partial cell.

        A placement with no partial cells reports True either way, which is the
        useful answer: there is nothing left unclassified. Where the two states
        differ -- partials present but never classified -- this is False.
        """
        return len(self.partial_classes) == self.partial

    @property
    def complete_area(self) -> float:
        return sum(c.area for c in self.complete_cells)

    @property
    def coverage(self) -> float:
        """Fraction of the usable region captured by COMPLETE cells (0..1)."""
        return self.complete_area / self.usable_area if self.usable_area else 0.0

    def __repr__(self) -> str:
        return (f"Placement(dx={self.dx:.3f}, dy={self.dy:.3f}, "
                f"angle={self.angle:.1f}, complete={self.complete}, "
                f"partial={self.partial}, coverage={self.coverage:.1%})")


def _iter_rings(geom):
    """Yield every ring of a polygonal geometry: exteriors AND interiors.

    `usable` is `shape.difference(obstacles)`, so it may be a Polygon with
    holes, a MultiPolygon (an obstacle can cut the shape in two) or, in
    degenerate cases, a GeometryCollection. Non-polygonal parts (stray lines
    or points left by the difference) carry no cells and are skipped.
    """
    gt = geom.geom_type
    if gt == "Polygon":
        if geom.is_empty:
            return
        yield geom.exterior
        for ring in geom.interiors:
            yield ring
    elif gt in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _iter_rings(part)


def _wrap(value: float, period: float) -> float:
    """Reduce `value` into [0, period), rounded to _COORD_DECIMALS decimals.

    Rounding after the modulo can push a value that sits a hair below the
    period up onto the period itself (e.g. -1e-15 % 10 -> 10.0); that is the
    same offset as 0, so fold it back.

    The rounding is what `_GEOM_TOL` compensates for in `evaluate`: it is what
    turns a wall at 0.40000000000000036 into a grid line at exactly 0.4.
    """
    v = round(value % period, _COORD_DECIMALS)
    return 0.0 if v >= period else v


def _objective(placement: "Placement", partial_penalty: float = 0.0):
    """The optimization objective of design note section 1, as a sort key.

    maximize N_complete - partial_penalty * N_partial, ties broken by fewer
    partials. Defined once at module level rather than as a closure inside each
    search so that `optimize`, `optimize_exact` and `optimize_guided` provably
    rank on the SAME objective -- which is what makes "guided is never worse
    than its own base" and "guided beats the fixed-angle scan" comparable
    claims rather than two different scoreboards.
    """
    return (placement.complete - partial_penalty * placement.partial,
            -placement.partial)


def _wrap_signed(value: float, period: float) -> float:
    """Reduce `value` into the half-open window (-period/2, +period/2].

    This is the note's `wrap` of section 8.2: the signed residual whose sign is
    the turn direction (positive = counter-clockwise) and whose magnitude is the
    turn. Taking the window half-open at -period/2 and closed at +period/2 makes
    the map single-valued at the antipode, where the two turns are equal and
    opposite and the choice is arbitrary; picking the positive one keeps the
    result a deterministic function of the input.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    v = value % period
    return v - period if v > period / 2.0 else v


def _circular_mean(items: Sequence[Tuple[float, float]], multiplier: float):
    """Weighted circular mean of orientations on the symmetry circle.

    items      : (orientation in degrees, weight) pairs.
    multiplier : m of design note section 8.1. Orientations are mapped to the
                 circle by phi -> m*phi before averaging, which is what makes
                 directions that the grid cannot tell apart average together
                 instead of cancelling.

    Returns (mean orientation in degrees, resultant length R in [0, 1], total
    weight). With no items, or with total weight 0, returns (0, 0, 0) rather
    than dividing by zero -- a fringe with nothing in it has no direction and
    zero confidence, which is exactly what the R gate should then see.
    """
    c = s = total = 0.0
    for phi, w in items:
        radians = math.radians(multiplier * phi)
        c += w * math.cos(radians)
        s += w * math.sin(radians)
        total += w
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    return (math.degrees(math.atan2(s, c)) / multiplier,
            math.hypot(c, s) / total,
            total)


def _orientation_bins(items: Sequence[Tuple[float, float]], period: float,
                      bin_deg: float):
    """Group (orientation, weight) pairs into cyclic bins of width `bin_deg`.

    Replaces the note's `votes[round(phi)]`. Two deliberate changes, both
    documented on `VOTE_BIN_DEG`: the bins are CYCLIC on [0, period) so a wall
    that scatters across the wrap point stays one wall, and they are offset by
    half a width so that a bin is CENTRED on 0 -- `round` already does this for
    width 1, and it matters because 0 (mod the alignment period) is the
    direction wall orientations pile up on.

    The bin count is rounded to an integer so the bins tile the circle exactly;
    the effective width is therefore period / round(period / bin_deg), which
    can differ slightly from the requested one.

    Yields (mean orientation, weight, resultant, count) per non-empty bin,
    ranked by weight. The orientation is the bin's weighted circular mean, not
    its centre.
    """
    if bin_deg <= 0:
        raise ValueError("bin_deg must be positive")
    n_bins = max(1, int(round(period / bin_deg)))
    width = period / n_bins
    multiplier = 360.0 / period

    buckets = {}
    for phi, w in items:
        index = int(math.floor((phi + width / 2.0) / width)) % n_bins
        buckets.setdefault(index, []).append((phi, w))

    clusters = []
    for index, members in buckets.items():
        mean, resultant, weight = _circular_mean(members, multiplier)
        clusters.append((mean % period, weight, resultant, len(members)))
    # Rank by weight; ties broken by orientation so the list is deterministic.
    clusters.sort(key=lambda c: (-c[1], c[0]))
    return clusters


def _grow(geom, tol: float = _GEOM_TOL):
    """`geom` widened by `tol` on every side, for a tolerant containment test.

    A positive buffer pushes the exterior out by `tol` and pulls interior rings
    (obstacles) IN by `tol`, which is exactly the wanted asymmetry: the outer
    wall becomes tol-forgiving while obstacles become tol-stricter, so the
    tolerance can never let a cell clip an obstacle. A mitre join keeps convex
    corners as corners rather than rounding them into arcs, so the result is
    the true tol-offset of the polygon rather than an inflated hull of it.

    A degenerate `usable` (empty, or a zero-area sliver left by a difference)
    buffers to an empty or harmless geometry rather than raising, but invalid
    input can still make GEOS fail; falling back to the un-grown geometry then
    only costs the tolerance, which is the pre-existing behaviour.
    """
    try:
        grown = geom.buffer(tol, join_style="mitre")
    except Exception:                           # pragma: no cover - GEOS guard
        return geom
    return geom if grown.is_empty else grown


def _full_width_intervals(region, prepared, x0: float, x1: float,
                          ylo: float, yhi: float,
                          tol: float = _GEOM_TOL) -> List[Tuple[float, float]]:
    """Heights t at which the segment from (x0,t) to (x1,t) lies inside `region`.

    This is the whole trick behind `optimize_columns`. A cell in the column
    [x0, x1] is complete exactly when EVERY horizontal cut of it spans the full
    column width, so the column's completeness is captured by this set of
    heights once, and the cell's height then enters as a single erosion --
    no per-cell containment test, and no dependence on dy at all.

    The set changes only at heights where the region's boundary does something,
    and there are exactly two ways it can:

      * a vertex of the region at that height, and
      * a boundary crossing of the column's OWN side lines x = x0 or x = x1.

    The second is what a search over vertex offsets alone misses. A slanted wall
    has no vertex between its ends, yet it crosses a column line at a height
    that depends on where the column happens to sit, and full-width coverage
    starts or stops precisely there. Those are the note's "diagonal critical
    lines on the torus" (section 4.1), and they are why a vertex-only offset set
    is exact for rectilinear regions but not for slanted ones.

    Between two consecutive breakpoints the answer cannot change, so one sample
    per gap decides it, and abutting gaps are merged.
    """
    heights = {ylo, yhi}
    for geom in _iter_polygons(region):
        for ring in [geom.exterior] + list(geom.interiors):
            for _, y in ring.coords:
                if ylo - tol <= y <= yhi + tol:
                    heights.add(min(max(y, ylo), yhi))

    for x in (x0, x1):
        crossings = region.boundary.intersection(
            LineString([(x, ylo), (x, yhi)]))
        for geom in getattr(crossings, "geoms", [crossings]):
            if geom.is_empty:
                continue
            points = ([(geom.x, geom.y)] if geom.geom_type == "Point"
                      else list(geom.coords))
            for _, y in points:
                if ylo - tol <= y <= yhi + tol:
                    heights.add(y)

    ordered = sorted(heights)
    intervals: List[Tuple[float, float]] = []
    for lo, hi in zip(ordered, ordered[1:]):
        if hi - lo <= tol:
            continue
        if not prepared.contains(LineString([(x0, (lo + hi) / 2.0),
                                             (x1, (lo + hi) / 2.0)])):
            continue
        if intervals and abs(intervals[-1][1] - lo) <= tol:
            intervals[-1] = (intervals[-1][0], hi)
        else:
            intervals.append((lo, hi))
    return intervals


def _erode_intervals(intervals: Sequence[Tuple[float, float]], height: float,
                     tol: float = _GEOM_TOL) -> List[Tuple[float, float]]:
    """{y : [y, y + height] lies inside one of `intervals`}.

    Turns "where does the column admit a full-width cut" into "where does a
    whole cell fit", which is the same question the containment test answers
    per cell -- asked once per interval instead.
    """
    return [(lo, hi - height) for lo, hi in intervals
            if hi - lo >= height - tol]


def _best_lattice_offset(columns: Sequence[Sequence[Tuple[float, float]]],
                         period: float, tol: float = _GEOM_TOL
                         ) -> Tuple[int, float]:
    """Exact argmax over the offset of a periodic lattice against intervals.

    Given, per column, the heights at which a cell is complete, the total count
    at offset `d` is the number of lattice points d + j*period landing in those
    intervals. As a function of d that is piecewise constant, and it can only
    change where an interval end passes a lattice point -- i.e. at the interval
    ends reduced modulo the period. Evaluating one sample per gap, plus the
    breakpoints themselves, therefore sees every attainable value.

    This is what removes the second enumeration: the dy axis is SOLVED, not
    sampled, so no resolution can step over its optimum and slanted-edge events
    are captured exactly rather than approximated by vertex offsets.
    """
    events = {0.0}
    for intervals in columns:
        for lo, hi in intervals:
            events.add(lo % period)
            events.add(hi % period)

    ordered = sorted(events)
    extended = ordered + [ordered[0] + period]
    candidates = set(ordered)
    candidates.update(((a + b) / 2.0) % period
                      for a, b in zip(extended, extended[1:]))

    best_count, best_offset = -1, 0.0
    for offset in sorted(candidates):
        count = 0
        for intervals in columns:
            for lo, hi in intervals:
                first = math.ceil((lo - offset) / period - tol)
                last = math.floor((hi - offset) / period + tol)
                if last >= first:
                    count += last - first + 1
        if count > best_count:
            best_count, best_offset = count, offset
    return best_count, best_offset


def _face_samples(vals: Sequence[float], period: float) -> List[float]:
    """One representative offset per constant-count interval.

    The critical values cut the offset circle [0, period) into arcs on which
    N_complete is constant. Evaluating the midpoint of every arc plus every
    critical value itself therefore sees every value the objective can take.

    Degenerate input (no vertices at all, or a single critical value) is
    handled: 0.0 is always part of the set, so the circle always has at least
    one arc and the result is never empty.
    """
    if period <= 0:
        raise ValueError("period must be positive")

    vals = sorted({_wrap(v, period) for v in vals} | {0.0})
    ext = vals + [vals[0] + period]
    mids = [_wrap((a + b) / 2.0, period) for a, b in zip(ext, ext[1:])]
    return sorted(set(mids) | set(vals))


# --------------------------------------------------------------------------- #
# partial-cell taxonomy: reading the shape of a clip K = C n U
# --------------------------------------------------------------------------- #
_Pt = Tuple[float, float]


def _iter_polygons(geom) -> Iterator[Polygon]:
    """Yield every non-empty Polygon component of a geometry.

    A clip C n U is usually a Polygon, but a neck or a pair of obstacles can
    split it into a MultiPolygon, and a clip that touches the region along a
    line comes back as a GeometryCollection with stray lines in it. Only the
    polygonal parts carry area, so only they carry a morphology.
    """
    gt = geom.geom_type
    if gt == "Polygon":
        if not geom.is_empty:
            yield geom
    elif gt in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _iter_polygons(part)


def _interior_ring_lines(geom):
    """The union of every INTERIOR ring of `geom`, as lines, or None.

    This is what makes class D (obstacle bite) decidable: a chord lying on one
    of these lines is an obstacle edge, a chord lying elsewhere on dU is an
    outer wall. Computed once per `evaluate` call and shared across cells --
    per-cell it would be the dominant cost of classification.

    Note the honest limit: an obstacle that TOUCHES the shape's outer boundary
    is dissolved into the exterior ring by `shape.difference(...)`, so it leaves
    no interior ring and its bite is indistinguishable from a concavity of the
    shape. That is a property of the region, not of this function -- once the
    obstacle is merged there is no geometry left that says "hole".
    """
    rings = []
    for part in _iter_polygons(geom):
        for ring in part.interiors:
            rings.append(LineString(ring.coords))
    if not rings:
        return None
    return unary_union(rings)


def _oriented_ring(ring, ccw: bool) -> List[_Pt]:
    """`ring`'s vertices, de-duplicated, unclosed, wound so material is LEFT.

    Callers pass ccw=True for an exterior ring and ccw=False for an interior
    one. With that convention the polygon's material is on the left throughout,
    so a left turn (positive cross product) at a vertex means a CONVEX corner of
    the material and a right turn means a reflex one -- for holes as well as for
    outer walls, with no special-casing at the call site.

    Points closer together than `_GEOM_TOL` are collapsed: GEOS overlays can
    emit a duplicate vertex where the cut meets a cell edge, and a zero-length
    segment has no orientation to report.
    """
    coords = list(ring.coords)
    out: List[_Pt] = []
    for x, y in coords:
        if not out or math.hypot(x - out[-1][0], y - out[-1][1]) > _GEOM_TOL:
            out.append((float(x), float(y)))
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0],
                                   out[0][1] - out[-1][1]) <= _GEOM_TOL:
        out.pop()
    if ring.is_ccw != ccw:
        out.reverse()
    return out


def _segment_on_cell_edge(p: _Pt, q: _Pt, bounds, tol: float = _GEOM_TOL) -> bool:
    """True when the segment p->q lies along ONE edge of the cell.

    This is the test the whole taxonomy rests on. A piece of dK is either a cell
    edge -- where the grid, not the region, cut the clip, carrying no
    information about the boundary -- or a chord. Both endpoints must sit on the
    SAME cell-edge line: a segment whose two ends happen to touch two DIFFERENT
    cell edges (the corner-clipping cut of a class-F sliver, for instance) runs
    through the cell's interior and is a genuine chord.

    `tol` is `_GEOM_TOL`, used here as what it is elsewhere in this module: the
    coordinate resolution at which a point and a grid line are the same place.
    """
    minx, miny, maxx, maxy = bounds
    return (
        (abs(p[0] - minx) <= tol and abs(q[0] - minx) <= tol)
        or (abs(p[0] - maxx) <= tol and abs(q[0] - maxx) <= tol)
        or (abs(p[1] - miny) <= tol and abs(q[1] - miny) <= tol)
        or (abs(p[1] - maxy) <= tol and abs(q[1] - maxy) <= tol)
    )


def _strictly_inside_cell(p: _Pt, bounds, tol: float = _GEOM_TOL) -> bool:
    """True when `p` is further than `tol` from every cell edge.

    The "is this vertex on the cell edge or interior to the cell" decision of
    the note's procedure. A vertex on a cell edge is where the boundary entered
    or left the cell -- an artefact of the grid. A vertex strictly inside is a
    genuine region vertex, and is what puts the cell in class C.
    """
    minx, miny, maxx, maxy = bounds
    x, y = p
    return (x - minx > tol and maxx - x > tol
            and y - miny > tol and maxy - y > tol)


def _perp_distance(p: _Pt, a: _Pt, b: _Pt) -> float:
    """Distance from `p` to the line through `a` and `b`."""
    ux, uy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(ux, uy)
    if norm == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(ux * (p[1] - a[1]) - uy * (p[0] - a[0])) / norm


def _simplify_polyline(pts: List[_Pt], closed: bool,
                       tol: float = _GEOM_TOL) -> List[_Pt]:
    """Drop vertices that lie (within `tol`) on the line through their neighbours.

    A region wall crossing a cell can arrive as several collinear segments --
    the polygon may simply have a vertex there, and clipping adds one wherever
    the wall meets a cell edge. Those are not corners: counting them would
    inflate the chord count, invent interior "vertices" that do not turn, and
    push a trapezoid (4 corners) into the pentagon bucket. Straightening first
    makes "one chord per straight run" and "one corner per turn" true by
    construction.

    `closed` marks a ring (pts[0] == pts[-1], every vertex is a candidate);
    otherwise the two endpoints are where the cut met the cell and are kept.
    """
    if closed:
        ring = list(pts[:-1])
        changed = True
        while changed and len(ring) > 3:
            changed = False
            for i in range(len(ring)):
                a, p, b = ring[i - 1], ring[i], ring[(i + 1) % len(ring)]
                if _perp_distance(p, a, b) <= tol:
                    del ring[i]
                    changed = True
                    break
        return ring + [ring[0]]

    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        if _perp_distance(pts[i], out[-1], pts[i + 1]) > tol:
            out.append(pts[i])
    out.append(pts[-1])
    return out


def _boundary_runs(coords: List[_Pt], on_edge: List[bool]):
    """Split a ring into its maximal runs of NON-cell-edge segments (the cuts).

    Yields (points, closed) per run. `coords` is the unclosed ring and
    `on_edge[i]` describes the segment coords[i] -> coords[i+1] (cyclically).

    A run is one connected piece of dU crossing the cell -- the note's "boundary
    cut". A ring made entirely of region boundary (an obstacle wholly inside the
    cell) is a single CLOSED cut; that is why `closed` is reported rather than
    inferred.
    """
    n = len(coords)
    if n < 3:
        return
    if not any(on_edge):
        # no segment touches a cell edge -> the whole ring is one closed cut
        yield coords + [coords[0]], True
        return
    if all(on_edge):
        # the clip's boundary is entirely grid lines -> no chords at all
        return
    for start in range(n):
        if on_edge[start] or not on_edge[start - 1]:
            continue                            # not the first segment of a run
        pts = [coords[start]]
        i = start
        while not on_edge[i]:
            pts.append(coords[(i + 1) % n])
            i = (i + 1) % n
        yield pts, False


def _cross(a: _Pt, p: _Pt, b: _Pt) -> float:
    """Signed turn at `p` walking a -> p -> b. Positive = left turn."""
    return ((p[0] - a[0]) * (b[1] - p[1]) - (p[1] - a[1]) * (b[0] - p[0]))


def _make_chord(p: _Pt, q: _Pt, cut_index: int, inside_fraction: float,
                hole_lines, on_hole_ring: bool) -> Chord:
    """Build a `Chord` from one straight piece of region boundary.

    Orientation is reduced modulo 180 because a chord is an undirected segment:
    the ring winding decides which way p -> q points, and that is bookkeeping,
    not geometry. `atan2` returns (-180, 180], so `% 180.0` lands in [0, 180)
    with a leftward horizontal (180) folding onto 0 exactly.

    Axis-alignment is tested on the TRANSVERSE deviation rather than on an angle
    threshold, which lets it reuse `_GEOM_TOL` at face value: a chord is
    horizontal when its two ends differ in y by less than the coordinate
    resolution, and that is precisely the condition under which the grid line
    and the wall are the same line to the resolution this module works at. An
    angular tolerance would instead need a new constant, and would call a long
    nearly-horizontal wall "aligned" while rejecting a short exactly-horizontal
    one.
    """
    dx, dy = q[0] - p[0], q[1] - p[1]
    on_hole = on_hole_ring
    if not on_hole and hole_lines is not None:
        mid = Point((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
        on_hole = hole_lines.distance(mid) <= _GEOM_TOL
    return Chord(
        length=math.hypot(dx, dy),
        angle_deg=math.degrees(math.atan2(dy, dx)) % 180.0,
        axis_aligned=abs(dy) <= _GEOM_TOL or abs(dx) <= _GEOM_TOL,
        inside_fraction=inside_fraction,
        on_hole=on_hole,
        cut_index=cut_index,
        geometry=LineString([p, q]),
    )


class GridPacker:
    def __init__(
        self,
        shape: Polygon,
        obstacles: Optional[Sequence[Polygon]] = None,
        cell_width: float = 1.0,
        cell_height: float = 1.0,
    ):
        if not isinstance(shape, Polygon):
            raise TypeError("shape must be a shapely Polygon")
        if cell_width <= 0 or cell_height <= 0:
            raise ValueError("cell dimensions must be positive")

        self.shape = shape
        self.obstacles = list(obstacles) if obstacles else []
        self.cw = float(cell_width)
        self.ch = float(cell_height)

        # usable region = shape with the obstacles punched out
        if self.obstacles:
            self.usable = shape.difference(unary_union(self.obstacles))
        else:
            self.usable = shape

        # An empty usable region has NaN bounds, which would blow up cell
        # generation with an opaque "cannot convert float NaN to integer".
        if self.usable.is_empty:
            raise ValueError(
                "no usable area left: the obstacles cover the entire shape")

        # pivot used for any rotation so results are reproducible
        self._pivot = self.shape.centroid

        #: How many times `evaluate` has run on this packer. `evaluate` is the
        #: unit of work every search here is built from, so counting it is the
        #: one cost metric that compares a uniform sweep, an exact solve and a
        #: guided solve on equal terms -- section 11's "evaluations" column,
        #: measured rather than inferred from a loop's bounds. Free to keep, and
        #: callers may zero it between runs.
        self.evaluations = 0

    @classmethod
    def from_image(
        cls,
        image,
        *,
        cell_width: float = 1.0,
        cell_height: float = 1.0,
        simplify_tol: float = 2.0,
        min_area: float = 64.0,
        scale: float = 1.0,
    ) -> "GridPacker":
        """Build a GridPacker from an image (file path or numpy array).

        The outer boundary detected in the image becomes the shape; enclosed
        interior regions become obstacles. Coordinates are in pixels (y-up)
        unless `scale` (units per pixel) is given. See
        image_boundary.polygons_from_image for the detection parameters.
        """
        from image_boundary import polygons_from_image

        shape, obstacles = polygons_from_image(
            image, simplify_tol=simplify_tol, min_area=min_area, scale=scale)
        return cls(shape, obstacles,
                   cell_width=cell_width, cell_height=cell_height)

    # ------------------------------------------------------------------ #
    # cell generation + classification
    # ------------------------------------------------------------------ #
    def _generate_cells(self, region: Polygon, dx: float, dy: float) -> List[Polygon]:
        """Tile `region`'s bounding box with cells whose grid lines are shifted
        by (dx, dy). Only cells whose bbox touches the region are kept."""
        minx, miny, maxx, maxy = region.bounds
        w, h = self.cw, self.ch

        # snap the first grid line so that the lattice passes through the offset
        x0 = math.floor((minx - dx) / w) * w + dx
        y0 = math.floor((miny - dy) / h) * h + dy

        cells = []
        x = x0
        while x < maxx:
            y = y0
            while y < maxy:
                cells.append(box(x, y, x + w, y + h))
                y += h
            x += w
        return cells

    def evaluate(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle: float = 0.0,
        *,
        classify: bool = False,
    ) -> Placement:
        """Classify every cell for one placement.

        Rotation trick: instead of rotating the grid, we rotate the usable
        region by -angle, run an axis-aligned analysis, then rotate the
        resulting cells back by +angle so they line up with the original shape.

        Containment is tested against the usable region grown by `_GEOM_TOL`
        (see the constant): the offsets fed in here were rounded to that same
        resolution, so a cell may miss a wall it is meant to be flush with by a
        few ulp. The complete/partial split is the only thing that tolerance
        touches; the partial/outside split below runs on the UN-grown region.
        Both prepared geometries are built once per call -- `evaluate` is the
        hot primitive, and this must not become per-cell work.

        The partial/outside split is by AREA, per the definition: partial is
        0 < area(C n U) < area(C), outside is area(C n U) = 0. `intersects` is
        only the cheap necessary condition -- it is also true for a measure-zero
        touch along a line, which is exactly what the cells `_generate_cells`
        emits past a rounded-up bound do (see `_AREA_TOL_REL`). The intersection
        is computed ONLY for cells that already failed `contains` and passed
        `intersects`, i.e. the boundary fringe, O(perimeter / cell size) cells
        and not all N; the complete-cell path stays a single prepared predicate.
        Cost is therefore O(N) predicates + O(perimeter / cell size) overlays,
        and the second term vanishes against the first as the grid refines.

        `classify` is OPT-IN and off by default. `evaluate` is the primitive the
        offset search calls once per face of the critical arrangement -- hundreds
        to thousands of times per solve -- and its cost is what the paper reports
        as wall-clock. The taxonomy is a read-back wanted on ONE chosen
        placement, not on every candidate, so it must never run inside a sweep.
        When it is on, the classes land in `Placement.partial_classes`,
        index-aligned with `partial_cells`.
        """
        self.evaluations += 1

        if angle:
            work = rotate(self.usable, -angle, origin=self._pivot)
        else:
            work = self.usable

        prepared = prep(work)               # exact: fringe candidate filter
        tolerant = prep(_grow(work))        # tol-forgiving: complete vs partial
        cells = self._generate_cells(work, dx, dy)

        # Pass 1, over all N cells: prepared predicates only. `intersects` is
        # the cheap NECESSARY condition for partial, not the definition, so what
        # it selects is the boundary fringe -- the candidates.
        complete, fringe = [], []
        for c in cells:
            if tolerant.contains(c):        # wholly inside usable -> complete
                complete.append(c)
            elif prepared.intersects(c):    # meets the region somehow -> decide
                fringe.append(c)
            # else: provably disjoint -> outside, ignore

        # Pass 2, over the fringe only (O(perimeter / cell size) cells): apply
        # the definition, partial <=> 0 < area(C n U). This is what separates a
        # cell that straddles the boundary from one that merely touches it along
        # a line -- `intersects` says True to both. One batched GEOS overlay
        # rather than a Python-level round trip per cell.
        min_overlap = _AREA_TOL_REL * self.cw * self.ch
        if fringe:
            # The clips K = C n U are kept, not just their areas: they ARE the
            # objects the taxonomy classifies, and recomputing the overlays for
            # the classifier would double the only expensive step of this call.
            clips = shapely.intersection(work, fringe)
            overlaps = shapely.area(clips)
        else:
            clips, overlaps = (), ()
        partial, partial_clips = [], []
        for c, k, a in zip(fringe, clips, overlaps):
            if a > min_overlap:
                partial.append(c)
                partial_clips.append(k)

        # The taxonomy is read HERE, in the analysis frame, while the cells are
        # still axis-aligned boxes and chord orientations are therefore measured
        # against the GRID. Below this point the cells are rotated back into the
        # display frame and that information is gone.
        partial_classes: List[PartialClassification] = []
        if classify and partial:
            hole_lines = _interior_ring_lines(work)
            partial_classes = [
                self._classify_partial(c, k, work, hole_lines=hole_lines)
                for c, k in zip(partial, partial_clips)
            ]

        if angle:  # rotate cells back into the original frame for display
            complete = [rotate(c, angle, origin=self._pivot) for c in complete]
            partial = [rotate(c, angle, origin=self._pivot) for c in partial]

        return Placement(
            dx=dx, dy=dy, angle=angle,
            complete=len(complete), partial=len(partial),
            complete_cells=complete, partial_cells=partial,
            usable_area=self.usable.area,
            partial_classes=partial_classes,
        )

    # ------------------------------------------------------------------ #
    # partial-cell taxonomy (design note sections 6-7)
    # ------------------------------------------------------------------ #
    def _classify_partial(
        self,
        cell: Polygon,
        clip: Polygon,
        work,
        *,
        hole_lines=None,
        sliver_fraction: float = _SLIVER_FRACTION,
    ) -> PartialClassification:
        """Classify one partial cell into the A-F taxonomy from its clip.

        cell  : the grid cell C, an axis-aligned box IN THE ANALYSIS FRAME.
        clip  : K = C n U, already computed by `evaluate`'s area test.
        work  : the usable region in the same frame as `cell` and `clip`.

        FRAME. Every angle this returns is measured against `cell`'s own axes.
        `evaluate` runs its analysis on the region rotated by -angle and only
        rotates the cells back afterwards for display, so cells are axis-aligned
        boxes here and "axis-aligned" means "aligned with the GRID" -- which is
        what the taxonomy means by it, and what the rotation vote of section 8
        needs its orientations phi to be measured against. Classifying the
        display-frame cells would measure every chord against the screen instead
        and make the vote meaningless. Callers must pass analysis-frame
        geometry; there is no way to detect a display-frame cell after the fact.

        hole_lines      : `_interior_ring_lines(work)`, hoisted out by `evaluate`
                          so it is computed once per placement rather than once
                          per cell. Computed here when omitted.
        sliver_fraction : the f below which a cell is class F. See
                          `_SLIVER_FRACTION` -- this is a modelling threshold and
                          is exposed for the paper's sensitivity study.

        The decision procedure, in order (the note gives the tests but not their
        precedence; the order below is the one that makes them mutually
        exclusive, and each step says why it sits where it does):

          1. f <= sliver_fraction               -> F. Checked FIRST because F is
             a statement that the cell is beyond rescue, which overrides
             whatever shape the crumb happens to have (a grazing sliver is
             always geometrically a B3 triangle or a C1 wedge).
          2. cut_count >= 2, or K disconnected  -> E. The note's rule verbatim.
          3. any chord on an obstacle ring      -> D.
          4. a region vertex inside the cell    -> C2 if any is reflex, else C1.
          5. the single chord is axis-aligned   -> A.
          6. otherwise                          -> B, split by clip corners.
        """
        cell_area = cell.area
        f = clip.area / cell_area if cell_area > 0 else 0.0
        bounds = cell.bounds

        if hole_lines is None:
            hole_lines = _interior_ring_lines(work)

        parts = list(_iter_polygons(clip))
        disconnected = len(parts) > 1

        chords: List[Chord] = []
        cut_count = 0
        n_convex = n_reflex = 0

        for part in parts:
            rings = [(part.exterior, True)] + [(r, False) for r in part.interiors]
            for ring, is_exterior in rings:
                coords = _oriented_ring(ring, ccw=is_exterior)
                n = len(coords)
                if n < 3:
                    continue
                on_edge = [
                    _segment_on_cell_edge(coords[i], coords[(i + 1) % n], bounds)
                    for i in range(n)
                ]
                for run, closed in _boundary_runs(coords, on_edge):
                    pts = _simplify_polyline(run, closed)
                    if len(pts) < 2:
                        continue
                    cut_index = cut_count
                    cut_count += 1

                    # Region vertices are the JUNCTIONS of a cut: a cut's two
                    # endpoints sit on cell edges by construction (that is where
                    # the boundary entered and left), so only the points between
                    # them can be interior. A closed cut has no endpoints, so
                    # every one of its vertices is a candidate.
                    junctions = range(len(pts) - 1) if closed else range(1, len(pts) - 1)
                    for j in junctions:
                        p = pts[j]
                        if not _strictly_inside_cell(p, bounds):
                            continue
                        a = pts[j - 1] if j > 0 else pts[-2]
                        turn = _cross(a, p, pts[j + 1])
                        # `_oriented_ring` put the material on the left, so a
                        # left turn is a convex corner of the region and a right
                        # turn a reflex one -- on obstacle rings too.
                        if turn > 0:
                            n_convex += 1
                        elif turn < 0:
                            n_reflex += 1

                    for i in range(len(pts) - 1):
                        chords.append(_make_chord(
                            pts[i], pts[i + 1], cut_index, f,
                            hole_lines, on_hole_ring=not is_exterior))

        # Corner count of the largest component: the B1 / B2 / B3 discriminator.
        # A straight cut through a box leaves a triangle (3), a trapezoid (4) or
        # a pentagon (5), so the count is exactly the note's three shapes.
        clip_vertex_count = 0
        if parts:
            main = max(parts, key=lambda g: g.area)
            main_ring = _oriented_ring(main.exterior, ccw=True)
            if len(main_ring) >= 3:
                clip_vertex_count = len(
                    _simplify_polyline(main_ring + [main_ring[0]], closed=True)) - 1

        on_hole = any(c.on_hole for c in chords)
        n_interior = n_convex + n_reflex
        label = self._partial_label(
            f=f, chords=chords, cut_count=cut_count,
            clip_vertex_count=clip_vertex_count, n_convex=n_convex,
            n_reflex=n_reflex, on_hole=on_hole, disconnected=disconnected,
            sliver_fraction=sliver_fraction)

        return PartialClassification(
            label=label,
            inside_fraction=f,
            chords=tuple(chords),
            cut_count=cut_count,
            clip_vertex_count=clip_vertex_count,
            interior_vertex_count=n_interior,
            interior_convex_count=n_convex,
            interior_reflex_count=n_reflex,
            on_hole=on_hole,
            disconnected=disconnected,
        )

    @staticmethod
    def _partial_label(*, f, chords, cut_count, clip_vertex_count, n_convex,
                       n_reflex, on_hole, disconnected,
                       sliver_fraction=_SLIVER_FRACTION) -> PartialClass:
        """The dispatch of `_classify_partial`, split out so it can be tested
        against measurement tuples without building geometry."""
        if f <= sliver_fraction:
            return PartialClass.F
        if disconnected or cut_count >= 2:
            return PartialClass.E
        if on_hole:
            return PartialClass.D
        if n_reflex:
            return PartialClass.C2
        if n_convex:
            return PartialClass.C1
        if not chords:
            # dK is entirely grid lines. Geometrically this means K is the cell,
            # i.e. the cell is complete; it can only be reached when a wall lies
            # exactly ON a cell edge and rounding left f a hair under 1. The
            # boundary is already flush with a grid axis, which is class A.
            return PartialClass.A
        if all(c.axis_aligned for c in chords):
            return PartialClass.A
        # One oblique cut. The clip's corner count is the note's B split; it is
        # exhaustive for a straight cut of a box (3 / 4 / 5), so the f fallback
        # below only fires on degenerate clips the note does not describe.
        if clip_vertex_count == 3:
            return PartialClass.B3
        if clip_vertex_count == 4:
            return PartialClass.B2
        if clip_vertex_count == 5:
            return PartialClass.B1
        return PartialClass.B1 if f >= 0.5 else PartialClass.B3

    # ------------------------------------------------------------------ #
    # search for the best placement
    # ------------------------------------------------------------------ #
    def _vertex_offsets(self) -> Tuple[List[float], List[float]]:
        """Offsets that snap grid lines onto the shape's / obstacles' own edges.
        These alignment points are where the optimum almost always sits, and a
        uniform sweep can step right over them, so we test them explicitly."""
        geoms = [self.shape.exterior] + [ob.exterior for ob in self.obstacles]
        xs, ys = set(), set()
        for ring in geoms:
            for x, y in ring.coords:
                xs.add(round(x % self.cw, _COORD_DECIMALS))
                ys.add(round(y % self.ch, _COORD_DECIMALS))
        return sorted(xs), sorted(ys)

    def optimize(
        self,
        steps: int = 12,
        angles: Sequence[float] = (0.0,),
        partial_penalty: float = 0.0,
        snap_to_edges: bool = True,
    ) -> Tuple[Placement, List[Placement]]:
        """Sweep grid offsets (and angles) and return the best placement.

        steps           : offsets tested per axis within one cell period.
                          The grid is periodic, so we only scan dx in [0, cw),
                          dy in [0, ch). Higher = finer search, slower.
        angles          : rotations to try, in degrees. Use e.g. range(0, 90, 15)
                          to let the grid rotate as well as slide.
        partial_penalty : objective is  complete - partial_penalty * partial.
                          0 -> maximize complete, break ties by fewer partials.
        snap_to_edges   : also test offsets that align grid lines with shape and
                          obstacle edges (where the optimum usually lives). Keep
                          this on; a plain uniform sweep can miss sharp optima.

        Returns (best_placement, all_placements_sorted_best_first).
        """
        dxs = list(np.linspace(0, self.cw, steps, endpoint=False))
        dys = list(np.linspace(0, self.ch, steps, endpoint=False))

        if snap_to_edges:
            vx, vy = self._vertex_offsets()
            dxs = sorted(set(dxs) | set(vx))
            dys = sorted(set(dys) | set(vy))

        results: List[Placement] = []
        for angle in angles:
            for dx in dxs:
                for dy in dys:
                    results.append(self.evaluate(dx, dy, angle))

        results.sort(key=lambda p: _objective(p, partial_penalty), reverse=True)
        return results[0], results

    # ------------------------------------------------------------------ #
    # exact search (Method 1): critical offsets instead of a uniform sweep
    # ------------------------------------------------------------------ #
    def _critical_offsets(self, angle: float = 0.0) -> Tuple[List[float], List[float]]:
        """The offsets at which the complete-cell count can possibly change.

        Sliding the grid only flips a cell between complete and partial when a
        grid line touches a vertex of the usable region, so N_complete is a step
        function whose jumps sit exactly at dx = xv (mod cw), dy = yv (mod ch)
        for every vertex (xv, yv) of the usable region. That makes the set below
        exhaustive, not a sample.

        Unlike `_vertex_offsets`, this reads the vertices off `self.usable`
        (which already has the obstacles punched out) and walks BOTH exterior
        and interior rings of every polygon in it -- `usable` is a MultiPolygon
        whenever an obstacle cuts the shape into disjoint pieces.

        `angle` matters: `evaluate()` implements rotation by rotating the usable
        region by -angle and then running an axis-aligned analysis, so the
        critical offsets for a placement at `angle` are those of the ROTATED
        geometry. Passing the un-rotated vertices would give offsets that have
        nothing to do with the grid lines actually used.
        """
        work = rotate(self.usable, -angle, origin=self._pivot) if angle else self.usable

        xs, ys = set(), set()
        for ring in _iter_rings(work):
            for x, y in ring.coords:
                xs.add(_wrap(x, self.cw))
                ys.add(_wrap(y, self.ch))
        return sorted(xs), sorted(ys)

    def optimize_exact(
        self,
        angles: Sequence[float] = (0.0,),
        partial_penalty: float = 0.0,
    ) -> Tuple[Placement, List[Placement]]:
        """Exact translation search: evaluate one offset per constant-count face.

        Same objective and same return contract as `optimize()`, but instead of
        a resolution-dependent uniform sweep it enumerates the arrangement of
        critical lines (see `_critical_offsets`) and tests one point inside each
        face. Cost drops from O(steps^2) to O(Vx * Vy) evaluations per angle,
        and no sharp optimum can be stepped over.

        angles          : rotations to try, in degrees. Each angle gets its own
                          critical set, computed from the geometry as that angle
                          sees it.
        partial_penalty : objective is  complete - partial_penalty * partial.

        Returns (best_placement, all_placements_sorted_best_first).

        Exactness note: the critical set is derived from region vertices, which
        covers every event for axis-aligned edges. A slanted edge can also flip
        a cell when a lattice CORNER grazes it, which is a diagonal event line
        in (dx, dy) and is not a vertex offset. The guarantee is therefore exact
        for rectilinear regions (and for any region viewed at an angle that
        makes it rectilinear); elsewhere it remains a strong superset of the
        alignment offsets an optimum needs.
        """
        angles = list(angles)
        if not angles:
            raise ValueError("angles must contain at least one angle")

        results: List[Placement] = []
        for angle in angles:
            vx, vy = self._critical_offsets(angle)
            dxs = _face_samples(vx, self.cw)
            dys = _face_samples(vy, self.ch)
            for dx in dxs:
                for dy in dys:
                    results.append(self.evaluate(dx, dy, angle))

        results.sort(key=lambda p: _objective(p, partial_penalty), reverse=True)
        return results[0], results

    def optimize_columns(
        self,
        angles: Sequence[float] = (0.0,),
        partial_penalty: float = 0.0,
    ) -> Tuple[Placement, List[Placement]]:
        """Translation search that SOLVES the dy axis instead of enumerating it.

        `optimize_exact` removed the resolution from the offset search but kept
        its shape: a double loop over Vx * Vy faces, each one a full
        reclassification of all N cells. That is still enumeration, and it is
        quadratic in the number of distinct vertex coordinates -- a 256-gon disc
        costs 64,517 evaluations where a room costs 4, and each of those pays
        the O(N) cell scan again.

        This removes the inner loop entirely. Cut the region into the columns
        the grid makes at a given dx. For one column, a cell is complete exactly
        when every horizontal cut of it spans the full column width, so
        `_full_width_intervals` gives the heights at which the column admits a
        cell, `_erode_intervals` accounts for the cell's own height, and the
        completeness of that column is then a fixed set of intervals with no
        dependence on dy at all. Summing over columns,

            N_complete(dx, dy) = # { (i, j) : dy + j*ch  lands in  Y_i }

        which is a lattice of period ch tested against a fixed interval set --
        a 1-D problem that `_best_lattice_offset` solves exactly, in one sweep
        of the interval ends. Cost per dx falls from Vy * O(N) predicates to
        O(columns * boundary complexity), and the whole search costs Vx
        evaluations rather than Vx * Vy.

        MORE EXACT, NOT LESS. Solving dy as a continuum also fixes a real gap in
        `optimize_exact`: a slanted edge flips a cell where a lattice corner
        grazes it, an event that is not any vertex's offset, so a vertex-derived
        dy set can miss the optimum outright. On a 36x27 room tilted 23 degrees
        at 3x3 cells, `optimize_exact` returns 83 complete cells and this
        returns 84 -- confirmed attainable by a 120x120 dense sweep, which also
        finds 84. The dx axis is still sampled from the critical offsets, so the
        same caveat now applies to one axis instead of two; what is claimed here
        is that this dominates `optimize_exact` everywhere and is exact whenever
        that is, never the reverse.

        Same objective and same return contract as `optimize` and
        `optimize_exact`: (best_placement, all_placements_sorted_best_first),
        one placement per dx.
        """
        angles = list(angles)
        if not angles:
            raise ValueError("angles must contain at least one angle")

        results: List[Placement] = []
        for angle in angles:
            work = (rotate(self.usable, -angle, origin=self._pivot)
                    if angle else self.usable)
            prepared = prep(_grow(work))
            minx, miny, maxx, maxy = work.bounds

            vx, _ = self._critical_offsets(angle)
            for dx in _face_samples(vx, self.cw):
                # Walk the columns this dx puts over the region. Adjacent
                # columns share a side line, so the crossings of each line are
                # computed once and handed to both of its columns.
                columns: List[List[Tuple[float, float]]] = []
                x = minx - ((minx - dx) % self.cw)
                while x < maxx - _GEOM_TOL:
                    intervals = _full_width_intervals(
                        work, prepared, x, x + self.cw, miny, maxy)
                    columns.append(_erode_intervals(intervals, self.ch))
                    x += self.cw

                _, dy = _best_lattice_offset(columns, self.ch)
                # The count is re-derived by the ordinary evaluation rather
                # than trusted from the sweep: the sweep chooses the placement,
                # `evaluate` remains the single definition of what a placement
                # scores, so the two can never drift apart.
                results.append(self.evaluate(dx % self.cw, dy % self.ch, angle))

        results.sort(key=lambda p: _objective(p, partial_penalty), reverse=True)
        return results[0], results

    # ------------------------------------------------------------------ #
    # Method 2: rotation read off the partial-cell pattern (design note 8)
    # ------------------------------------------------------------------ #
    @property
    def rotation_period(self) -> float:
        """The period of the placement's angle: 90 for square cells, 180 else.

        Turning a SQUARE cell by 90 degrees maps the grid onto itself, so only
        [0, 90) is distinct. Turning a rectangular cell by 90 degrees swaps its
        width and height, which is a different tiling, so its angle ranges over
        [0, 180). This is the note's P. It is NOT the circle the orientation
        vote averages on -- see `_ALIGNMENT_PERIOD`.
        """
        return 90.0 if self.cw == self.ch else 180.0

    def _dominant_orientations(
        self,
        placement: Placement,
        *,
        vote_period: Optional[float] = None,
        bin_deg: float = VOTE_BIN_DEG,
        max_candidates: int = MAX_CANDIDATES,
        weight_power: float = 1.0,
        include_irreducible: bool = True,
    ) -> RotationVote:
        """The rotation vote: let the oblique fringe pick the angle (note 8.1).

        Every oblique chord of every partial cell casts a vote for its own
        orientation phi, weighted w = L * g(f) with L the chord length and
        g(f) = f**weight_power the inside fraction of the cell it came from.
        g up-weights near-complete B1 cells (much to reclaim) and starves
        hopeless B3 triangles and F slivers (nothing to reclaim). The votes are
        averaged on the symmetry circle, giving the dominant orientation phi*
        and a concentration R that says whether turning is worth it at all.

        placement must have been evaluated with `classify=True`; the vote reads
        `partial_classes`, and a placement whose partials were never classified
        would silently vote with an empty fringe.

        WHICH CHORDS VOTE. Every oblique chord, whatever class its cell was
        labelled. The note's pseudocode comments "B-class chords only", but its
        own section 7.1 puts "oblique cut (B1-B3, oblique C2/D)" in the rotation
        bucket -- the lever is dispatched by the CUT's orientation, not by the
        cell's label, and a reflex corner or an obstacle bite with a slanted
        edge wants the same turn a B2 trapezoid does. `oblique_chords` is
        exactly that set. The label-blind reading is also the one that degrades
        gracefully: on a tilted room most cells classify as B2 but the corner
        cells come out C1/C2, and dropping their chords would bias phi* toward
        whichever wall happened to avoid the corners.

        `include_irreducible=False` drops chords from regime-X cells (E, F) as
        an ablation. It is off by default because the weight already handles
        them: an F sliver has f ~ 0 by definition, so w ~ 0, and it cannot move
        the mean whether it is present or not.

        vote_period : the symmetry circle, default `_ALIGNMENT_PERIOD` (90).
                      Pass `self.rotation_period` to reproduce the note's
                      literal m = 360/P.
        bin_deg     : orientation clustering resolution, see `VOTE_BIN_DEG`.
        max_candidates : how many CLUSTERS become candidates, see
                      `MAX_CANDIDATES`. With rectangular cells each cluster
                      emits two angles, phi and phi + 90.
        """
        if not placement.classified:
            raise ValueError(
                "the rotation vote needs the partial-cell taxonomy: evaluate "
                "the placement with classify=True before voting")

        period = self.rotation_period
        vote_period = _ALIGNMENT_PERIOD if vote_period is None else float(vote_period)
        if vote_period <= 0:
            raise ValueError("vote_period must be positive")
        multiplier = 360.0 / vote_period
        base_angle = placement.angle

        votes: List[Tuple[float, float]] = []
        for classification in placement.partial_classes:
            if not include_irreducible and not classification.recoverable:
                continue
            for chord in classification.oblique_chords:
                weight = chord.length * (chord.inside_fraction ** weight_power)
                if weight > 0.0:
                    votes.append((chord.angle_deg % vote_period, weight))

        phi_star, resultant, total_weight = _circular_mean(votes, multiplier)
        phi_star %= vote_period

        # A cluster's orientation is relative to the GRID, so the grid angle
        # that aligns it is the base angle plus the orientation. The note takes
        # phi* to be the absolute angle, which is only true because its caller
        # happens to vote from theta = 0.
        def to_angle(orientation: float) -> float:
            return (base_angle + orientation) % period

        candidates: List[OrientationCandidate] = []
        if votes:
            # One cluster can name several distinct PLACEMENTS when the vote
            # circle is finer than the placement period: with rectangular cells
            # a wall at phi is flush with the grid both at grid angle phi (the
            # wall runs along the cell width) and at phi + 90 (along the
            # height), and those are different tilings that must both be tried.
            repeats = max(1, int(round(period / vote_period)))
            for mean, weight, spread, count in _orientation_bins(
                    votes, vote_period, bin_deg)[:max(0, max_candidates)]:
                for k in range(repeats):
                    orientation = mean + k * vote_period
                    candidates.append(OrientationCandidate(
                        angle=to_angle(orientation),
                        weight=weight,
                        residual=_wrap_signed(orientation, vote_period),
                        resultant=spread,
                        chord_count=count,
                    ))

        return RotationVote(
            base_angle=base_angle,
            period=period,
            vote_period=vote_period,
            multiplier=multiplier,
            phi_star=phi_star,
            angle=to_angle(phi_star),
            delta=_wrap_signed(phi_star, vote_period),
            resultant=resultant,
            total_weight=total_weight,
            chord_count=len(votes),
            candidates=tuple(candidates),
        )

    def optimize_guided(
        self,
        *,
        partial_penalty: float = 0.0,
        r_min: float = R_MIN,
        refine: str = REFINE_METHOD,
        refine_half_window: float = REFINE_HALF_WINDOW,
        refine_probes: int = REFINE_PROBES,
        recover_min: float = RECOVERABLE_AREA_MIN,
        max_candidates: int = MAX_CANDIDATES,
        bin_deg: float = VOTE_BIN_DEG,
        vote_period: Optional[float] = None,
        weight_power: float = 1.0,
    ) -> Tuple[Placement, List[Placement]]:
        """Align-then-solve-exactly (design note section 8.4).

        Solve translation exactly at theta = 0, classify that placement, let its
        oblique fringe vote for the angle, and solve translation exactly again
        at each angle the vote named. No angle scan happens anywhere.

        NEVER WORSE THAN THE BASE, BY CONSTRUCTION. The un-rotated base is kept
        in the result pool and the winner is the argmax over the pool under
        `_objective` -- the same objective every other search here ranks on. A
        vote that misfires therefore costs evaluations and nothing else. Exact
        ties go to the smallest turn, so a candidate has to be strictly better
        to move the grid.

        Returns (best_placement, all_placements_sorted_best_first), matching
        `optimize` and `optimize_exact`. The vote is attached to the returned
        placement as `.rotation_vote`, carrying phi*, R, the candidates with
        their weights, and the evaluation count.

        r_min          : vote-confidence gate, see `R_MIN`. Below it the fringe
                         has no dominant direction and the base is returned
                         unrotated, without spending a candidate solve.
        refine         : local refine, see `REFINE_METHOD` -- "grid" (default),
                         "golden" (the note's prescription) or "none".
        refine_probes  : solves the refine may spend; shared budget, whichever
                         method is chosen, so the two are comparable.
        recover_min    : section 8.4's stop, in cell areas -- skip the refine
                         once the leading placement's fringe holds less
                         reclaimable area than this. 0 disables it. See
                         `RECOVERABLE_AREA_MIN` for why it is read here and not
                         at the entrance.
        """
        if refine not in ("none", "grid", "golden"):
            raise ValueError(f"unknown refine method: {refine!r}")

        base_best, results = self.optimize_exact(
            angles=(0.0,), partial_penalty=partial_penalty)
        # Re-evaluate the winning offset WITH the taxonomy. Classification is
        # opt-in precisely so the sweep above does not pay for it, so the one
        # placement the vote reads is classified here, on its own.
        base = self.evaluate(base_best.dx, base_best.dy, 0.0, classify=True)
        results = [base] + list(results[1:])
        evaluations = len(results) + 1          # + the classifying re-evaluate

        vote = self._dominant_orientations(
            base,
            vote_period=vote_period,
            bin_deg=bin_deg,
            max_candidates=max_candidates,
            weight_power=weight_power,
        )

        def attach(best: Placement) -> Tuple[Placement, List[Placement]]:
            best.rotation_vote = replace(vote, evaluations=evaluations)
            return best, results

        if not vote.confident(r_min):
            # No dominant wall -> nothing to align to. Note section 8.1.
            results.sort(key=lambda p: _objective(p, partial_penalty), reverse=True)
            return attach(base)

        def solve(angle: float) -> Placement:
            nonlocal evaluations
            best, batch = self.optimize_exact(
                angles=(angle,), partial_penalty=partial_penalty)
            results.extend(batch)
            evaluations += len(batch)
            return best

        for angle in vote.candidate_angles:
            solve(angle)

        if refine != "none" and refine_probes > 0 and refine_half_window > 0:
            # Refine around the best angle found so far rather than around each
            # candidate. The note says "each candidate"; refining only the
            # argmax costs 1/k as many solves and, on every instance measured,
            # reached the same count -- and the refine turns out not to earn its
            # place at all (see `REFINE_METHOD`), so spending k times more on it
            # would be worse than pointless.
            leader = max(results, key=lambda p: _objective(p, partial_penalty))
            centre = leader.angle

            worth_it = True
            if recover_min > 0:
                # Section 8.4's stop: if the fringe at the leading placement has
                # nothing left to reclaim, no probe in the window can find
                # anything, so one classifying evaluation decides against
                # `refine_probes` solves. See `RECOVERABLE_AREA_MIN`.
                classified = self.evaluate(leader.dx, leader.dy, centre,
                                           classify=True)
                evaluations += 1
                worth_it = (self._recoverable_area(classified)
                            >= recover_min * self.cw * self.ch)

            if worth_it and refine == "golden":
                self._golden_refine(centre, refine_half_window, refine_probes,
                                    solve, partial_penalty)
            elif worth_it:
                self._grid_refine(centre, refine_half_window, refine_probes, solve)

        # Rank on the objective; break exact ties toward the SMALLEST turn from
        # the base, so the grid only moves when moving is strictly better and a
        # tie leaves the un-rotated placement in front.
        results.sort(
            key=lambda p: (_objective(p, partial_penalty),
                           -abs(_wrap_signed(p.angle - base.angle, vote.period))),
            reverse=True)
        return attach(results[0])

    @staticmethod
    def _grid_refine(centre: float, half_window: float, probes: int, solve):
        """Uniformly sample the +/- window around `centre`.

        `probes` angles at the centres of `probes` equal sub-intervals, so none
        of them repeats `centre` (already solved) and the set is symmetric about
        it. Unlike golden-section this makes no unimodality assumption, which
        matters because N_complete(theta) is a piecewise-constant step function:
        a plateau tells a bracketing search nothing, but sampling still sees
        every plateau wide enough to be worth landing on.
        """
        step = 2.0 * half_window / probes
        for i in range(probes):
            solve(centre - half_window + (i + 0.5) * step)

    @staticmethod
    def _golden_refine(centre: float, half_window: float, probes: int, solve,
                       partial_penalty: float):
        """Golden-section search of the +/- window (the note's prescription).

        KEPT FOR THE ABLATION, NOT BECAUSE IT WORKS. Golden-section contracts a
        bracket by comparing two interior probes, which requires the objective
        to be unimodal and to actually vary between them. N_complete(theta) is
        integer-valued and piecewise-constant, so most comparisons inside a few
        degrees are between two EQUAL values; the tie is broken by the code's
        arbitrary convention and the bracket contracts on no information,
        converging to an arbitrary interior point. See `REFINE_METHOD` for what
        that measures out as.
        """
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        lo, hi = centre - half_window, centre + half_window
        a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        fa = _objective(solve(a), partial_penalty)
        fb = _objective(solve(b), partial_penalty)
        for _ in range(max(0, probes - 2)):
            if fa < fb:
                lo, a, fa = a, b, fb
                b = lo + ratio * (hi - lo)
                fb = _objective(solve(b), partial_penalty)
            else:
                hi, b, fb = b, a, fa
                a = hi - ratio * (hi - lo)
                fa = _objective(solve(a), partial_penalty)

    # ------------------------------------------------------------------ #
    # per-instance optimality certificate (design note section 9)
    # ------------------------------------------------------------------ #
    def certificate(
        self,
        placement: Placement,
        *,
        recover_threshold: float = RECOVERABLE_FRACTION_MIN,
    ) -> "Certificate":
        """How far from optimal this placement could possibly be.

        The taxonomy splits the partials into three optimality regimes (note
        section 9). Regime X -- classes E and F, features smaller than a cell --
        cannot be recovered by ANY placement, so they are not a failure but a
        certificate: they bound how well anyone could do on this instance.

        Two independent lower bounds on the partial count are combined.

        |X|, the regime-X cells. Those cells are partial under every placement.

        The boundary-covering floor. Section 9.1 argues that "the region
        boundary of length L must be covered by partial cells", giving a floor
        of L / (c * s). As written that is false, and it fails on the simplest
        instance there is: a 12x9 room at 3x3 cells tiles exactly into 12
        complete cells and ZERO partials, while L / diagonal claims a floor of
        10. Boundary that lies ALONG grid lines is covered by the edges of
        complete cells; only boundary forced through a cell's INTERIOR makes a
        cell partial.

        So the length that counts is not L. Group the boundary by edge
        orientation modulo 90 degrees: at grid angle theta, edges oriented
        theta (mod 90) can be made to lie on grid lines, and every other edge
        must cross cell interiors. Choosing theta to be the most common
        orientation minimises what is forced, so

            forced = L - (total length of the single best-aligned direction)

        and the floor is forced / (the most boundary one cell can hold). That
        denominator the note leaves as "a small constant c" has an exact value:
        a straight piece of boundary crossing an axis-aligned cell is a chord of
        that cell, and the longest chord of a cw x ch box is its diagonal.

        This bound is placement-independent -- it holds for every (dx, dy,
        theta) -- and it is deliberately conservative in two ways, both of which
        can only make it smaller and therefore still valid: it assumes the best
        direction can be aligned perfectly, and it ignores that an aligned wall
        must additionally land ON a grid line rather than merely parallel to one.

        It rests on one assumption: that each cell carries a single straight
        crossing. A boundary that wiggles inside one cell can hold more than a
        diagonal's worth, which would make the floor too high and the reported
        gap too flattering. So the assumption is MEASURED rather than asserted
        -- `observed_max_chord` is the most boundary any one partial cell
        actually holds at this placement, and `certified` is False when it
        exceeds the diagonal. A caller reading the gap can see whether it holds.

        `placement` must have been evaluated with `classify=True`.
        """
        if not placement.classified:
            raise ValueError(
                "the certificate needs the partial-cell taxonomy: evaluate the "
                "placement with classify=True first")

        diagonal = math.hypot(self.cw, self.ch)
        boundary_length = self.usable.length

        irreducible = sum(1 for c in placement.partial_classes
                          if not c.recoverable)

        recoverable_area = self._recoverable_area(placement, recover_threshold)

        observed_max_chord = max(
            (c.chord_length for c in placement.partial_classes), default=0.0)

        forced_length, best_direction = self._forced_boundary_length()
        boundary_floor = forced_length / diagonal if diagonal else 0.0
        floor = max(irreducible, int(math.ceil(boundary_floor - _GEOM_TOL)))

        return Certificate(
            partial=placement.partial,
            irreducible=irreducible,
            boundary_length=boundary_length,
            forced_length=forced_length,
            best_direction=best_direction,
            boundary_floor=boundary_floor,
            cell_diagonal=diagonal,
            observed_max_chord=observed_max_chord,
            floor=floor,
            recoverable_area=recoverable_area,
            recover_threshold=recover_threshold,
        )

    def _recoverable_area(
        self,
        placement: Placement,
        fraction_min: float = RECOVERABLE_FRACTION_MIN,
    ) -> float:
        """How much area is still on the table at this placement.

        Section 8.4's "sum over f > tau of (1 - f) * area(C)": for every partial
        worth reclaiming, the slice of it currently OUTSIDE the region. Cells
        below `fraction_min` are mostly outside and are not worth turning the
        whole grid for -- see `RECOVERABLE_FRACTION_MIN`.

        Read twice, for two different questions: `certificate` reports it, and
        `optimize_guided` stops on it (see `RECOVERABLE_AREA_MIN`). One
        definition serves both so the number the paper prints is the same number
        the solver acted on.
        """
        cell_area = self.cw * self.ch
        return sum((1.0 - c.inside_fraction) * cell_area
                   for c in placement.partial_classes
                   if c.recoverable and c.inside_fraction > fraction_min)

    def _forced_boundary_length(
        self,
        *,
        bin_deg: float = VOTE_BIN_DEG,
    ) -> Tuple[float, float]:
        """Boundary that no grid angle can lay along a grid line.

        Returns (forced_length, best_direction). Edges are grouped by
        orientation modulo the alignment period; the direction carrying the most
        length is the one a grid should align to, and everything else is forced
        through cell interiors. See `certificate` for why this, and not the full
        perimeter, is what bounds the partial count.

        `bin_deg` is the tolerance within which two edges count as parallel,
        because a boundary traced from an image and decimated by
        Douglas-Peucker scatters a straight wall's orientation by a degree or
        two. A generous tolerance can only move length INTO the aligned group
        and so only shrinks the floor, which is the safe direction for a lower
        bound.

        The candidate directions are the edge orientations themselves rather
        than fixed bins. Fixed bins put an arbitrary boundary somewhere, and a
        wall that lands on one is split in half: a rectangle tilted 23 degrees
        has its two edge families at 23 and 113, which are the same direction
        modulo 90, but 2-degree bins round them to opposite sides of a bin edge
        and report half the perimeter as forced -- on a room that tiles exactly.
        """
        edges: List[Tuple[float, float]] = []       # (orientation, length)
        total = 0.0
        for ring in _iter_rings(self.usable):
            coords = list(ring.coords)
            for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
                length = math.hypot(x1 - x0, y1 - y0)
                if length <= _GEOM_TOL:
                    continue
                total += length
                edges.append((math.degrees(math.atan2(y1 - y0, x1 - x0))
                              % _ALIGNMENT_PERIOD, length))

        if not edges:
            return 0.0, 0.0

        def apart(a: float, b: float) -> float:
            """Circular separation of two orientations on the alignment circle."""
            d = abs(a - b) % _ALIGNMENT_PERIOD
            return min(d, _ALIGNMENT_PERIOD - d)

        best_aligned, best_direction = 0.0, edges[0][0]
        for candidate, _ in edges:
            members = [(o, L) for o, L in edges if apart(o, candidate) <= bin_deg]
            aligned = sum(L for _, L in members)
            if aligned > best_aligned:
                # Report the length-weighted mean of the group, not the
                # candidate edge, so a scattered wall reports its centre.
                offset = sum(L * _wrap_signed(o - candidate, _ALIGNMENT_PERIOD)
                             for o, L in members) / aligned
                best_aligned = aligned
                best_direction = (candidate + offset) % _ALIGNMENT_PERIOD
                if best_aligned >= total - _GEOM_TOL:
                    break                           # everything is one family

        return max(0.0, total - best_aligned), best_direction

    # ------------------------------------------------------------------ #
    # visualization
    # ------------------------------------------------------------------ #
    def plot(self, placement: Placement, ax=None, title: Optional[str] = None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 7))

        def draw(poly, **kw):
            ax.add_patch(MplPolygon(np.array(poly.exterior.coords), **kw))

        # shape outline
        draw(self.shape, closed=True, fill=False, edgecolor="black", lw=2, zorder=1)

        # obstacles
        for ob in self.obstacles:
            draw(ob, closed=True, facecolor="#444", edgecolor="black",
                 alpha=0.85, zorder=4)

        # cells
        for c in placement.partial_cells:
            draw(c, closed=True, facecolor="#f4a259", edgecolor="#b56a1e",
                 alpha=0.55, lw=0.6, zorder=2)
        for c in placement.complete_cells:
            draw(c, closed=True, facecolor="#5cb85c", edgecolor="#2f6f2f",
                 alpha=0.75, lw=0.6, zorder=3)

        minx, miny, maxx, maxy = self.shape.bounds
        pad = max(maxx - minx, maxy - miny) * 0.05
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal")
        ax.grid(True, ls=":", alpha=0.3)

        if title is None:
            title = (f"complete={placement.complete}  partial={placement.partial}  "
                     f"coverage={placement.coverage:.1%}  "
                     f"(dx={placement.dx:.2f}, dy={placement.dy:.2f}, "
                     f"angle={placement.angle:.0f}°)")
        ax.set_title(title)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#5cb85c", alpha=0.75, label="complete"),
            Patch(facecolor="#f4a259", alpha=0.55, label="partial"),
            Patch(facecolor="#444", label="obstacle"),
        ], loc="upper right", fontsize=9)
        return ax
