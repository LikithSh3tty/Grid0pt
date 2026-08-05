"""Tests for the partial-cell shape taxonomy (design note sections 6-7).

A partial cell is defined by its clip K = C n U. The shape of K encodes how the
region boundary crosses the cell and therefore which grid move, if any, recovers
the cell. This module pins that read-off down:

  * one hand-built instance per class A, B1, B2, B3, C1, C2, D, E, F, asserting
    the label AND the measurements the label was derived from;
  * chord extraction -- the piece of dK on dU is a chord, the piece on a cell
    edge is not. That distinction is what the taxonomy rests on, so it is tested
    on its own and not only through the labels;
  * orientation -- reduced to [0, 180) because a chord is undirected, and
    measured against the GRID rather than the display frame. The latter is the
    property the rotation vote of section 8 consumes, and it is only true
    because classification runs before `evaluate` rotates its cells back;
  * classification is opt-in and never runs inside the search.
"""
import math

import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import LineString, Polygon, box

from grid_packer import (
    _GEOM_TOL,
    _SLIVER_FRACTION,
    GridPacker,
    PartialClass,
    _segment_on_cell_edge,
)


# --------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------- #
#: The unit cell every hand-built instance is classified against. Placing it at
#: the origin means every coordinate below reads directly as a fraction of the
#: cell, so the intended morphology is legible from the polygon literal.
CELL = box(0.0, 0.0, 1.0, 1.0)


def classify(shape, obstacles=None, cell=CELL):
    """Classify `cell` against `shape - obstacles`, on the true clip K = C n U.

    Calls the classifier the way `evaluate` does -- with an axis-aligned cell,
    its clip, and the region, all in one frame -- but with the geometry written
    out by hand so the expected morphology is unambiguous.
    """
    packer = GridPacker(shape, obstacles, cell_width=1.0, cell_height=1.0)
    clip = packer.usable.intersection(cell)
    assert 0.0 < clip.area < cell.area, "instance is not a PARTIAL cell"
    return packer._classify_partial(cell, clip, packer.usable)


def angles(result):
    return sorted(round(c.angle_deg, 6) for c in result.chords)


# --------------------------------------------------------------------- #
# A -- axis-aligned slab: one cut parallel to a grid axis, K a rectangle
# --------------------------------------------------------------------- #
def test_class_A_axis_aligned_slab():
    """A horizontal wall at y = 0.6 cuts the cell into a rectangle."""
    result = classify(box(-1, -1, 2, 0.6))

    assert result.label == PartialClass.A
    assert result.inside_fraction == pytest.approx(0.6)
    assert result.cut_count == 1
    assert len(result.chords) == 1
    assert result.clip_vertex_count == 4              # K is a rectangle
    assert result.interior_vertex_count == 0

    (chord,) = result.chords
    assert chord.axis_aligned
    assert chord.angle_deg == pytest.approx(0.0)
    assert chord.length == pytest.approx(1.0)         # the wall spans the cell


def test_class_A_vertical_wall():
    """The other grid axis: a vertical wall is equally an A slab, at 90 deg."""
    result = classify(box(-1, -1, 0.25, 2))

    assert result.label == PartialClass.A
    assert result.inside_fraction == pytest.approx(0.25)
    (chord,) = result.chords
    assert chord.axis_aligned
    assert chord.angle_deg == pytest.approx(90.0)


# --------------------------------------------------------------------- #
# B -- one oblique cut: pentagon / trapezoid / triangle
# --------------------------------------------------------------------- #
def test_class_B1_pentagon_corner_sliced_off():
    """A cut from (0.7, 1) to (1, 0.6) slices the top-right corner off.

    The removed triangle has area 0.5 * 0.3 * 0.4 = 0.06, so f = 0.94 -- the
    note's "f ~ 1" pentagon. The cut direction (0.3, -0.4) is atan2(-0.4, 0.3)
    = -53.13 deg, i.e. 126.87 deg once reduced to [0, 180).
    """
    shape = Polygon([(-1, -1), (2, -1), (2, 0.6), (1, 0.6),
                     (0.7, 1), (0.7, 2), (-1, 2)])
    result = classify(shape)

    assert result.label == PartialClass.B1
    assert result.inside_fraction == pytest.approx(0.94)
    assert result.inside_fraction > 0.9               # the note's f ~ 1
    assert result.cut_count == 1
    assert len(result.chords) == 1
    assert result.clip_vertex_count == 5              # pentagon
    assert result.interior_vertex_count == 0

    (chord,) = result.chords
    assert not chord.axis_aligned
    assert chord.angle_deg == pytest.approx(math.degrees(math.atan2(-0.4, 0.3)) % 180.0)
    assert chord.angle_deg == pytest.approx(126.8698976)
    assert chord.length == pytest.approx(0.5)         # 3-4-5 triangle


def test_class_B2_trapezoid_at_30_degrees():
    """A 30 deg cut entering the left cell edge and leaving the right one.

    y = 0.3 + tan(30) x meets x = 0 at y = 0.3 and x = 1 at y = 0.8774, both
    strictly inside the cell, so the cut crosses OPPOSITE edges and both pieces
    are trapezoids. f is the mean of the two parallel sides.
    """
    slope = math.tan(math.radians(30.0))
    shape = Polygon([(-1, -1), (2, -1), (2, 0.3 + 2 * slope), (-1, 0.3 - slope)])
    result = classify(shape)

    assert result.label == PartialClass.B2
    assert result.inside_fraction == pytest.approx((0.3 + 0.3 + slope) / 2.0)
    assert result.cut_count == 1
    assert len(result.chords) == 1
    assert result.clip_vertex_count == 4              # trapezoid
    assert result.interior_vertex_count == 0

    (chord,) = result.chords
    assert not chord.axis_aligned
    # The measurement the rotation vote consumes: a 30 deg cut reports 30 deg.
    assert chord.angle_deg == pytest.approx(30.0)
    assert chord.length == pytest.approx(1.0 / math.cos(math.radians(30.0)))


def test_class_B3_triangle_only_a_corner_kept():
    """A cut from (0.3, 0) to (0, 0.4) keeps only the bottom-left corner.

    f = 0.5 * 0.3 * 0.4 = 0.06 -- small, but two orders of magnitude above the
    sliver threshold, so this is a B3 and not an F.
    """
    shape = Polygon([(-1, -1), (0.3, -1), (0.3, 0), (0, 0.4), (-1, 0.4)])
    result = classify(shape)

    assert result.label == PartialClass.B3
    assert result.inside_fraction == pytest.approx(0.06)
    assert result.inside_fraction > _SLIVER_FRACTION
    assert result.cut_count == 1
    assert result.clip_vertex_count == 3              # triangle
    assert result.interior_vertex_count == 0

    (chord,) = result.chords
    assert not chord.axis_aligned
    assert chord.angle_deg == pytest.approx(math.degrees(math.atan2(0.4, -0.3)) % 180.0)
    assert chord.length == pytest.approx(0.5)


def test_B_family_is_split_by_clip_corner_count():
    """B1 / B2 / B3 differ only in how many corners the clip keeps."""
    slope = math.tan(math.radians(30.0))
    b1 = classify(Polygon([(-1, -1), (2, -1), (2, 0.6), (1, 0.6),
                           (0.7, 1), (0.7, 2), (-1, 2)]))
    b2 = classify(Polygon([(-1, -1), (2, -1), (2, 0.3 + 2 * slope), (-1, 0.3 - slope)]))
    b3 = classify(Polygon([(-1, -1), (0.3, -1), (0.3, 0), (0, 0.4), (-1, 0.4)]))

    assert (b1.clip_vertex_count, b2.clip_vertex_count, b3.clip_vertex_count) == (5, 4, 3)
    assert b1.inside_fraction > b2.inside_fraction > b3.inside_fraction
    # Every B is one oblique cut with no region vertex inside the cell.
    for result in (b1, b2, b3):
        assert result.cut_count == 1
        assert len(result.chords) == 1
        assert not result.chords[0].axis_aligned
        assert result.interior_vertex_count == 0


# --------------------------------------------------------------------- #
# C -- a region vertex lies interior to the cell
# --------------------------------------------------------------------- #
def test_class_C1_convex_vertex_is_a_wedge():
    """A triangular spike pokes its apex (0.5, 0.4) into the cell.

    The boundary turns INSIDE the cell at a convex corner, so the clip is a
    wedge with f small. The two chords meet at that corner and therefore belong
    to ONE cut -- if they counted as two cuts this would be misread as class E.
    """
    result = classify(Polygon([(0.2, -0.5), (0.8, -0.5), (0.5, 0.4)]))

    assert result.label == PartialClass.C1
    assert result.interior_vertex_count == 1
    assert result.interior_convex_count == 1
    assert result.interior_reflex_count == 0
    assert result.cut_count == 1                      # one cut, two chords
    assert len(result.chords) == 2
    assert result.inside_fraction < 0.1               # the note's "f small"
    assert result.inside_fraction == pytest.approx(0.5 * (0.6 * 0.4 / 0.9) * 0.4)

    # The apex is strictly interior to the cell, which is what makes this a C.
    apex_hits = [c for c in result.chords
                 if c.geometry.distance(LineString([(0.5, 0.4), (0.5, 0.4001)])) < 1e-6]
    assert apex_hits, "both chords should end at the interior apex"


def test_class_C2_reflex_vertex_is_an_L_shape():
    """A concave corner of the region sits at (0.5, 0.5), inside the cell.

    The clip is the cell minus its top-right quadrant: an L with f = 0.75. Both
    chords are axis-aligned, which is exactly the note's "axis-aligned chords ->
    A / C / D-aligned" case -- alignment alone does not make it an A slab.
    """
    shape = Polygon([(-1, -1), (2, -1), (2, 0.5), (0.5, 0.5), (0.5, 2), (-1, 2)])
    result = classify(shape)

    assert result.label == PartialClass.C2
    assert result.inside_fraction == pytest.approx(0.75)
    assert result.inside_fraction > 0.5               # the note's "f large"
    assert result.interior_vertex_count == 1
    assert result.interior_reflex_count == 1
    assert result.interior_convex_count == 0
    assert result.cut_count == 1
    assert len(result.chords) == 2
    assert all(c.axis_aligned for c in result.chords)
    assert angles(result) == [0.0, 90.0]
    assert result.chord_length == pytest.approx(1.0)  # two half-cell walls


def test_C_split_is_by_corner_convexity_not_by_f():
    """C1 vs C2 is decided by the sign of the turn at the interior vertex."""
    convex = classify(Polygon([(0.2, -0.5), (0.8, -0.5), (0.5, 0.4)]))
    reflex = classify(Polygon([(-1, -1), (2, -1), (2, 0.5),
                               (0.5, 0.5), (0.5, 2), (-1, 2)]))

    assert convex.label == PartialClass.C1 and convex.interior_convex_count == 1
    assert reflex.label == PartialClass.C2 and reflex.interior_reflex_count == 1


# --------------------------------------------------------------------- #
# D -- an interior hole intrudes: K = cell minus a bite
# --------------------------------------------------------------------- #
def test_class_D_obstacle_wholly_inside_the_cell():
    """An obstacle strictly inside the cell: K = cell - bite, one closed cut.

    dK here is two rings: the cell's own four edges (no chords at all) and the
    obstacle ring, every segment of which is region boundary. That ring is a
    single CLOSED cut, so cut_count stays 1 and the cell is a D, not an E.
    """
    result = classify(box(-1, -1, 2, 2), [box(0.3, 0.3, 0.7, 0.7)])

    assert result.label == PartialClass.D
    assert result.on_hole
    assert result.inside_fraction == pytest.approx(1.0 - 0.16)
    assert result.cut_count == 1
    assert len(result.chords) == 4                    # the four obstacle edges
    assert result.chord_length == pytest.approx(1.6)
    assert all(c.axis_aligned and c.on_hole for c in result.chords)
    # The obstacle's corners are region vertices inside the cell, and they are
    # reflex with respect to the material that surrounds the hole.
    assert result.interior_vertex_count == 4
    assert result.interior_reflex_count == 4


def test_class_D_obstacle_crossing_a_cell_edge():
    """The bite need not be wholly inside: an obstacle entering through the
    bottom edge still leaves chords on the hole ring."""
    result = classify(box(-1, -1, 2, 2), [box(0.3, -0.2, 0.7, 0.6)])

    assert result.label == PartialClass.D
    assert result.on_hole
    assert result.inside_fraction == pytest.approx(1.0 - 0.4 * 0.6)
    assert result.cut_count == 1
    assert len(result.chords) == 3                    # up, across, back down
    assert all(c.on_hole for c in result.chords)


def test_class_D_is_distinguished_from_C2_by_the_ring_it_cuts():
    """A hole ring and a concave wall can produce the same LOCAL shape; what
    separates D from C is which ring of U the chords lie on."""
    obstacle_cell = classify(box(-1, -1, 2, 2), [box(0.5, 0.5, 1.5, 1.5)])
    wall_cell = classify(Polygon([(-1, -1), (2, -1), (2, 0.5),
                                  (0.5, 0.5), (0.5, 2), (-1, 2)]))

    assert obstacle_cell.inside_fraction == pytest.approx(wall_cell.inside_fraction)
    assert obstacle_cell.label == PartialClass.D and obstacle_cell.on_hole
    assert wall_cell.label == PartialClass.C2 and not wall_cell.on_hole


# --------------------------------------------------------------------- #
# E -- sub-cell feature / neck: two or more cuts
# --------------------------------------------------------------------- #
def test_class_E_band_through_the_cell():
    """A strip of U 0.4 tall crosses a cell of height 1: a neck.

    Both long sides of the strip are region boundary, so dK carries TWO separate
    cuts -- no single translation flushes both, which is what makes E
    irreducible at this cell size.
    """
    result = classify(box(-1, 0.3, 2, 0.7))

    assert result.label == PartialClass.E
    assert result.inside_fraction == pytest.approx(0.4)
    assert result.cut_count == 2
    assert len(result.chords) == 2
    assert not result.disconnected
    assert angles(result) == [0.0, 0.0]               # a band, both sides flat
    assert not result.recoverable                     # regime X


def test_class_E_disconnected_clip():
    """An obstacle slicing the cell in two leaves K with two components."""
    result = classify(box(-1, -1, 2, 2), [box(-1, 0.3, 2, 0.7)])

    assert result.label == PartialClass.E
    assert result.disconnected
    assert result.cut_count == 2
    assert result.inside_fraction == pytest.approx(0.6)
    assert not result.recoverable


# --------------------------------------------------------------------- #
# F -- grazing sliver: f ~ 0, the cell is essentially outside
# --------------------------------------------------------------------- #
def test_class_F_grazing_sliver():
    """The boundary clips the (0, 0) corner with legs 0.03 and 0.04.

    f = 6e-4 -- below the sliver threshold, so the cell is reported as
    essentially outside even though its clip is geometrically a B3 triangle.
    """
    shape = Polygon([(-1, -1), (0.03, -1), (0.03, 0), (0, 0.04), (-1, 0.04)])
    result = classify(shape)

    assert result.label == PartialClass.F
    assert result.inside_fraction == pytest.approx(0.5 * 0.03 * 0.04)
    assert result.inside_fraction < _SLIVER_FRACTION
    assert result.cut_count == 1
    assert result.clip_vertex_count == 3              # shaped like a B3 ...
    assert not result.recoverable                     # ... but unrecoverable


def test_F_and_B3_differ_only_by_scale():
    """The same corner cut, scaled up past the threshold, becomes a B3."""
    sliver = classify(Polygon([(-1, -1), (0.03, -1), (0.03, 0), (0, 0.04), (-1, 0.04)]))
    triangle = classify(Polygon([(-1, -1), (0.3, -1), (0.3, 0), (0, 0.4), (-1, 0.4)]))

    assert sliver.label == PartialClass.F
    assert triangle.label == PartialClass.B3
    assert angles(sliver) == angles(triangle)         # identical morphology
    assert sliver.inside_fraction < _SLIVER_FRACTION < triangle.inside_fraction


def test_sliver_threshold_is_a_tunable_modelling_choice():
    """The F/B3 boundary is a reported parameter, not a hidden constant."""
    packer = GridPacker(Polygon([(-1, -1), (0.3, -1), (0.3, 0), (0, 0.4), (-1, 0.4)]),
                        cell_width=1.0, cell_height=1.0)
    clip = packer.usable.intersection(CELL)

    strict = packer._classify_partial(CELL, clip, packer.usable)
    loose = packer._classify_partial(CELL, clip, packer.usable, sliver_fraction=0.1)

    assert strict.label == PartialClass.B3
    assert loose.label == PartialClass.F


# --------------------------------------------------------------------- #
# chord extraction: dU boundary is a chord, a cell edge is not
# --------------------------------------------------------------------- #
def test_chord_on_the_region_boundary_is_returned():
    """The wall at y = 0.6 is dU inside the cell, and is reported as a chord."""
    result = classify(box(-1, -1, 2, 0.6))
    (chord,) = result.chords

    assert chord.geometry.equals(LineString([(0.0, 0.6), (1.0, 0.6)]))
    assert chord.length == pytest.approx(1.0)


def test_clip_edges_lying_on_cell_edges_are_not_chords():
    """K's boundary has four sides here; three are grid lines and only one is dU.

    The clip box(0, 0, 1, 0.6) is bounded below by y = 0, left by x = 0 and
    right by x = 1 -- all cell edges, where the GRID cut the clip, carrying no
    information about the region. Reporting them would turn every partial cell
    into a class E with four "cuts" and destroy the taxonomy.
    """
    result = classify(box(-1, -1, 2, 0.6))
    clip = box(0.0, 0.0, 1.0, 0.6)

    assert len(clip.exterior.coords) - 1 == 4         # K has four sides ...
    assert len(result.chords) == 1                    # ... exactly one is a chord
    assert result.cut_count == 1

    for cell_edge in (LineString([(0, 0), (1, 0)]),       # bottom
                      LineString([(0, 0), (0, 1)]),       # left
                      LineString([(1, 0), (1, 1)])):      # right
        for chord in result.chords:
            assert not chord.geometry.equals(cell_edge)
            # not even a sub-segment of one
            assert not cell_edge.buffer(_GEOM_TOL).contains(chord.geometry)


def test_L_shaped_clip_reports_only_its_two_region_walls():
    """Six sides, four on cell edges, two chords -- and their lengths are the
    lengths of the walls, not of the cell edges."""
    result = classify(Polygon([(-1, -1), (2, -1), (2, 0.5),
                               (0.5, 0.5), (0.5, 2), (-1, 2)]))

    assert len(result.chords) == 2
    assert sorted(round(c.length, 9) for c in result.chords) == [0.5, 0.5]
    assert result.chord_length == pytest.approx(1.0)   # not 3.0 (the whole dK)


def test_segment_on_cell_edge_predicate():
    """The predicate the chord split is built on, tested directly.

    A segment counts as a cell edge only when BOTH ends lie on the SAME edge
    line. A cut whose two ends happen to touch two DIFFERENT cell edges runs
    through the cell's interior and is a genuine chord -- that is every corner
    cut in the taxonomy (B3, F), so getting this wrong would erase two classes.
    """
    bounds = (0.0, 0.0, 1.0, 1.0)

    assert _segment_on_cell_edge((0.0, 0.0), (1.0, 0.0), bounds)      # bottom
    assert _segment_on_cell_edge((0.0, 0.2), (0.0, 0.9), bounds)      # left, partial
    assert _segment_on_cell_edge((1.0, 1.0), (1.0, 0.4), bounds)      # right
    assert _segment_on_cell_edge((0.3, 1.0), (0.9, 1.0), bounds)      # top

    # both endpoints on cell edges, but on two DIFFERENT ones -> a chord
    assert not _segment_on_cell_edge((0.03, 0.0), (0.0, 0.04), bounds)
    # a wall crossing the cell interior -> a chord
    assert not _segment_on_cell_edge((0.0, 0.6), (1.0, 0.6), bounds)
    # one end on an edge, one end inside (a cut ending at a region vertex)
    assert not _segment_on_cell_edge((0.0, 0.5), (0.5, 0.5), bounds)
    # within tolerance of an edge line still counts as that edge
    assert _segment_on_cell_edge((0.0, 0.2), (_GEOM_TOL / 2, 0.9), bounds)


# --------------------------------------------------------------------- #
# orientation: undirected, in [0, 180), and measured against the GRID
# --------------------------------------------------------------------- #
def test_orientation_is_reduced_to_half_open_180():
    """A chord has no direction: the same wall reports the same angle whichever
    side of it the clip lies on, and never 180 or more."""
    slope = math.tan(math.radians(30.0))
    line = [(-1, 0.3 - slope), (2, 0.3 + 2 * slope)]
    below = classify(Polygon([(-1, -1), (2, -1), line[1], line[0]]))
    above = classify(Polygon([line[0], line[1], (2, 3), (-1, 3)]))

    assert below.chords[0].angle_deg == pytest.approx(30.0)
    assert above.chords[0].angle_deg == pytest.approx(30.0)
    # the two clips are complementary, so the winding traverses the SAME wall in
    # opposite directions -- and still lands on the same orientation
    assert below.inside_fraction + above.inside_fraction == pytest.approx(1.0)

    # a horizontal wall traversed right-to-left folds onto 0, not 180
    assert classify(box(-1, 0.4, 2, 2)).chords[0].angle_deg == pytest.approx(0.0)

    for shape in (box(-1, -1, 2, 0.6), box(-1, -1, 0.25, 2),
                  Polygon([(-1, -1), (2, -1), (2, 0.6), (1, 0.6),
                           (0.7, 1), (0.7, 2), (-1, 2)])):
        for chord in classify(shape).chords:
            assert 0.0 <= chord.angle_deg < 180.0


def test_orientation_is_measured_against_the_grid_not_the_display_frame():
    """The frame test: the SAME instance, seen by two grid angles.

    A 6x4 rectangle is tilted 20 deg. `evaluate` analyses the region rotated by
    -angle, so at angle = 20 the grid sees a rectangle whose walls are parallel
    to its own axes -- every chord must report 0 or 90 and every wall cell must
    be an A slab. At angle = 0 the same walls are oblique and must report 20 and
    110. If classification ran on the display-frame cells (after `evaluate`
    rotates them back) both placements would report 20 / 110, the taxonomy would
    call an aligned grid oblique, and the rotation vote of section 8 would be
    driven by the screen instead of by the grid.
    """
    tilted = shp_rotate(box(0, 0, 6, 4), 20.0, origin="centroid")
    packer = GridPacker(tilted, cell_width=1.0, cell_height=1.0)

    aligned = packer.evaluate(0.37, 0.21, 20.0, classify=True)
    oblique = packer.evaluate(0.37, 0.21, 0.0, classify=True)

    assert aligned.partial > 0 and oblique.partial > 0

    aligned_angles = {round(c.angle_deg, 6)
                      for pc in aligned.partial_classes for c in pc.chords}
    oblique_angles = {round(c.angle_deg, 6)
                      for pc in oblique.partial_classes for c in pc.chords}

    assert aligned_angles == {0.0, 90.0}, "grid-aligned walls must read as 0/90"
    assert oblique_angles == {20.0, 110.0}, "tilted walls must read as 20/110"

    # ... and the labels follow the orientation, which is the point.
    assert all(c.axis_aligned for pc in aligned.partial_classes for c in pc.chords)
    assert not any(c.axis_aligned for pc in oblique.partial_classes for c in pc.chords)
    assert PartialClass.A in {pc.label for pc in aligned.partial_classes}
    assert not any(pc.label == PartialClass.A for pc in oblique.partial_classes)
    assert {pc.label for pc in oblique.partial_classes} & {
        PartialClass.B1, PartialClass.B2, PartialClass.B3}


def test_classification_is_index_aligned_with_partial_cells():
    """`partial_classes[i]` describes `partial_cells[i]`, at a rotated angle too.

    Rotation preserves area, so the inside fraction recomputed in the DISPLAY
    frame must match the one measured in the analysis frame -- cell by cell, in
    order. A misalignment would shuffle the classes onto the wrong cells while
    leaving every aggregate statistic intact.
    """
    tilted = shp_rotate(box(0, 0, 6, 4), 20.0, origin="centroid")
    packer = GridPacker(tilted, cell_width=1.0, cell_height=1.0)
    placement = packer.evaluate(0.37, 0.21, 20.0, classify=True)

    assert placement.classified
    assert len(placement.partial_classes) == len(placement.partial_cells)
    assert placement.partial_cells                    # not a vacuous check

    for cell, result in zip(placement.partial_cells, placement.partial_classes):
        expected = packer.usable.intersection(cell).area / cell.area
        assert result.inside_fraction == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------- #
# opt-in: classification never runs inside the search
# --------------------------------------------------------------------- #
def l_shape_packer() -> GridPacker:
    """The design note's headline instance: a 12x12 L with an off-grid obstacle."""
    return GridPacker(
        Polygon([(0, 0), (12, 0), (12, 6), (6, 6), (6, 12), (0, 12)]),
        [Polygon([(7, 1), (10.5, 1), (10.5, 2.5), (7, 2.5)])],
        cell_width=1.0, cell_height=1.0)


class SpyPacker(GridPacker):
    """GridPacker that records every call to the classifier."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifications = 0

    def _classify_partial(self, *args, **kwargs):
        self.classifications += 1
        return super()._classify_partial(*args, **kwargs)


def test_evaluate_does_not_classify_by_default():
    packer = l_shape_packer()
    placement = packer.evaluate(0.4, 0.6)

    assert placement.partial > 0                      # there IS something to classify
    assert placement.partial_classes == []
    assert not placement.classified

    opted_in = packer.evaluate(0.4, 0.6, classify=True)
    assert opted_in.classified
    assert len(opted_in.partial_classes) == opted_in.partial
    # opting in changes nothing else about the placement
    assert (opted_in.complete, opted_in.partial) == (placement.complete, placement.partial)


def test_optimize_exact_never_classifies():
    """The hot path stays hot: the search must not pay for the taxonomy."""
    spy = SpyPacker(
        Polygon([(0, 0), (12, 0), (12, 6), (6, 6), (6, 12), (0, 12)]),
        [Polygon([(7, 1), (10.5, 1), (10.5, 2.5), (7, 2.5)])],
        cell_width=1.0, cell_height=1.0)

    best, results = spy.optimize_exact()

    assert spy.classifications == 0
    assert len(results) > 1                           # a real sweep ran
    assert all(p.partial_classes == [] for p in results)
    assert best.partial > 0 and not best.classified

    # and the search result is exactly what it was without the feature
    plain_best, plain_results = l_shape_packer().optimize_exact()
    assert (best.complete, best.partial) == (plain_best.complete, plain_best.partial)
    assert len(results) == len(plain_results)


def test_optimize_never_classifies():
    spy = SpyPacker(box(0, 0, 5.5, 4.5), cell_width=1.0, cell_height=1.0)
    spy.optimize(steps=4)
    assert spy.classifications == 0


def test_classification_of_a_solved_placement_covers_every_partial():
    """End to end: solve, then classify the chosen placement only."""
    packer = l_shape_packer()
    best, _ = packer.optimize_exact()
    classified = packer.evaluate(best.dx, best.dy, best.angle, classify=True)

    assert classified.partial == best.partial
    assert len(classified.partial_classes) == best.partial
    assert all(isinstance(pc.label, PartialClass) for pc in classified.partial_classes)
    # every partial cell of this instance is an obstacle bite: at the optimum
    # the grid is flush with the L's walls, so the only boundary left inside a
    # cell is the off-grid obstacle's.
    assert {pc.label for pc in classified.partial_classes} == {PartialClass.D}
    assert all(pc.on_hole for pc in classified.partial_classes)
