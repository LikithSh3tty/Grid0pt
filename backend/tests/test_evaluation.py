"""Tests for the evaluation harness (design note section 11).

A harness that measures a method is only worth as much as its own correctness:
a wrong cost counter or a corpus whose "known optimum" is not actually optimal
would produce a table that looks authoritative and says nothing. So the pieces
that the results table rests on are tested directly:

  * the cost metric -- `GridPacker.evaluations` counts the primitive every
    search is built from, so it must count every call and only real calls;
  * the known answers -- the corpus claims a proven optimum for the exact-tiling
    families. That claim is checked by attaining it, including on the tilted
    rooms, where attaining it requires finding the rotation;
  * reproducibility -- section 11 promises a released corpus. The corpus is its
    generator, so the generator must be deterministic;
  * the driver -- one row per (instance, method), with the fields the tables
    read, measured rather than assumed.

The corpus used here is deliberately tiny and local. Running the real one is the
harness's job, not the test suite's.
"""
import math

import pytest

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon

from evaluation import corpus as corpus_module
from evaluation import methods as methods_module
from evaluation import run as run_module
from evaluation.corpus import Instance, l_shape, random_rectilinear, rectangle
from grid_packer import GridPacker


# --------------------------------------------------------------------------- #
# the cost metric
# --------------------------------------------------------------------------- #

def test_the_evaluation_counter_counts_evaluations():
    """One increment per `evaluate`, starting at zero, including the classifying
    read-backs -- the counter is the paper's cost column, so it may not quietly
    exclude work the method actually did."""
    packer = GridPacker(rectangle(12, 9), [], cell_width=3, cell_height=3)

    assert packer.evaluations == 0
    packer.evaluate(0.0, 0.0)
    packer.evaluate(1.0, 1.0, classify=True)
    assert packer.evaluations == 2


def test_the_counter_matches_the_number_of_placements_a_sweep_returns():
    """A sweep returns one placement per evaluation, so the two must agree.

    This is the check that makes the counter trustworthy on the methods where no
    independent count exists (the guided pipeline, whose evaluation total is not
    a product of loop bounds).
    """
    packer = GridPacker(l_shape(12, 9, 5, 4), [], cell_width=3, cell_height=3)
    _, results = packer.optimize(steps=5, snap_to_edges=False)

    assert packer.evaluations == len(results)


def test_each_instance_hands_out_a_fresh_packer():
    """Two methods must not inherit each other's counter."""
    instance = Instance("t", "rectangle", rectangle(12, 9), (), 3.0, 3.0)

    first = instance.packer()
    first.evaluate(0.0, 0.0)
    second = instance.packer()

    assert first.evaluations == 1
    assert second.evaluations == 0


# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #

def test_the_corpus_is_reproducible():
    """Built twice, identical -- names and geometry.

    The corpus is released as code, so determinism IS the release: a reader who
    runs this module must get the instances the paper's numbers came from.
    """
    first, second = corpus_module.build(quick=True), corpus_module.build(quick=True)

    assert [i.name for i in first] == [i.name for i in second]
    for a, b in zip(first, second):
        assert a.shape.equals(b.shape), a.name
        assert len(a.obstacles) == len(b.obstacles), a.name


def test_the_random_plans_are_seeded():
    same = random_rectilinear(3).equals(random_rectilinear(3))
    different = random_rectilinear(3).equals(random_rectilinear(4))

    assert same
    assert not different


def test_instance_names_are_unique():
    """Rows are keyed by name; a collision would silently merge two instances."""
    names = [i.name for i in corpus_module.build(quick=False)]

    assert len(names) == len(set(names))


def test_every_instance_is_a_valid_solvable_problem():
    for instance in corpus_module.build(quick=True):
        assert instance.shape.is_valid, instance.name
        assert instance.shape.area > 0, instance.name
        assert instance.packer().evaluate(0.0, 0.0).complete >= 0, instance.name


@pytest.mark.parametrize("width, height, cw, ch, expected", [
    (12, 9, 3, 3, 12),          # 4 x 3 cells
    (12, 9, 2, 3, 18),          # 6 x 3 cells
    (12, 10, 3, 3, None),       # does not divide: no claim made
])
def test_a_known_optimum_is_only_claimed_when_it_is_proven(width, height, cw, ch,
                                                           expected):
    assert corpus_module._exact_tiling_optimum(width, height, cw, ch) == expected


def test_the_claimed_optimum_of_an_exact_tiling_is_attained():
    """The claim is checked by attaining it, not by trusting the arithmetic."""
    instance = Instance("room", "rectangle", rectangle(12, 9), (), 3.0, 3.0,
                        known_optimum=12)
    best, _ = instance.packer().optimize_exact(angles=(0.0,))

    assert best.complete == instance.known_optimum
    assert best.partial == 0


def test_a_tilted_exact_tiling_keeps_its_optimum():
    """The corpus's sharpest claim: rotating a room rigidly cannot change what a
    grid can do to it, so the tilted instance carries the same known optimum --
    and reaching it requires finding the rotation."""
    tilted = shp_rotate(rectangle(12, 9), 23.0)
    instance = Instance("room-tilt23", "rotated", tilted, (), 3.0, 3.0,
                        known_optimum=12)

    unrotated, _ = instance.packer().optimize_exact(angles=(0.0,))
    guided, _ = instance.packer().optimize_guided()

    assert unrotated.complete < instance.known_optimum      # the rotation matters
    assert guided.complete == instance.known_optimum
    assert guided.angle == pytest.approx(23.0, abs=0.5)


def test_traced_instances_survive_the_image_pipeline():
    """A rasterised-and-retraced room comes back as the same room, to within the
    pixel quantisation the tracing introduces -- which is the noise these
    instances exist to carry."""
    source = rectangle(24, 18)
    shape, obstacles = corpus_module.traced(source, scale=0.25)

    assert shape.is_valid
    assert obstacles == []
    assert shape.area == pytest.approx(source.area, rel=0.05)


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #

def test_every_method_is_runnable_and_costs_something():
    instance = Instance("room", "rectangle", rectangle(12, 9), (), 3.0, 3.0)

    for method in methods_module.build(quick=True):
        packer = instance.packer()
        best = method.run(packer)

        assert best.complete >= 0, method.name
        assert packer.evaluations > 0, method.name


def test_classification_is_a_constant_cost_not_a_per_candidate_one():
    """The taxonomy may not run inside a sweep.

    The guided pipeline genuinely has to classify -- the vote is read off the
    taxonomy -- so the claim is not "never", it is "a constant number of times":
    the fringe read that produces the vote, and the probe the area stop spends.
    Everything else, including every baseline, classifies nothing at all. If
    classification ever scaled with the number of candidates, the method's
    wall-clock would carry a cost the baselines do not pay and the comparison in
    section 11 would be meaningless.
    """
    instance = Instance("l", "l_shape", l_shape(12, 9, 5, 4), (), 3.0, 3.0)

    for method in methods_module.build(quick=True):
        packer = instance.packer()
        classified = []
        plain = packer.evaluate

        def counting(*args, classify=False, **kwargs):
            classified.append(classify)
            return plain(*args, classify=classify, **kwargs)

        packer.evaluate = counting
        method.run(packer)

        expected = 2 if method.group in ("grid0pt", "ablation") else 0
        assert sum(classified) <= expected, method.name
        assert len(classified) > sum(classified), method.name


def test_the_baselines_are_the_original_code_path():
    """The baseline must be the sweep as it was, not a re-implementation.

    Asserted by its signature behaviour: a uniform sweep evaluates exactly
    steps x steps offsets when snapping is off.
    """
    instance = Instance("room", "rectangle", rectangle(12, 9), (), 3.0, 3.0)
    method = next(m for m in methods_module.build(quick=True)
                  if m.name == "uniform-nosnap-s10")

    packer = instance.packer()
    method.run(packer)

    assert packer.evaluations == 100


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def rows():
    instances = [
        Instance("room", "rectangle", rectangle(12, 9), (), 3.0, 3.0,
                 known_optimum=12),
        Instance("room-tilt23", "rotated", shp_rotate(rectangle(12, 9), 23.0),
                 (), 3.0, 3.0, known_optimum=12),
    ]
    methods = [m for m in methods_module.build(quick=True)
               if m.name in ("uniform-s10", "exact", "guided", "abl-norefine")]
    return run_module.run(instances, methods, verbose=False)


def test_the_driver_measures_every_pair(rows):
    assert len(rows) == 8
    assert {r.method for r in rows} == {"uniform-s10", "exact", "guided",
                                        "abl-norefine"}


def test_the_driver_records_cost_and_bounds(rows):
    for row in rows:
        assert row.evaluations > 0
        assert row.seconds >= 0.0
        assert row.partial_floor >= row.irreducible
        if row.certified:
            # The floor is a lower bound, so a certified row cannot sit under it.
            assert row.optimality_gap >= 0, row.instance


def test_the_shortfall_column_is_relative_to_the_best_anyone_reached(rows):
    for instance in {r.instance for r in rows}:
        group = [r for r in rows if r.instance == instance]

        assert max(r.complete_vs_best for r in group) == 0
        assert all(r.complete_vs_best <= 0 for r in group)


def test_the_known_answer_column_reports_the_rotation_the_baseline_misses(rows):
    """On the tilted room the baselines cannot reach the known optimum and the
    guided method can -- the headline the table has to be able to show."""
    tilted = {r.method: r for r in rows if r.instance == "room-tilt23"}

    assert tilted["guided"].reached_known_optimum is True
    assert tilted["uniform-s10"].reached_known_optimum is False
    assert tilted["exact"].reached_known_optimum is False


def test_the_tables_render(rows):
    """Smoke-test the reporting, so a long run cannot die at the last step."""
    for table in (run_module.per_method_table(rows),
                  run_module.ablation_table(rows),
                  run_module.certificate_table(rows)):
        assert table.strip()
        assert not math.isnan(len(table))


def test_rows_from_separate_chunks_report_as_one_run(rows, tmp_path, monkeypatch):
    """A full run may be split across processes, so the report must combine.

    The shortfall column is the thing that cannot simply be concatenated: it is
    relative to the best result on each instance, so it has to be recomputed
    over the union. Here each chunk holds a DIFFERENT method for the same
    instances, which is exactly the case where a per-chunk shortfall would be
    wrong -- every chunk would call its own method the best there is.
    """
    monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path)
    weak = [r for r in rows if r.method == "uniform-s10"]
    strong = [r for r in rows if r.method == "guided"]
    run_module.write_outputs(weak, {}, "chunk-weak")
    run_module.write_outputs(strong, {}, "chunk-strong")

    merged = run_module.load_rows([tmp_path / "chunk-weak.json",
                                   tmp_path / "chunk-strong.json"])

    assert len(merged) == len(weak) + len(strong)
    tilted = {r.method: r for r in merged if r.instance == "room-tilt23"}
    assert tilted["guided"].complete_vs_best == 0
    assert tilted["uniform-s10"].complete_vs_best < 0


def test_a_rerun_chunk_supersedes_the_old_one(rows, tmp_path, monkeypatch):
    """Re-running a chunk must replace its rows, not double them."""
    monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path)
    run_module.write_outputs(rows, {}, "first")
    run_module.write_outputs(rows, {}, "second")

    merged = run_module.load_rows([tmp_path / "first.json",
                                   tmp_path / "second.json"])

    assert len(merged) == len(rows)


def test_the_outputs_are_written(rows, tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path)
    path = run_module.write_outputs(rows, {"quick": True}, "unit")

    assert path.exists()
    assert (tmp_path / "unit.json").exists()
    assert path.read_text(encoding="utf-8").count("\n") == len(rows) + 1


# --------------------------------------------------------------------------- #
# the rotation certificate, measured rather than quoted
# --------------------------------------------------------------------------- #
# The certificate is a property of the INSTANCE -- of that grid on that region --
# not of the method that searched it. So it is computed once per instance and
# attached to every row, which is what lets a row be scored against the proven
# optimum instead of against the best answer anything happened to find.

@pytest.fixture(scope="module")
def certified_rows():
    instances = [
        Instance("room", "rectangle", rectangle(12, 9), (), 3.0, 3.0,
                 known_optimum=12),
        Instance("room-tilt23", "rotated", shp_rotate(rectangle(12, 9), 23.0),
                 (), 3.0, 3.0, known_optimum=12),
    ]
    methods = [m for m in methods_module.build(quick=True)
               if m.name in ("uniform-s10", "guided")]
    return run_module.run(instances, methods, verbose=False, certify=True)


def test_the_certificate_is_absent_unless_asked_for(rows):
    """It costs far more than every method in the table put together, so a plain
    run must not start paying for it."""
    for row in rows:
        assert row.rotation_bound is None
        assert row.complete_vs_proven is None


def test_every_row_of_an_instance_carries_the_same_bound(certified_rows):
    """One certificate per instance, not per method. A bound that differed by
    method would mean it was measuring the search rather than the region."""
    for name in ("room", "room-tilt23"):
        bounds = {r.rotation_bound for r in certified_rows if r.instance == name}
        assert bounds == {12}


def test_the_shortfall_is_measured_against_the_proven_optimum(certified_rows):
    """The column the certificate exists to make possible. `complete_vs_best`
    can only say "nothing here did better"; this says how far from the best
    there could be -- and the uniform sweep on a tilted room is the case where
    those two differ."""
    tilted = {r.method: r for r in certified_rows if r.instance == "room-tilt23"}

    assert tilted["guided"].complete_vs_proven == 0
    assert tilted["uniform-s10"].complete_vs_proven < 0


def test_the_certificate_reports_that_it_closed(certified_rows):
    for row in certified_rows:
        assert row.rotation_optimal is True
        assert row.rotation_nodes > 0


def test_the_rotation_table_renders(certified_rows):
    text = run_module.rotation_table(certified_rows)

    assert "room-tilt23" in text
    assert "proven" in text.lower()


def test_the_rotation_table_says_so_when_nothing_was_certified(rows):
    assert "--certify" in run_module.rotation_table(rows)


def test_the_driver_takes_a_certify_flag():
    parsed = run_module.build_parser().parse_args(["--certify"])

    assert parsed.certify is True
    assert run_module.build_parser().parse_args([]).certify is False


def test_the_per_method_table_scores_against_the_proven_optimum(certified_rows):
    """Once an instance is certified, "how far below the best anyone got" is
    the weaker question. The column only appears when there is a proof to
    measure against, so an uncertified run reads exactly as it did before."""
    certified = run_module.per_method_table(certified_rows)
    plain = run_module.per_method_table(
        [run_module.Row(**{**vars(r), "rotation_bound": None,
                           "complete_vs_proven": None}) for r in certified_rows])

    assert "vs-optimal" in certified
    assert "vs-optimal" not in plain


# --------------------------------------------------------------------------- #
# the report reads what was actually run
# --------------------------------------------------------------------------- #
# Twice now the document has described one run while rendering another: first
# because it was pinned to a hand-named CSV, then because the full corpus was
# run and the tables carried on showing the quick one. The fix is not another
# correct constant -- it is to stop having constants that can disagree with the
# data, so the counts are derived from the rows the report actually loaded.

def test_the_report_prefers_the_fullest_run_available(tmp_path, monkeypatch):
    from evaluation import report as report_module

    monkeypatch.setattr(report_module, "RESULTS_DIR", tmp_path)
    (tmp_path / "quick.csv").write_text("instance,method\na,guided\n", encoding="utf-8")

    assert report_module.results_path().name == "quick.csv"

    (tmp_path / "full.csv").write_text("instance,method\na,guided\n", encoding="utf-8")

    assert report_module.results_path().name == "full.csv"


def test_the_report_says_which_corpus_it_rendered(rows):
    """The sentence introducing the results table is generated from the rows,
    so it cannot claim 16 instances while the table shows 72."""
    from evaluation import report as report_module

    described = report_module.corpus_sentence(
        [{"instance": r.instance, "method": r.method, "family": r.family}
         for r in rows])

    assert "2 instances" in described
    assert "4 methods" in described
    assert f"{len(rows)} rows" in described


def test_the_report_builds_its_certification_table_from_the_run(certified_rows):
    """It was a table of literals pasted from a one-off loop, which went stale
    the moment the search got faster -- the disc's window count fell by an
    order of magnitude and the document still quoted the old one."""
    from evaluation import report as report_module

    data = report_module.certification_rows(
        [{**vars(r)} for r in certified_rows])

    header, *body = data
    assert header[0] == "instance"
    names = {row[0] for row in body}
    assert "room-tilt23" in names
    assert body[-1][0].startswith("all ")


def test_the_certification_table_says_when_nothing_was_certified(rows):
    from evaluation import report as report_module

    assert report_module.certification_rows([{**vars(r)} for r in rows]) is None


def test_the_certification_table_comes_from_whichever_run_certified(tmp_path,
                                                                    monkeypatch):
    """The main tables want the fullest run; this one wants the certified run,
    and they need not be the same file. Preferring the fullest for both meant a
    72-instance uncertified run hid a 16-instance certified one, and the table
    silently emptied."""
    from evaluation import report as report_module

    monkeypatch.setattr(report_module, "RESULTS_DIR", tmp_path)
    (tmp_path / "full.csv").write_text(
        "instance,method,complete,rotation_bound\na,guided,5,\n", encoding="utf-8")
    (tmp_path / "quick.csv").write_text(
        "instance,method,complete,rotation_bound,rotation_nodes,"
        "rotation_seconds,rotation_optimal\nb,guided,7,7,3,1.5,True\n",
        encoding="utf-8")

    assert report_module.results_path().name == "full.csv"
    assert report_module.certification_path().name == "quick.csv"


# --------------------------------------------------------------------------- #
# the paper
# --------------------------------------------------------------------------- #
# Its abstract states three numbers -- how many instances certified, how many
# the pipeline reaches, at what cost. They are computed from the run for the
# same reason every other figure in these documents is: numbers written into
# prose beside the data have gone stale three times in this project's week.

def test_the_paper_counts_its_abstract_from_the_run(certified_rows):
    from evaluation import paper as paper_module

    facts = paper_module.headline([{**vars(r)} for r in certified_rows])

    assert facts["instances"] == "2"
    assert facts["closed"] == "2"
    assert facts["guided_misses"] == "0"


def test_the_paper_scores_methods_against_the_proven_optimum(certified_rows):
    """The table the certificate exists to make printable: a method's standing
    against the truth rather than against whatever else was run."""
    from evaluation import paper as paper_module

    rows = paper_module.optimality_table([{**vars(r)} for r in certified_rows])

    header, *body = rows
    assert header[1] == "optimal on"
    guided = next(r for r in body if r[0] == "guided")
    assert guided[1] == "2/2"
    sweep = next(r for r in body if r[0] == "uniform-s10")
    assert sweep[1] != guided[1]           # the sweep misses the tilted room


def test_the_paper_says_nothing_it_cannot_measure(rows):
    """Without a certified run there is no proven optimum to score against, so
    the table is withheld rather than filled with the next best thing."""
    from evaluation import paper as paper_module

    assert paper_module.optimality_table([{**vars(r)} for r in rows]) is None


def test_both_documents_render(tmp_path):
    from evaluation import paper as paper_module
    from evaluation import report as report_module

    assert paper_module.build(tmp_path / "paper.pdf").exists()
    assert report_module.build(tmp_path / "report.pdf").exists()
