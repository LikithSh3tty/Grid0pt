"""Render the implementation report to PDF.

Everything in this document is a measurement taken from this repository or a
statement about code in it. Where a number appears it was produced by a command
recorded in the "How to reproduce" section, and where a claim contradicts the
design note the counterexample that settles it is given inline.

    python -m evaluation.report [output.pdf]

Written with reportlab. The results tables are read from the evaluation run's
CSV when it is present, so re-running the corpus and re-rendering keeps the
document and the data in step.
"""
from __future__ import annotations

import csv
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
RESULTS_DIR = BACKEND / "evaluation" / "results"

#: Result files in order of preference, fullest first -- the stems `run.py`
#: writes for `--full` and for the default quick corpus.
#:
#: This has now drifted twice. First the path was pinned to a hand-named copy,
#: so re-running the corpus left the tables on the old data while the prose
#: quoted the new. Then the full corpus was run and the tables carried on
#: rendering the quick one, because nothing here knew `full.csv` had appeared.
#: Both times the document contradicted itself while every individual number in
#: it was, at some point, correct.
#:
#: So the rule is preference rather than a fixed name, and every count in
#: section 5 is derived from the rows actually loaded (see `corpus_sentence`)
#: rather than written down beside them. A constant that can disagree with the
#: data eventually will.
RESULTS_STEMS = ("full", "quick")
DEFAULT_OUT = REPO / "Grid0pt_Implementation_Report.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#c8c8c8")
BAND = colors.HexColor("#eef1f4")
ACCENT = colors.HexColor("#8a2f2f")


# --------------------------------------------------------------------------- #
# styles
# --------------------------------------------------------------------------- #

def build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=base["Title"], fontName="Times-Bold", fontSize=21,
        leading=25, textColor=INK, spaceAfter=4)
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"], fontName="Times-Italic", fontSize=11.5,
        leading=15, textColor=MUTED, alignment=1, spaceAfter=14)
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontName="Times-Bold", fontSize=14.5,
        leading=18, textColor=INK, spaceBefore=16, spaceAfter=6)
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="Times-Bold", fontSize=11.5,
        leading=15, textColor=INK, spaceBefore=11, spaceAfter=4)
    s["body"] = ParagraphStyle(
        "body", parent=base["Normal"], fontName="Times-Roman", fontSize=10,
        leading=14.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=12, bulletIndent=2, spaceAfter=3)
    s["mono"] = ParagraphStyle(
        "mono", parent=base["Normal"], fontName="Courier", fontSize=8.1,
        leading=10.4, textColor=INK, backColor=colors.HexColor("#f4f5f7"),
        borderPadding=6, spaceBefore=4, spaceAfter=8)
    s["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"], fontName="Times-Italic", fontSize=8.6,
        leading=11, textColor=MUTED, spaceBefore=2, spaceAfter=10)
    s["finding"] = ParagraphStyle(
        "finding", parent=s["body"], fontName="Times-Bold", fontSize=10.5,
        textColor=ACCENT, spaceBefore=10, spaceAfter=2, alignment=0)
    return s


class Doc(BaseDocTemplate):
    """Two-part page: content frame plus a footer with the page number."""

    def __init__(self, path: str, **kw):
        super().__init__(path, pagesize=A4, leftMargin=22 * mm,
                         rightMargin=22 * mm, topMargin=20 * mm,
                         bottomMargin=18 * mm, title="Grid0pt implementation report",
                         author="Grid0pt", **kw)
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="body")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                            onPage=self._decorate)])

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(A4[0] / 2.0, 11 * mm, f"{doc.page}")
        if doc.page > 1:
            canvas.setStrokeColor(RULE)
            canvas.setLineWidth(0.4)
            canvas.line(doc.leftMargin, A4[1] - 15 * mm,
                        A4[0] - doc.rightMargin, A4[1] - 15 * mm)
            canvas.drawString(doc.leftMargin, A4[1] - 13 * mm,
                              "Grid0pt — implementation report")
        canvas.restoreState()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def para(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: Sequence[str], style) -> List[Paragraph]:
    return [Paragraph(f"•&nbsp;&nbsp;{t}", style) for t in items]


def code(text: str, style) -> Paragraph:
    escaped = (text.replace("&", "&amp;").replace("<", "&lt;")
               .replace(">", "&gt;").replace(" ", "&nbsp;")
               .replace("\n", "<br/>"))
    return Paragraph(escaped, style)


def table(data: List[List[str]], widths: Sequence[float],
          align_right: Sequence[int] = ()) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("LEADING", (0, 0), (-1, -1), 10.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    for row in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, row), (-1, row),
                      colors.HexColor("#fafbfc")))
    t.setStyle(TableStyle(style))
    return t


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                          # pragma: no cover - not a repo
        return ""


def results_path() -> Optional[Path]:
    """The fullest results file present, or None."""
    for stem in RESULTS_STEMS:
        candidate = RESULTS_DIR / f"{stem}.csv"
        if candidate.exists():
            return candidate
    return None


def load_results(path: Optional[Path] = None) -> List[dict]:
    path = path or results_path()
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def headline_caption(rows: Sequence[dict]) -> str:
    """Table 5's caption, read off the table rather than remembered beside it.

    Same reason as `corpus_sentence`: this caption has already been wrong once,
    quoting the cost and the margin from a run that was no longer the one being
    rendered.
    """
    def mean_of(method: str, field: str, cast=float) -> Optional[float]:
        values = [cast(r[field]) for r in rows if r["method"] == method]
        return statistics.mean(values) if values else None

    method_ev = mean_of("guided", "evaluations", int)
    ladder_ev = mean_of("fixed15-s10", "evaluations", int)
    method_short = mean_of("guided", "complete_vs_best", int)
    ladder_short = mean_of("fixed15-s10", "complete_vs_best", int)
    if None in (method_ev, ladder_ev, method_short, ladder_short):
        return "Table 5. Every method over the corpus."

    return (f"Table 5. Every method over the corpus. The full method averages "
            f"{method_ev:.0f} evaluations; the rotation baseline it replaces "
            f"needs {ladder_ev:.0f} for a result {method_short - ladder_short:.2f} "
            f"cells worse.")


def corpus_sentence(rows: Sequence[dict]) -> str:
    """Describe the run being rendered, counted from it.

    Written rather than stated so the introduction to the results table cannot
    claim one corpus while the table below shows another -- which is the failure
    this whole arrangement exists to prevent.
    """
    instances = {r["instance"] for r in rows}
    methods = {r["method"] for r in rows}
    families = {r.get("family", "") for r in rows} - {""}
    return (f"{len(instances)} instances across {len(families)} families, "
            f"{len(methods)} methods, {len(rows)} rows.")


def per_method_rows(rows: Sequence[dict]) -> List[List[str]]:
    by_method: Dict[str, List[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    order = {"grid0pt": 0, "baseline": 1, "ablation": 2, "reference": 3}
    out = [["method", "group", "mean vs best", "worst", "known hit",
            "mean eval", "mean s"]]
    for name, group in sorted(by_method.items(),
                              key=lambda kv: (order.get(kv[1][0]["group"], 9),
                                              kv[0])):
        short = [int(r["complete_vs_best"]) for r in group]
        known = [r for r in group if r["reached_known_optimum"] not in ("", "None")]
        hit = sum(1 for r in known if r["reached_known_optimum"] == "True")
        out.append([
            name, group[0]["group"],
            f"{statistics.mean(short):+.2f}", f"{min(short):+d}",
            f"{hit}/{len(known)}" if known else "–",
            f"{statistics.mean(int(r['evaluations']) for r in group):.0f}",
            f"{statistics.mean(float(r['seconds']) for r in group):.2f}",
        ])
    return out


def instance_rows(rows: Sequence[dict], instances: Sequence[str],
                  methods: Sequence[str]) -> List[List[str]]:
    out = [["instance", "method", "complete", "evaluations", "seconds"]]
    for inst in instances:
        for method in methods:
            match = [r for r in rows
                     if r["instance"] == inst and r["method"] == method]
            if match:
                r = match[0]
                out.append([inst, method, r["complete"], r["evaluations"],
                            f"{float(r['seconds']):.2f}"])
    return out


def certificate_summary(rows: Sequence[dict]) -> List[str]:
    guided = [r for r in rows if r["method"] == "guided"]
    if not guided:
        return []
    certified = [r for r in guided if r["certified"] == "True"]
    exact = [r for r in certified if int(r["optimality_gap"]) == 0]
    gaps = [int(r["optimality_gap"]) for r in certified]
    return [
        f"instances solved: {len(guided)}",
        f"certificate assumption holds: {len(certified)} "
        f"({100.0 * len(certified) / len(guided):.0f}%)",
        f"certified optimal, gap = 0: {len(exact)} of those {len(certified)}",
        f"gap over certified instances: mean {statistics.mean(gaps):.2f}, "
        f"max {max(gaps)}",
    ]


# --------------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------------- #

def build(out_path: Path) -> Path:
    s = build_styles()
    rows = load_results()
    story: List = []

    W = A4[0] - 44 * mm

    # ---------------------------------------------------------------- title
    story += [
        Spacer(1, 34 * mm),
        para("Grid0pt — Implementation Report", s["title"]),
        para("What was built against the design note, what it was measured to "
             "do, and where the note turned out to be wrong", s["subtitle"]),
        Spacer(1, 4 * mm),
    ]

    head = git("log", "--format=%h %ad", "--date=short", "-1")
    story += [table([
        ["repository", "Grid0pt — backend/ (Python), frontend/ (React)"],
        ["design note", "Grid0pt_Method_and_Strategy.pdf, 11 pages, 13 sections"],
        ["work covered", "note roadmap steps 1–4, plus corrections to the note"],
        ["head when rendered", head or "–"],
        ["test suite", "333 tests, all passing"],
        ["evaluation", "quick: 16 × 16 = 256 rows; full: 72 × 18 = 1296 rows"],
    ], [38 * mm, W - 38 * mm])]

    story += [
        Spacer(1, 8 * mm),
        para("Summary", s["h2"]),
        para(
            "The placement search no longer samples, and no longer enumerates "
            "either. Translation is solved on both axes at once, by asking where "
            "a single cell may sit rather than where the whole grid should go: "
            "the answer is the region eroded by the cell, and folding it modulo "
            "the grid period turns every offset's complete count into an overlap "
            "depth whose maximum is attained at finitely many points. Rotation is "
            "read off the partial-cell fringe instead of scanned. Every result "
            "carries a per-instance certificate bounding how far from optimal it "
            "could possibly be, and an evaluation harness measures all of it "
            "against the code as it was.",
            s["body"]),
        para(
            "That also changes what can be claimed. The overlap depth is the "
            "complete count over every offset at once, so its maximum bounds any "
            "placement whatsoever and reaching it proves the placement optimal in "
            "translation — for any shape, not only for a boundary square to the "
            "grid.",
            s["body"]),
        para(
            "The same bound, applied to a window of angles rather than to one, "
            "closes the last gap. Turning by theta moves a point at radius r by "
            "r·theta, so a whole angular window is bounded by the erosion of the "
            "region grown by radius × half-window, and branch and bound over the "
            "placement period discards windows wholesale without ever locating "
            "the angles at which the count jumps. The vote of section 3.3 becomes "
            "the incumbent generator and this becomes the proof. Optimality is "
            "therefore global — over offsets and angles together — and the answer "
            "is a theorem about the region rather than the best of a search.",
            s["body"]),
        para(
            "Three claims in the design note did not survive being implemented "
            "and are corrected here with the measurements that settle them: the "
            "boundary-covering floor of section 9.1 is false as written, the "
            "stop criterion of section 8.4 separates nothing where the note "
            "places it, and the exact translation search of section 5 is not in "
            "fact exact on slanted boundaries — a counterexample returns 83 "
            "complete cells where 84 is attainable. Solving one axis fixed that "
            "instance without fixing the class: a trapezoid returns 59 where 60 "
            "is attainable until both axes are solved.",
            s["body"]),
        PageBreak(),
    ]

    # ---------------------------------------------------------------- 1
    story += [
        para("1. What was already there, and what this work added", s["h1"]),
        para(
            "The note's roadmap has five steps. Steps 1–3 were implemented "
            "before this work: exact translation over critical offsets, the "
            "A–F partial-cell taxonomy, and the orientation-guided rotation "
            "with its local refine. Step 4 — the certificate, the stats "
            "plumbing, the corpus, the baselines and the ablations — and the "
            "solver work below were added here. Step 5 is the write-up and is "
            "not code.",
            s["body"]),
        table([
            ["note", "component", "state"],
            ["§5", "exact translation via critical offsets", "existed; found not exact, superseded"],
            ["§6–7", "A–F partial-cell taxonomy", "existed"],
            ["§8", "rotation vote, guided pipeline, refine", "existed"],
            ["§8.4", "recoverable-area stop", "added (relocated, see §4.2)"],
            ["§9", "per-instance optimality certificate", "added (corrected, see §4.1)"],
            ["§10", "service surfaces the new statistics", "added"],
            ["§11", "corpus, baselines, metrics, ablations", "added"],
            ["—", "column solver: dy solved, not enumerated", "added (superseded, see §4.3)"],
            ["—", "erosion solver: both axes solved", "added (see §3.2.1)"],
            ["—", "rotation certificate: optimality over angles", "added (see §3.5)"],
        ], [16 * mm, 74 * mm, W - 90 * mm]),
        para("Table 1. Design-note sections against the state of the code.", s["caption"]),

        para("1.1 Files", s["h2"]),
        table([
            ["file", "lines", "holds"],
            ["backend/grid_packer.py", "2773", "the solver: evaluate, taxonomy, vote, certificate, the translation searches"],
            ["backend/packer_service.py", "197", "request path; assembles the stats the API returns"],
            ["backend/evaluation/corpus.py", "350", "instance generators, including proven-optimum families"],
            ["backend/evaluation/methods.py", "177", "baselines, the method, and the one-at-a-time ablations"],
            ["backend/evaluation/run.py", "397", "driver, metrics, reporting, multi-run merge"],
            ["backend/evaluation/report.py", "–", "this document"],
            ["backend/tests/ (16 modules)", "–", "333 tests"],
        ], [52 * mm, 14 * mm, W - 66 * mm], align_right=(1,)),
        para("Table 2. Where the work lives.", s["caption"]),
    ]

    # ---------------------------------------------------------------- 2
    story += [
        para("2. What is used", s["h1"]),
        para(
            "No new runtime dependency was introduced. The solver's geometry is "
            "Shapely throughout, image tracing is OpenCV, and the evaluation "
            "harness adds only the standard library. reportlab is used for this "
            "document alone and is not imported by any code path the service "
            "touches.",
            s["body"]),
        table([
            ["Shapely", "runtime", "polygons, prepared predicates, rotation, erosion of the usable region"],
            ["OpenCV", "runtime", "thresholding and contour tracing behind GridPacker.from_image"],
            ["NumPy", "runtime", "offset ladders in the retained uniform-sweep baseline"],
            ["FastAPI", "runtime", "the HTTP layer, untouched by this work"],
            ["pytest", "development", "243 tests"],
            ["reportlab", "document", "this PDF only"],
        ], [26 * mm, 24 * mm, W - 50 * mm]),
        para("Table 3. Dependencies and what each is for.", s["caption"]),
    ]

    # ---------------------------------------------------------------- 3
    story += [
        PageBreak(),
        para("3. The method as implemented", s["h1"]),

        para("3.1 The problem", s["h2"]),
        para(
            "Let U be the usable region — the shape with the obstacles "
            "removed, possibly with holes. A placement of the grid is a triple "
            "(dx, dy, theta). Under it the plane is tiled by cells; each is "
            "complete when it lies wholly inside U, partial when it straddles "
            "the boundary, outside otherwise. The objective is to maximise the "
            "complete count, with a tie-break penalty on partials. Translation "
            "lives on a torus — dx in [0, cw), dy in [0, ch) — because "
            "the grid is periodic; theta ranges over 90° for square cells "
            "and 180° for rectangular ones, since a quarter turn of a "
            "rectangular cell swaps its sides and is a genuinely different "
            "tiling.",
            s["body"]),

        para("3.2 Translation, first attempt: the column solver", s["h2"]),
        para(
            "The complete count is piecewise-constant in the offsets and jumps "
            "only where a grid line meets a boundary vertex, so a finite set of "
            "critical offsets covers every attainable value. The previous "
            "implementation took that literally in both axes and evaluated one "
            "point per face of the Vx × Vy arrangement, reclassifying all N "
            "cells each time. That is exact-in-principle but still enumeration, "
            "and it is quadratic in the number of distinct vertex coordinates.",
            s["body"]),
        para(
            "The inner loop is unnecessary. Fix dx and cut the region into the "
            "columns the grid makes. Within one column, a cell is complete "
            "exactly when every horizontal cut of it spans the full column "
            "width, so the column's completeness is a set of height intervals "
            "that does not depend on dy at all. Eroding each interval by the "
            "cell height turns “where does a cut fit” into “where "
            "does a cell fit”, and the count becomes a lattice-hit problem:",
            s["body"]),
        code("N_complete(dx, dy)  =  # { (i, j) : dy + j·ch  lands in  Y_i }",
             s["mono"]),
        para(
            "where Y_i is column i's eroded interval set. As a function of dy "
            "that count is piecewise-constant with breakpoints only at the "
            "interval ends reduced modulo ch, so one sweep of those ends solves "
            "the axis exactly. Cost per dx falls from Vy full evaluations to one, "
            "and the O(N) per-cell scan leaves the inner loop entirely.",
            s["body"]),
        para(
            "Two families of breakpoint matter when computing the intervals, and "
            "missing the second is what made the old search wrong on slanted "
            "boundaries: a vertex of the region at that height, and a crossing "
            "of the column's own side lines by the boundary. A slanted wall has "
            "no vertex between its ends, yet it stops covering the full column "
            "width exactly where it crosses the column's side — a height "
            "that depends on where the column happens to sit. These are the "
            "note's own “diagonal critical lines on the torus” from "
            "section 4.1.",
            s["body"]),
        para(
            "This solves dy and leaves dx enumerated over the same vertex-derived "
            "critical set, so the caveat above moves onto dx rather than going "
            "away: the diagonal events are exactly as invisible to a vertex "
            "offset in one axis as in two. Section 4.3 gives the trapezoid where "
            "that costs a cell.",
            s["body"]),

        para("3.2.1 Translation, solved: erosion, fold, deepest overlap",
             s["h2"]),
        para(
            "Both axes go together once the question is asked about a cell rather "
            "than about the grid. A cell is complete exactly when its lower-left "
            "corner p satisfies",
            s["body"]),
        code("p + [0,cw] × [0,ch]  ⊆  U        i.e.   p ∈ F = U ⊖ cell",
             s["mono"]),
        para(
            "F is the region eroded by the cell — a condition on p alone, with "
            "no reference to any offset. Erosion is not a Shapely primitive, but "
            "through the complement it becomes a dilation, and a rectangle is "
            "separable, so F is two segment sweeps of the region's outside: exact "
            "polygon work, no sampling, obstacles handled for free because a hole "
            "dilates into a forbidden zone one cell wider than itself.",
            s["body"]),
        para(
            "Cell corners sit at (dx + i·cw, dy + j·ch), so the complete count is "
            "the number of points of a translated lattice landing in a fixed set. "
            "Reduce F modulo the lattice — cut it along the grid lines and stack "
            "the pieces onto one cell of the grid — and that count becomes the "
            "number of pieces covering the single point (dx, dy):",
            s["body"]),
        code("N_complete(dx, dy)  =  # { pieces of F mod lattice covering (dx, dy) }",
             s["mono"]),
        para(
            "So the 2-D search is “where do these pieces overlap most "
            "deeply”. The pieces are closed, so the depth is upper "
            "semi-continuous and its maximum survives at a corner of the "
            "arrangement they cut the cell into; those corners are the endpoints "
            "of the pieces' outlines once their crossings are noded, which a "
            "single union computes. Finitely many candidates, nothing sampled on "
            "either axis, and no assumption about the boundary — which is what "
            "makes this exact for any shape and not only a rectilinear one.",
            s["body"]),
        para(
            "One detail is load-bearing. A region that tiles exactly meets the "
            "far grid line flush, so its last column of corners is the edge of F "
            "— a piece of zero area. Discard it as degenerate and a 36×27 room at "
            "3×3 reports 88 complete cells instead of the 108 that plainly fit. "
            "The lower-dimensional pieces are kept for that reason.",
            s["body"]),
        para(
            "The depth is not merely a cheaper search: it is N_complete over the "
            "whole torus at once, so the deepest overlap is an upper bound on "
            "every placement there is. The implementation walks the candidates in "
            "depth order and stops as soon as the placement in hand scores at "
            "least the next candidate's bound — normally immediately, which puts "
            "translation at one evaluation per angle regardless of boundary "
            "complexity. The candidate is still scored by the ordinary "
            "evaluation, so the search chooses a placement and one definition "
            "scores it.",
            s["body"]),

        para("3.3 Rotation: the fringe vote", s["h2"]),
        para(
            "Oblique partial cells each contribute a chord orientation weighted "
            "by chord length and inside fraction; the weighted circular mean on "
            "the alignment circle gives the dominant wall orientation, and the "
            "resultant length R measures how concentrated the fringe is. R gates "
            "whether to rotate at all. Candidate orientations are clustered and "
            "each is solved exactly for translation; the un-rotated base stays "
            "in the pool, so a misfiring vote costs evaluations and nothing else. "
            "Exact ties go to the smallest turn.",
            s["body"]),

        para("3.4 The certificate", s["h2"]),
        para(
            "Two independent lower bounds on the partial count are combined. The "
            "first is |X|, the count of regime-X cells — classes E and F, "
            "features smaller than a cell — which are partial under every "
            "placement. The second is a boundary-covering bound, corrected as "
            "described in section 4.1 below. The reported floor is the larger, "
            "and the gap is the achieved partial count minus that floor; a gap of "
            "zero certifies that no placement of that grid on that region has "
            "fewer partial cells, including over rotation.",
            s["body"]),

        para("3.5 Certifying the angle", s["h2"]),
        para(
            "Section 3.2.1 settles translation at whatever angle it is asked "
            "about, which leaves exactly one thing unproven: the angle. The vote "
            "of 3.3 is evidence, and nothing in it bounds what a different angle "
            "could reach. Enumerating the angles at which the count jumps is the "
            "obvious repair and the wrong one — a placement is tight in three "
            "degrees of freedom, so an event needs three simultaneous contacts "
            "and the candidate triples run cubic in the boundary size, each a "
            "trigonometric system.",
            s["body"]),
        para(
            "Bound a whole window of angles instead, and never find the jumps at "
            "all. Turning by theta moves a point at radius r from the pivot by "
            "exactly r·theta, so for every theta within a window of theta₀:",
            s["body"]),
        code("R(theta)  ⊆  R(theta₀) grown by  radius × half-window\n"
             "max M(theta) over the window  ≤  maxdepth(fold(erode(that, cell)))",
             s["mono"]),
        para(
            "— the erosion solver's own machinery run once on a slightly "
            "fattened region. The radius is the minimum bounding circle's rather "
            "than the centroid pivot's, which roughly halves what a window costs "
            "on a long room; reading it about a different centre is sound only "
            "because translation is solved, since rotating about another point "
            "differs by a translation and the quantity bounded is the maximum "
            "over translations.",
            s["body"]),
        para(
            "Branch and bound then closes the period. Windows come off a heap "
            "best-bound-first, so the moment the best remaining bound fails to "
            "beat the incumbent, every remaining window fails too. Two things "
            "make it terminate rather than merely converge. The vote seeds it: an "
            "optimum usually sits at a wall-flush angle attained at one exact "
            "value, which bisection approaches and never lands on, so an unseeded "
            "search would keep losing to a bound it could never match. And "
            "splitting stops once a window is worth less geometry than the "
            "tolerance every containment test here already grants, at which point "
            "the bound is the value.",
            s["body"]),
        para(
            "So the vote is not replaced, it is demoted to what it is good at. It "
            "generates the incumbent; the branch and bound generates the proof.",
            s["body"]),
    ]

    # ---------------------------------------------------------------- 4
    story += [
        PageBreak(),
        para("4. Findings", s["h1"]),
        para(
            "These are the results of implementing the note rather than reading "
            "it. Each is stated with the measurement that settles it.",
            s["body"]),

        para("F1. The boundary-covering floor of section 9.1 is false as written",
             s["finding"]),
        para(
            "The note bounds the partial count below by L / (c·s) for "
            "boundary length L and cell size s. That fails on the simplest "
            "instance available: a 12×9 room at 3×3 cells tiles exactly "
            "into 12 complete cells and zero partials, while L / diagonal claims "
            "a floor of 10. Boundary lying along grid lines is covered by the "
            "edges of complete cells and makes nothing partial; only boundary "
            "forced through a cell interior does.",
            s["body"]),
        para(
            "The implemented bound therefore uses the forced length: group the "
            "boundary edges by orientation on the alignment circle, subtract the "
            "single best-aligned family, and divide by the longest chord one cell "
            "can hold — which is not an unspecified constant but exactly the "
            "cell diagonal. It is conservative in two ways that keep it a valid "
            "lower bound: it assumes the best direction can be aligned perfectly, "
            "and it ignores that an aligned wall must also land on a grid line "
            "rather than merely parallel to one. Its one assumption — that a "
            "cell carries a single straight crossing — is measured, not "
            "asserted: the certificate reports the largest chord actually "
            "observed and marks itself uncertified when that exceeds the diagonal.",
            s["body"]),

        para("F2. The stop criterion of section 8.4 separates nothing where the "
             "note places it", s["finding"]),
        para(
            "Read at the entrance as a should-we-rotate gate, the fringe's "
            "recoverable area cannot distinguish the case that must rotate from "
            "the case that must not. Measured before rotating, at 3×3 cells:",
            s["body"]),
        table([
            ["instance", "recoverable area", "R", "is rotating worth it?"],
            ["24×18 room tilted 12°", "2.89 cells", "1.000", "yes: +14 complete cells"],
            ["disc, radius 12", "3.10 cells", "0.029", "no: nothing to align to"],
        ], [46 * mm, 30 * mm, 16 * mm, W - 92 * mm]),
        para("Table 4. Any threshold that silences the disc also silences the "
             "headline result. R already separates them cleanly.", s["caption"]),
        para(
            "One stage later it separates perfectly. After the vote's angle has "
            "been solved, an instance whose walls came out flush holds exactly "
            "0.00 cells of recoverable area, while one still compromising between "
            "two wall families holds 1.2–1.7. So the reading was placed "
            "there, gating the refine — the expensive stage — rather "
            "than the rotation. One classifying evaluation decides against eight "
            "translation solves: on the tilted room at 23°, 12 evaluations become "
            "5 for an identical complete count. Those figures used to read 905 "
            "and 105; solving both translation axes made a solve cost one "
            "evaluation instead of one per critical offset, so the stop now saves "
            "a smaller multiple of a far smaller number.",
            s["body"]),

        para("F3. The exact translation search of section 5 is not exact",
             s["finding"]),
        para(
            "Its critical set is derived from region vertices, which covers every "
            "event for axis-aligned edges. A slanted edge also flips a cell when a "
            "lattice corner grazes it, and that is a diagonal event line in "
            "(dx, dy), not any vertex's offset. The consequence is not theoretical:",
            s["body"]),
        code("36×27 room tilted 23°, 3×3 cells\n"
             "  optimize_exact       83 complete\n"
             "  dense 120×120 sweep  84 complete   <- attainable\n"
             "  optimize_columns     84 complete",
             s["mono"]),
        para(
            "The column solver reaches it because it solves dy as a continuum "
            "instead of sampling vertex-derived offsets. dx stayed enumerated "
            "over the critical set, however, so this fixed the instance and not "
            "the class — the same diagonal events are as invisible to a vertex "
            "offset in one axis as in two. A trapezoid exhibits the other half:",
            s["body"]),
        code("trapezoid (0,0) (34,0) (28,22) (5,22), 3×3 cells\n"
             "  optimize_columns     59 complete\n"
             "  optimize_erosion     60 complete   <- proven optimal",
             s["mono"]),
        para(
            "“Proven” rather than “found”: the erosion "
            "solver's overlap depth is the complete count over every offset at "
            "once, so 60 is not the best of a search but a bound no placement of "
            "that grid on that region can beat. Two further instances in the "
            "hunt — a pentagon at 76 against 77, a kite at 57 against 58 — behave "
            "the same way. The claim that survives is therefore stronger than "
            "dominance: translation is exact for any shape, and each of the two "
            "earlier searches is exact only on the strictly smaller class the "
            "next one subsumes.",
            s["body"]),

        para("F4. The local refine rarely earns its cost, and is now free when it "
             "does not", s["finding"]),
        para(
            "Across the tilted rooms and the tilted L, turning the refine on "
            "multiplied evaluations severalfold and changed the complete "
            "count on none of them — 5 evaluations against 12 on the "
            "tilted room at 23°. The one instance in the suite where it does change the "
            "answer is a parallelogram, whose two wall families are not "
            "perpendicular so that no angle flushes both and the optimum is a "
            "compromise the candidates straddle. The recoverable-area stop keeps "
            "the refine available for that case while removing its cost from the "
            "cases that do not need it.",
            s["body"]),

        para("F5. The case against golden-section weakened when translation "
             "improved", s["finding"]),
        para(
            "The note prescribes a golden-section refine, which assumes a "
            "unimodal continuous objective; the complete count is integer-valued "
            "and piecewise-constant, so on a plateau the bracket contracts on a "
            "comparison between equal values. While translation was enumerated, "
            "golden-section actually lost on the parallelogram — 5 against "
            "uniform sampling's 6 — because the placement it converged on "
            "could not be rescued by the offsets that search could see. With the "
            "dy axis solved, the same angle now yields 6. The honest claim is "
            "therefore no longer “it loses” but “it does not "
            "win”, which still leaves uniform sampling as the default.",
            s["body"]),

        para("F6. Removing the enumeration is cheaper and better at once",
             s["finding"]),
        para(
            "Not a trade. On the same corpus, the full pipeline costs 4 "
            "evaluations on average; enumerating the offset arrangement instead "
            "costs 704 and is worse on one instance, and enumerating only dx "
            "costs 38. The gains concentrate exactly where the old cost was "
            "quadratic — boundaries with many distinct vertex coordinates. The "
            "disc is the clearest single case: 1 evaluation against 30 against "
            "900, for a better result than the last of them.",
            s["body"]),

        para("F9. The corpus does not contain the instance that motivated the "
             "second axis", s["finding"]),
        para(
            "Reported because it qualifies the headline. Solving dx as well as dy "
            "changes the complete count on none of the sixteen corpus instances "
            "— the dx-enumerating ablation ties the full method everywhere here — "
            "so the corpus alone would not have justified the work. The "
            "counterexamples were found by searching for them, and are pinned in "
            "the test suite rather than in the corpus. What the corpus does "
            "measure is the other two consequences, and both are real: cost falls "
            "roughly ninefold at equal quality, and the partial count improves "
            "where the complete count cannot. On the disc, solving both axes "
            "returns 62 complete and 38 partial against the dx-enumerating "
            "solver's 62 and 39, at one evaluation against thirty.",
            s["body"]),
        para(
            "So the claim the measurements support is not “it finds more "
            "cells on this corpus”. It is that it cannot find fewer, it "
            "costs a fraction as much, and it is the only one of the three whose "
            "exactness does not depend on which way the walls happen to run.",
            s["body"]),

        para("F10. The vote was right every time — and now that is a theorem",
             s["finding"]),
        para(
            "Certifying the whole quick corpus closes the gap on all sixteen "
            "instances: every one returns a bound equal to what the pipeline "
            "achieved, so on none of them does any placement of that grid at any "
            "angle do better. All eight instances with an optimum proven by "
            "construction are reached, and the other eight — curved, random and "
            "image-traced, where no optimum was known — are now settled too, "
            "which is the first time anything in this work has been able to say "
            "so about them.",
            s["body"]),
        para(
            "The cost profile is the interesting part. Thirteen instances close "
            "in 15 windows, because the incumbent the vote supplies is already "
            "optimal and the search only has to confirm it. The exceptions are "
            "the curved ones: a disc costs 239 windows, since the count barely "
            "varies with angle, so nothing prunes on quality and the split runs "
            "down to the tolerance. 522 windows and 862 seconds for the corpus.",
            s["body"]),
        para(
            "This also retires a hedge. Section F4 argued the refine rarely earns "
            "its cost from the fact that no instance measured showed otherwise, "
            "which is an argument from absence. On this corpus the answer without "
            "the refine is now provably optimal, so on these instances the refine "
            "cannot earn anything — there is nothing left for it to find. The "
            "parallelogram of F4 remains the case where it does, and it is not in "
            "the corpus.",
            s["body"]),

        para("F11. Certifying the corpus found a crash nothing else did",
             s["finding"]),
        para(
            "Reported because of how it hid rather than because it was hard to "
            "fix. Widening a region for an angular window shrinks its holes, and "
            "at one window on one instance a 3-wide pillar shrank to 0.8 — "
            "narrower than the sweep that computes the erosion. Sweeping that "
            "hole's ring closed it onto itself, and the union of the swept "
            "parallelograms came back self-touching: a polygon GEOS produces "
            "willingly and then refuses to consume, failing the next difference "
            "with “unable to assign free hole to a shell”.",
            s["body"]),
        para(
            "A crash rather than a wrong answer, on instances with obstacles "
            "only, at one window out of many. 317 tests did not reach it; running "
            "the certificate over the corpus did, on the ninth instance. Two "
            "attempts at a regression test for it PASSED against the unfixed "
            "code, because they approximated the geometry instead of reproducing "
            "it — a plain buffer rather than the widening the bound applies, and "
            "the region swept rather than its complement. A test that cannot fail "
            "is worse than no test, since it reports the bug as fixed.",
            s["body"]),

        para("F7. The certificate answers where a known optimum cannot",
             s["finding"]),
        para(
            "On the instances with a proven optimum the method reaches it, so the "
            "certificate is redundant there. Its value is on the rest — "
            "curved, random and image-traced instances — where nothing else "
            "bounds the result. Its assumption holds on most of the corpus and it "
            "reports its own failure on the remainder rather than quoting a gap "
            "that does not hold.",
            s["body"]),
        para(
            "The rotation certificate of section 3.5 does not replace it, and "
            "the two should not be read as one number. This one bounds the "
            "PARTIAL count from below by a covering argument that can fail to "
            "apply, and declines on 19% of the corpus when it does. That one "
            "bounds the COMPLETE count from above by an argument that holds for "
            "any shape, and never declines — it either closes or reports the gap "
            "it could not close. Where both are available they are independent "
            "statements about the same placement.",
            s["body"]),

        para("F8. A duplicated cost counter in the test suite", s["finding"]),
        para(
            "Adding an evaluation counter to the packer collided with a counting "
            "subclass a test module kept for the same purpose, which then counted "
            "every evaluation twice — 72 against 36. The subclass was "
            "retired in favour of the packer's own counter. Two counters are two "
            "chances to be wrong about the number the paper reports as cost.",
            s["body"]),

        para("F14. The partial floor does not need its assumption after all",
             s["finding"]),
        para(
            "F1 corrected section 9.1's covering bound; this replaces it where "
            "it matters. The bound divides boundary forced through cell "
            "interiors by the most one cell can hold, taken to be the cell "
            "diagonal — which is true only when each cell carries a single "
            "straight crossing, so the certificate has to measure that and "
            "decline when it fails, on 9 of the 72 instances. Where it does not "
            "decline it is frequently vacuous: on a 13×10 room at 3×3 it returns "
            "a floor of 0 while every placement whatsoever leaves 8 partials.",
            s["body"]),
        para(
            "The same construction that solves the complete count answers this "
            "outright. A cell is complete when its corner lies in the region "
            "eroded by the cell, and meets the region at all when its corner "
            "lies in the region dilated by it, so",
            s["body"]),
        code("partial(dx, dy)  =  |lattice in the dilation|  -  |lattice in the erosion|",
             s["mono"]),
        para(
            "and folding both modulo the lattice leaves one piecewise-constant "
            "function on one torus to minimise. On that 13×10 room it returns 8.",
            s["body"]),
        para(
            "One detail decides whether it is a floor at all, and a first "
            "attempt got it backwards. For the complete count the pieces are "
            "closed, depth is upper semi-continuous, and a MAXIMUM therefore "
            "survives at a vertex of the arrangement — which is why that search "
            "reads off vertices alone. A minimum does not: it can sit strictly "
            "inside a face, so sampling vertices returns a value ABOVE the true "
            "minimum, which is the wrong side for a floor and would certify a "
            "result that is not optimal. Every cell of the arrangement is "
            "sampled instead — faces by a representative point, edges by a "
            "midpoint, vertices themselves — and the two tolerances are pushed "
            "in opposite directions so the answer can only fall short.",
            s["body"]),
        para(
            "It is reported beside the covering floor rather than replacing it, "
            "because the two are different claims: this one needs no assumption "
            "but speaks for one angle, and that one speaks for every angle but "
            "can fail to apply. Merging them would take the weaker scope from "
            "one and the weaker guarantee from the other.",
            s["body"]),

        para("F12. The full corpus agrees with the quick one, including where "
             "that was inconvenient", s["finding"]),
        para(
            "72 instances against 18 methods, 1296 rows, 60 minutes. It was run "
            "because the quick corpus is 16 instances and a conclusion drawn "
            "from 16 is a conclusion about 16.",
            s["body"]),
        para(
            "It confirms F9 at four times the size, which is the finding it "
            "would have been most convenient to overturn: solving dx as well as "
            "dy changes the complete count on NONE of the 72 instances — the "
            "dx-enumerating ablation is worse on zero and better on zero. The "
            "counterexamples that motivated the work are still only the ones "
            "found by hunting for them. What the corpus does confirm is the "
            "cost, at 4 evaluations against 29, and it now also shows the "
            "erosion solver beating the full offset enumeration on quality: both "
            "disc instances come back 61 against 62.",
            s["body"]),
        para(
            "The partial-cell certificate's assumption holds a little more often "
            "here than on the quick corpus, 63 of 72 against 13 of 16, and it "
            "fails on exactly the shapes the explanation predicts: traced "
            "outlines, a stadium, and off-grid pillars — boundaries with detail "
            "finer than a cell.",
            s["body"]),

        para("F13. The certificate settles the vote weight, and the default "
             "changes", s["finding"]),
        para(
            "The vote weights each chord by w = L · g(f), with f the inside "
            "fraction of the cell it came from. Which g is right was not "
            "previously measurable: the candidates disagree only about where a "
            "knife edge falls — five thousandths of a degree on the traced "
            "tilted L — and nothing here could say which side was correct. The "
            "corpus could only report that they differed.",
            s["body"]),
        para(
            "Certifying all 72 instances answers it, because the certificate "
            "names the optimum rather than the best anything found. Every one "
            "closed:",
            s["body"]),
        code("g(f) = f     reaches the proven optimum on 65 of 72\n"
             "g(f) = f^2   reaches it on 67 of 72, worse on none, same cost",
             s["mono"]),
        para(
            "So the default is now f². Three wins on one traced outline would be "
            "thin evidence for a new parameter; it is enough for this one "
            "because the comparison is a dominance across all seven families "
            "rather than a mean — switching costs nothing anywhere measured. The "
            "note's linear weight is retained as the ablation that measures the "
            "change.",
            s["body"]),

        para("F15. The vote misses the optimum more often than the corpus could "
             "show", s["finding"]),
        para(
            "The headline table reports the method reaching every proven optimum, "
            "36 of 36. That counts only instances whose optimum is known BY "
            "CONSTRUCTION, which is half the corpus and all of it rectilinear or "
            "rigidly tilted. Certifying the other half changes the picture: "
            "against the proven optimum the shipped pipeline misses on 7 of 72, "
            "and the new default still misses on 5.",
            s["body"]),
        code("plan-seed1-2x3   bound 60   f 58   f^2 58\n"
             "plan-seed3-2x3   bound 54   f 52   f^2 52\n"
             "plan-seed4-2x3   bound 50   f 48   f^2 48\n"
             "plan-seed5-2x3   bound 52   f 50   f^2 50\n"
             "traced-l23-2x3   bound 83   f 81   f^2 82",
             s["mono"]),
        para(
            "Both weights miss the same four random plans by two cells each, and "
            "every one of them is at 2×3 cells. That is not a weighting problem: "
            "it is the vote doing worse with rectangular cells, where a wall "
            "flush with the cell width and the same wall flush with its height "
            "are different placements and the vote must choose. No instance with "
            "square cells misses at all.",
            s["body"]),
        para(
            "The certificate found those placements itself, seeded with the "
            "vote's answer — so this is not only a measurement of the gap but a "
            "way to close it, at the cost of the proof. It also shows what the "
            "known-optimum families could not: they are exactly the instances "
            "the vote finds easy.",
            s["body"]),
    ]

    # ---------------------------------------------------------------- 5
    story += [PageBreak(), para("5. Measurements", s["h1"])]

    if rows:
        story += [
            para(
                f"{corpus_sentence(rows)} Quality is reported as the mean "
                "shortfall against the best result any method reached on the "
                "same instance, so it is comparable across instances of "
                "different size. “Known hit” counts instances whose "
                "optimum is proven by construction and was reached.",
                s["body"]),
            table(per_method_rows(rows),
                  [30 * mm, 20 * mm, 22 * mm, 14 * mm, 20 * mm, 20 * mm,
                   W - 126 * mm],
                  align_right=(2, 3, 5, 6)),
            para(headline_caption(rows), s["caption"]),
        ]

        story += [
            para("5.1 Where the old cost lived", s["h2"]),
            table(instance_rows(
                rows,
                ["disc-3x3", "room-tilt23-3x3", "plan-seed0-3x3"],
                ["uniform-s10", "fixed15-s10", "exact", "columns", "erosion",
                 "guided"]),
                [40 * mm, 30 * mm, 22 * mm, 26 * mm, W - 118 * mm],
                align_right=(2, 3, 4)),
            para("Table 6. Per-instance. The disc is where the enumeration was "
                 "quadratic — 900 evaluations for 61 cells, against 1 for 62 — "
                 "and the tilted room is where it was also wrong. On the "
                 "rectilinear plan every exact method is already cheap, and "
                 "solving both axes still reduces it to a single evaluation.",
                 s["caption"]),
        ]

        summary = certificate_summary(rows)
        if summary:
            story += [
                para("5.2 What the certificate says", s["h2"]),
                code("\n".join(summary), s["mono"]),
                para("Figure 1. Certificate over the corpus. A gap of zero is the "
                     "strongest per-instance statement available: no placement of "
                     "that grid on that region has fewer partial cells.",
                     s["caption"]),
            ]
    else:
        story += [para("No results CSV found; run "
                       "<font face='Courier'>python -m evaluation.run</font> "
                       "first.", s["body"])]

    story += [
        para("5.3 Complexity", s["h2"]),
        table([
            ["search", "evaluations", "per evaluation", "exact?"],
            ["uniform sweep", "steps² per angle", "O(N)", "no — resolution-bound"],
            ["critical-offset enumeration", "Vx · Vy per angle", "O(N)", "rectilinear only"],
            ["column solver", "Vx per angle", "O(columns · boundary)", "exact in dy; dx as above"],
            ["erosion solver", "1 per angle", "O(N)", "yes — any shape"],
        ], [44 * mm, 34 * mm, 38 * mm, W - 116 * mm]),
        para("Table 7. Vx, Vy are the counts of distinct vertex coordinates "
             "modulo the cell size; N is the number of cells over the bounding "
             "box.", s["caption"]),

        para("5.4 What the rotation certificate says", s["h2"]),
        para(
            "Every instance in the quick corpus, certified: the bound over all "
            "angles against what the pipeline achieved. A gap of zero with the "
            "space closed is the strongest statement in this document — not "
            "“the best anything here found”, but “no placement "
            "of this grid on this region, at any angle, holds more cells”.",
            s["body"]),
        table([
            ["instance", "complete", "bound", "windows", "seconds"],
            ["room-aligned-3x3", "108", "108", "15", "25"],
            ["room-offset-3x3", "108", "108", "15", "23"],
            ["room-tilt12-3x3", "108", "108", "15", "32"],
            ["room-tilt23-3x3", "108", "108", "15", "65"],
            ["l-shape-3x3", "88", "88", "15", "17"],
            ["u-shape-3x3", "96", "96", "15", "24"],
            ["plus-3x3", "80", "80", "15", "17"],
            ["l-shape-tilt12-3x3", "88", "88", "15", "23"],
            ["room-pillars-3x3", "101", "101", "15", "26"],
            ["room-pillars-offgrid-3x3", "91", "91", "30", "45"],
            ["disc-3x3", "62", "62", "239", "117"],
            ["stadium-3x3", "80", "80", "27", "67"],
            ["plan-seed0-3x3", "68", "68", "15", "53"],
            ["plan-seed1-3x3", "96", "96", "15", "61"],
            ["traced-l0-3x3", "88", "88", "15", "78"],
            ["traced-l23-3x3", "77", "77", "46", "190"],
            ["all sixteen", "—", "gap 0", "522", "862"],
        ], [46 * mm, 22 * mm, 20 * mm, 22 * mm, W - 110 * mm],
            align_right=(1, 2, 3, 4)),
        para("Table 8. 16 of 16 certified globally optimal, and all 8 proven "
             "optima reached. Curved boundaries cost the most windows: the count "
             "barely varies with angle, so nothing prunes on quality and the "
             "split runs to the tolerance.", s["caption"]),
    ]

    # ---------------------------------------------------------------- 6
    story += [
        PageBreak(),
        para("6. How this was verified", s["h1"]),
        para(
            "Every commit was tested at its own state rather than only at the "
            "end. That caught two breakages the end-state run surfaced: the "
            "duplicated counter of F8, and a test asserting the refine costs four "
            "times the vote, which the new stop makes false by design on that "
            "instance. Both were fixed in the commit that caused them.",
            s["body"]),
        table([
            ["claim", "how it is checked"],
            ["the column solver agrees where the enumeration is exact",
             "equality of the optimum on five rectilinear instances and three cell aspect ratios"],
            ["it beats the enumeration where that is not exact",
             "a 120×120 dense sweep arbitrates the tilted room; dominance asserted on five tilts and a disc"],
            ["the erosion solver attains its own upper bound",
             "achieved count asserted equal to the overlap depth on ten shapes and four cell aspects — a proof, not a comparison"],
            ["and the bound is not an artefact of the argument that produced it",
             "asserted at or above a dense sweep, and above the column solver, on seven non-rectilinear instances"],
            ["the fold keeps the flush placement",
             "a 36×27 room at 3×3 must return 108, which the zero-area pieces alone carry"],
            ["the angular bound really bounds its window",
             "a dense scan of the window is asserted under it, on a tilted room and on one with an obstacle"],
            ["it sees an optimum hiding inside a window",
             "a window centred 3° off a flush angle must still bound 108, or branch and bound would prune the interval holding the answer"],
            ["a certificate never claims more than it proved",
             "a deliberately tiny node budget must report exhausted=False and decline optimality while still holding a valid bound"],
            ["the 1-D lattice solve is exact",
             "compared against a 2000-point dense scan on five interval sets, including empty"],
            ["the placement reported really scores what is claimed",
             "the sweep chooses the placement, evaluate scores it, and the two are asserted equal"],
            ["the stop never costs a cell",
             "same complete count with and without, on four instances; cost bounded by its one probe"],
            ["the corpus's proven optima are real",
             "attained, including on tilted instances where attaining requires finding the rotation"],
            ["the corpus is reproducible",
             "built twice and compared geometry-wise; random plans are seeded"],
            ["cost is measured, not inferred",
             "the packer's counter is asserted equal to the number of placements a sweep returns"],
        ], [56 * mm, W - 56 * mm]),
        para("Table 9. The load-bearing assertions.", s["caption"]),

        para("6.1 How to reproduce", s["h2"]),
        code("cd backend\n"
             "python -m pytest tests -q                       # 333 tests\n"
             "python -m evaluation.run                        # quick corpus\n"
             "python -m evaluation.run --full                 # every tilt, cell, seed\n"
             "python -m evaluation.run --with-reference       # add the brute-force yardstick\n"
             "python -m evaluation.run --report results/*.json  # combine chunked runs\n"
             "python -m evaluation.report                     # this document",
             s["mono"]),
        para(
            "Table 8 is not produced by any of those. The certificate is a proof "
            "rather than a search, so putting it in the methods registry would "
            "file its cost in a column that means something else; it is run over "
            "the corpus directly instead, and until it has a column of its own "
            "the loop that produced those numbers is stated here in full:",
            s["body"]),
        code("from evaluation import corpus\n"
             "for inst in corpus.build(quick=True):\n"
             "    best, cert = inst.packer().certify_rotation()\n"
             "    print(inst.name, cert.complete, cert.bound, cert.optimal,\n"
             "          cert.nodes)",
             s["mono"]),
        para(
            "The corpus is generated from code, not shipped as data, so releasing "
            "the generator releases the corpus. Results land in "
            "backend/evaluation/results and are not tracked; a run takes minutes "
            "for the quick corpus.",
            s["body"]),
    ]

    # ---------------------------------------------------------------- 7
    story += [
        para("7. Limitations and what is left", s["h1"]),
        *bullets([
            "<b>The rotation certificate is opt-in, and priced accordingly.</b> "
            "It costs tens of seconds against under one to merely find the angle, "
            "so the request path does not run it. A curved boundary is its worst "
            "case: the count barely varies with angle, so nothing prunes on "
            "quality and the split runs down to the tolerance.",
            "<b>The computed partial floor holds at one angle, not all of "
            "them.</b> Partials are now minimised outright rather than bounded "
            "by the covering argument — see F14 — but that is a statement about "
            "the angle it was asked for. The covering bound remains the only one "
            "claiming to hold across rotation, and it is the one that declines. "
            "Extending the computed floor is the same branch and bound the "
            "complete count already uses: over a window the dilation grows and "
            "the erosion shrinks, which bounds their difference from below.",
            "<b>And it is not tight.</b> Exact on four of six shapes tried, one "
            "cell low on the other two, because the tolerance is pushed in the "
            "safe direction on both sides at once. A floor one below the truth "
            "is honest and still costs a cell of apparent gap.",
            "<b>A flat objective is the certificate's worst case, and the "
            "cost is arithmetic rather than a defect.</b> A 48-gon disc of "
            "radius 13 at 3×3 holds 45 cells at every angle, so no window is "
            "ever pruned on quality and the search must split until the bound "
            "falls on its own. It does, but only below a half-window of "
            "0.072° — 0.016 units of slack on a turning radius of 13 — which "
            "is 627 leaves. It closes at 1471 windows in 503s; the default "
            "budget of 600 stops it one cell short. What would help is a bound "
            "that tightens faster when the objective is flat, which is precisely "
            "when nothing else does.",
            "<b>The note's event-driven update was not implemented.</b> Section 5 "
            "offers it as optional, to remove the O(N) factor by recomputing only "
            "cells near a crossed vertex. Both later solvers removed that factor "
            "by other routes, and the erosion solver removed the offset loop it "
            "would have accelerated, so the motive is gone rather than deferred.",
            "<b>The erosion is polygon work, and pays for boundary complexity "
            "there instead.</b> Evaluations no longer grow with the number of "
            "vertices, but the erosion and the fold do: a 256-gon dilates 256 "
            "edges and folds into as many pieces as the region has cells. It is "
            "linear rather than quadratic, and it is not free.",
            "<b>The certificate can decline to certify.</b> When a single cell "
            "holds more boundary than its own diagonal the covering bound "
            "over-counts, and the certificate reports that rather than quoting a "
            "gap that does not hold.",
            "<b>The reference method is a lower bound too.</b> A dense sweep can "
            "step over a sharp optimum, which is the weakness the whole method "
            "exists to remove, so it is reported as the best anyone found and "
            "never as the optimum.",
            "<b>The vote still misses on rectangular cells.</b> See F15. "
            "Against the proven optimum the shipped pipeline now misses on 5 of "
            "72, every one of them at 2×3 cells and none at square ones, so it "
            "is a property of the vote's choice between two flush placements "
            "rather than of its weighting. The certificate names the placement "
            "that should have been found, which leaves the diagnosis settled and "
            "only the fix outstanding.",
            "<b>Step 5 of the roadmap is the paper.</b> Not code, not attempted "
            "here.",
        ], s["bullet"]),
    ]

    # ---------------------------------------------------------------- 8
    # Two calls rather than one with a separator: a commit subject may
    # contain any character, so there is no punctuation safe to split on.
    span = "47ef1a0..HEAD"
    hashes = git("log", "--format=%h", span).splitlines()
    subjects = git("log", "--format=%s", span).splitlines()
    if hashes and len(hashes) == len(subjects):
        entries = [[h, subject] for h, subject in zip(hashes, subjects)]
        story += [
            para("8. Commits", s["h1"]),
            table([["commit", "subject"]] + list(reversed(entries)),
                  [20 * mm, W - 20 * mm]),
            para("Table 10. This session's work, oldest first. Each was verified "
                 "against the suite at its own state before being pushed. The "
                 "commit carrying this rendered PDF cannot appear in a table "
                 "the PDF contains, so it is the one entry always absent.",
                 s["caption"]),
        ]

    doc = Doc(str(out_path))
    doc.build(story)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(args[0]) if args else DEFAULT_OUT
    path = build(out)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":      # pragma: no cover - entry point
    raise SystemExit(main())
