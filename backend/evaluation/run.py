"""The evaluation driver (design note section 11).

Runs every method over every instance and writes one row per pair. What each row
carries is section 11's metric list, and nothing is inferred that could be
measured:

  quality -- complete cells and coverage; and, where the instance has a PROVEN
             optimum, whether the method reached it.
  cost    -- evaluations (counted on the packer, not derived from loop bounds)
             and wall-clock seconds.
  bounds  -- the certificate's floor and the gap to it, so a result is placed
             against something absolute even when no optimum is known.

Usage, from the backend directory:

    python -m evaluation.run                      # quick: minutes
    python -m evaluation.run --full               # the paper run
    python -m evaluation.run --with-reference     # add the brute-force yardstick
    python -m evaluation.run --certify            # prove the optimum per instance
    python -m evaluation.run --methods guided,exact --families rotated
    python -m evaluation.run --report results/*.json   # combine earlier chunks

Output goes to `evaluation/results/` as CSV (one row per instance x method) and
JSON (the same rows plus the run's parameters, so a table can be regenerated
without re-running).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from evaluation import corpus as corpus_module
from evaluation import methods as methods_module
from evaluation.corpus import Instance
from evaluation.methods import Method

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class Row:
    """One (instance, method) measurement."""

    instance: str
    family: str
    method: str
    group: str
    cell_width: float
    cell_height: float

    complete: int
    partial: int
    coverage: float
    angle: float
    dx: float
    dy: float

    evaluations: int
    seconds: float

    known_optimum: Optional[int]
    reached_known_optimum: Optional[bool]
    complete_vs_best: Optional[int]        # filled in once every method has run

    irreducible: int
    partial_floor: int
    optimality_gap: int
    certified: bool
    recoverable_area: float

    resultant: Optional[float]
    rotated: Optional[bool]

    #: The rotation certificate. It belongs to the INSTANCE rather than to the
    #: method: a statement about that grid on that region, proved without
    #: reference to any search. So it is computed once per instance and copied
    #: onto every row of it, which is what makes `complete_vs_proven` possible --
    #: `complete_vs_best` can only say "nothing here did better", while this says
    #: how far from the best there could be.
    #:
    #: Defaulted, and last, so rows written before this existed still load: a
    #: results file is a complete measurement and re-running the corpus to read
    #: an old one would defeat the point of keeping it. None unless the run was
    #: asked to certify, which costs more than every method in the table put
    #: together.
    rotation_bound: Optional[int] = None
    rotation_optimal: Optional[bool] = None
    rotation_nodes: Optional[int] = None
    rotation_seconds: Optional[float] = None
    complete_vs_proven: Optional[int] = None


def measure(instance: Instance, method: Method) -> Row:
    """Run one method on one instance and read everything off the result.

    The packer is fresh per measurement so the evaluation counter belongs to
    this method alone, and the taxonomy is applied AFTER the timer stops: the
    certificate is a report about the answer, not part of producing it, and
    charging the search for it would misstate the cost the paper compares.
    """
    packer = instance.packer()

    start = time.perf_counter()
    best = method.run(packer)
    seconds = time.perf_counter() - start
    evaluations = packer.evaluations

    classified = packer.evaluate(best.dx, best.dy, best.angle, classify=True)
    certificate = packer.certificate(classified)
    vote = getattr(best, "rotation_vote", None)

    known = instance.known_optimum
    return Row(
        instance=instance.name,
        family=instance.family,
        method=method.name,
        group=method.group,
        cell_width=instance.cell_width,
        cell_height=instance.cell_height,
        complete=best.complete,
        partial=best.partial,
        coverage=round(best.coverage, 4),
        angle=round(best.angle, 4),
        dx=round(best.dx, 6),
        dy=round(best.dy, 6),
        evaluations=evaluations,
        seconds=round(seconds, 4),
        known_optimum=known,
        reached_known_optimum=None if known is None else best.complete >= known,
        complete_vs_best=None,
        irreducible=certificate.irreducible,
        partial_floor=certificate.floor,
        optimality_gap=certificate.gap,
        certified=certificate.certified,
        recoverable_area=round(certificate.recoverable_area, 4),
        resultant=None if vote is None else round(vote.resultant, 4),
        rotated=None if vote is None else vote.confident(),
    )


def certify_instance(instance: Instance, verbose: bool = True):
    """Prove the best any placement of this grid on this region could do.

    Once per INSTANCE, not per method: the bound is a property of the region and
    the cell, established without reference to any search, so charging it to a
    method or recomputing it per method would both misrepresent it.

    Returns the certificate and the seconds it took, or (None, 0.0) if the proof
    raised -- a certificate that cannot be produced must not take the results
    table down with it, since every measured row is still valid without it.
    """
    packer = instance.packer()
    start = time.perf_counter()
    try:
        _, certificate = packer.certify_rotation()
    except Exception as exc:                    # pragma: no cover - GEOS guard
        if verbose:
            print(f"  certificate failed on {instance.name}: {exc}")
        return None, 0.0
    return certificate, time.perf_counter() - start


def run(instances: Sequence[Instance], methods: Sequence[Method],
        verbose: bool = True, certify: bool = False) -> List[Row]:
    rows: List[Row] = []
    total = len(instances) * len(methods)
    done = 0

    for instance in instances:
        for method in methods:
            done += 1
            if verbose:
                print(f"[{done:4d}/{total}] {instance.name:<28s} {method.name}",
                      end="", flush=True)
            row = measure(instance, method)
            rows.append(row)
            if verbose:
                print(f"  complete={row.complete:4d} angle={row.angle:6.2f} "
                      f"ev={row.evaluations:6d} {row.seconds:7.2f}s")

    # "How far below the best result anyone got on this instance" -- the only
    # comparison available where no optimum is known by construction. Computed
    # after the fact because it needs every method's answer.
    best_per_instance: Dict[str, int] = {}
    for row in rows:
        best_per_instance[row.instance] = max(
            best_per_instance.get(row.instance, 0), row.complete)
    for row in rows:
        row.complete_vs_best = row.complete - best_per_instance[row.instance]

    if certify:
        # After the measurements, never during them: the proof must not appear
        # in any method's evaluation count or wall clock.
        by_instance: Dict[str, List[Row]] = {}
        for row in rows:
            by_instance.setdefault(row.instance, []).append(row)

        for instance in instances:
            if instance.name not in by_instance:
                continue
            if verbose:
                print(f"certifying {instance.name:<28s}", end="", flush=True)
            certificate, seconds = certify_instance(instance, verbose)
            if certificate is None:
                continue
            if verbose:
                print(f"  bound={certificate.bound:4d} "
                      f"optimal={certificate.optimal} "
                      f"windows={certificate.nodes:4d} {seconds:7.2f}s")
            for row in by_instance[instance.name]:
                row.rotation_bound = certificate.bound
                row.rotation_optimal = certificate.optimal
                row.rotation_nodes = certificate.nodes
                row.rotation_seconds = round(seconds, 4)
                # Signed like `complete_vs_best`: 0 means this method reached
                # the proven optimum, negative says by how much it fell short.
                row.complete_vs_proven = row.complete - certificate.bound

    return rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def per_method_table(rows: Sequence[Row]) -> str:
    """Aggregate by method: quality, cost, and how often it hit a known answer.

    Quality is reported as the mean shortfall against the best result any method
    reached on the same instance, so it is comparable across instances of wildly
    different size -- an absolute mean complete-cell count would just measure
    which instances are big.
    """
    by_method: Dict[str, List[Row]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)

    # "vs best" compares a method against whatever else happened to run, which
    # cannot tell "everything found the optimum" from "everything missed it by
    # the same amount". Where the run certified the instances there is an
    # absolute reference, so the stronger column is shown instead of only the
    # relative one -- and it is omitted entirely otherwise rather than filled
    # with a placeholder, so an uncertified run reads exactly as it always did.
    proven = any(r.complete_vs_proven is not None for r in rows)

    header = (f"{'method':<20s} {'group':<10s} {'mean-vs-best':>12s} "
              f"{'worst':>6s} {'known-hit':>10s} {'mean-ev':>9s} {'mean-s':>8s} "
              f"{'mean-gap':>9s}"
              + (f" {'mean-vs-optimal':>15s} {'at-optimal':>10s}" if proven else ""))
    lines = [header, "-" * len(header)]

    for name, group in sorted(by_method.items(),
                              key=lambda kv: (kv[1][0].group, kv[0])):
        shortfalls = [r.complete_vs_best for r in group
                      if r.complete_vs_best is not None]
        known = [r for r in group if r.reached_known_optimum is not None]
        hit = sum(1 for r in known if r.reached_known_optimum)
        certified = [r.optimality_gap for r in group if r.certified]
        line = (f"{name:<20s} {group[0].group:<10s} "
                f"{statistics.mean(shortfalls):12.2f} "
                f"{min(shortfalls):6d} "
                f"{(f'{hit}/{len(known)}' if known else '-'):>10s} "
                f"{statistics.mean(r.evaluations for r in group):9.0f} "
                f"{statistics.mean(r.seconds for r in group):8.2f} "
                f"{(statistics.mean(certified) if certified else float('nan')):9.2f}")
        if proven:
            against = [r.complete_vs_proven for r in group
                       if r.complete_vs_proven is not None]
            reached = sum(1 for d in against if d == 0)
            line += (f" {statistics.mean(against):15.2f} "
                     f"{f'{reached}/{len(against)}':>10s}") if against else \
                    f" {'-':>15s} {'-':>10s}"
        lines.append(line)

    return "\n".join(lines)


def ablation_table(rows: Sequence[Row]) -> str:
    """Each ablation against the full method, on the instances both ran.

    A component earns its place if switching it off costs quality, or if leaving
    it on costs less than it saves. Both directions are printed, because two of
    these components exist to save cost at no quality and would look like noise
    in a quality-only table.
    """
    full = {r.instance: r for r in rows if r.method == "guided"}
    if not full:
        return "(no `guided` rows to compare against)"

    by_method: Dict[str, List[Row]] = {}
    for row in rows:
        if row.group == "ablation":
            by_method.setdefault(row.method, []).append(row)

    header = (f"{'ablation':<20s} {'d-complete':>11s} {'instances-worse':>16s} "
              f"{'instances-better':>17s} {'d-evaluations':>14s} {'d-seconds':>10s}")
    lines = [header, "-" * len(header)]

    for name, group in sorted(by_method.items()):
        paired = [(r, full[r.instance]) for r in group if r.instance in full]
        if not paired:
            continue
        d_complete = [a.complete - f.complete for a, f in paired]
        lines.append(
            f"{name:<20s} "
            f"{statistics.mean(d_complete):11.2f} "
            f"{sum(1 for d in d_complete if d < 0):16d} "
            f"{sum(1 for d in d_complete if d > 0):17d} "
            f"{statistics.mean(a.evaluations - f.evaluations for a, f in paired):14.0f} "
            f"{statistics.mean(a.seconds - f.seconds for a, f in paired):10.2f}")

    return "\n".join(lines)


def certificate_table(rows: Sequence[Row]) -> str:
    """What the certificate says about the results the full method returned.

    A gap of 0 means no placement of that grid on that region has fewer partial
    cells -- the strongest statement available per instance, and one that holds
    over rotation too, which is the part the method cannot solve exactly.
    """
    guided = [r for r in rows if r.method == "guided"]
    if not guided:
        return "(no `guided` rows)"

    certified = [r for r in guided if r.certified]
    exact = [r for r in certified if r.optimality_gap == 0]
    lines = [
        f"instances                     : {len(guided)}",
        f"certificate assumption holds  : {len(certified)} "
        f"({100.0 * len(certified) / len(guided):.0f}%)",
        f"certified optimal (gap = 0)   : {len(exact)} of those {len(certified)}",
    ]
    if certified:
        gaps = [r.optimality_gap for r in certified]
        lines.append(f"gap: mean {statistics.mean(gaps):.2f}, max {max(gaps)}")
    return "\n".join(lines)


def rotation_table(rows: Sequence[Row]) -> str:
    """Per instance: the proven optimum, and which methods reached it.

    This is the only table here whose reference point is absolute. Every other
    one scores a method against the best result some other method happened to
    get, which cannot distinguish "everything found the optimum" from
    "everything missed it by the same amount".
    """
    certified = [r for r in rows if r.rotation_bound is not None]
    if not certified:
        return "(not certified: re-run with --certify)"

    by_instance: Dict[str, List[Row]] = {}
    for row in certified:
        by_instance.setdefault(row.instance, []).append(row)

    header = (f"{'instance':<28s} {'proven':>7s} {'verdict':>9s} "
              f"{'windows':>8s} {'seconds':>8s} {'methods at optimum':>20s}")
    lines = [header, "-" * len(header)]
    for name, group in sorted(by_instance.items()):
        first = group[0]
        reached = sum(1 for r in group if r.complete_vs_proven == 0)
        lines.append(
            f"{name:<28s} "
            f"{first.rotation_bound:7d} "
            f"{('proven' if first.rotation_optimal else 'open'):>9s} "
            f"{first.rotation_nodes:8d} "
            f"{first.rotation_seconds:8.2f} "
            f"{f'{reached}/{len(group)}':>20s}")

    proven = sum(1 for g in by_instance.values() if g[0].rotation_optimal)
    lines.append("")
    lines.append(f"instances certified globally optimal : "
                 f"{proven} of {len(by_instance)}")
    lines.append(f"windows / seconds total              : "
                 f"{sum(g[0].rotation_nodes for g in by_instance.values())} / "
                 f"{sum(g[0].rotation_seconds for g in by_instance.values()):.0f}")
    return "\n".join(lines)


def load_rows(paths: Sequence[Path]) -> List[Row]:
    """Re-read rows written by earlier runs.

    A full run does not have to happen in one process. The corpus splits
    cleanly by family and the expensive families are the ones with many
    boundary vertices, so running in chunks and reporting over the union is
    often the only practical way to get the whole table -- and it costs nothing,
    because a row is a complete measurement on its own.
    """
    rows: List[Row] = []
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload["rows"]:
            key = (raw["instance"], raw["method"])
            if key in seen:
                # Later files win: a re-run of a chunk supersedes the old one.
                rows = [r for r in rows if (r.instance, r.method) != key]
            seen.add(key)
            rows.append(Row(**raw))

    # The shortfall column is relative to the best result on each instance, so
    # it has to be recomputed over the union rather than trusted per chunk.
    best: Dict[str, int] = {}
    for row in rows:
        best[row.instance] = max(best.get(row.instance, 0), row.complete)
    for row in rows:
        row.complete_vs_best = row.complete - best[row.instance]

    return rows


def write_outputs(rows: Sequence[Row], meta: dict, stem: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{stem}.csv"
    json_path = RESULTS_DIR / f"{stem}.json"

    fields = list(asdict(rows[0]).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "rows": [asdict(r) for r in rows]}, fh, indent=2)

    return csv_path


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true",
                        help="the paper run: every tilt, cell geometry and seed")
    parser.add_argument("--with-reference", action="store_true",
                        help="also run the brute-force yardstick (slow)")
    parser.add_argument("--certify", action="store_true",
                        help="prove the optimum per instance and score every "
                             "method against it (slow: tens of seconds each)")
    parser.add_argument("--methods", default="",
                        help="comma-separated method names to keep")
    parser.add_argument("--families", default="",
                        help="comma-separated instance families to keep")
    parser.add_argument("--out", default="",
                        help="output file stem (default: quick / full)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--report", nargs="+", default=None, metavar="JSON",
                        help="run nothing: report over rows from earlier runs")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.report:
        paths = [Path(p) for p in args.report]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"no such results file: {missing[0]}", file=sys.stderr)
            return 2
        rows = load_rows(paths)
        print(f"{len(rows)} rows over "
              f"{len({r.instance for r in rows})} instances\n")
        print("== per method ==")
        print(per_method_table(rows))
        print("\n== ablations, against the full method ==")
        print(ablation_table(rows))
        print("\n== certificate ==")
        print(certificate_table(rows))
        print("\n== rotation certificate ==")
        print(rotation_table(rows))
        return 0

    quick = not args.full
    instances = corpus_module.build(quick=quick)
    methods = methods_module.build(quick=quick, with_reference=args.with_reference)

    if args.families:
        keep = {f.strip() for f in args.families.split(",")}
        instances = [i for i in instances if i.family in keep]
    if args.methods:
        keep = {m.strip() for m in args.methods.split(",")}
        methods = [m for m in methods if m.name in keep]

    if not instances or not methods:
        print("nothing to run: check --methods / --families", file=sys.stderr)
        return 2

    print(f"corpus ({'quick' if quick else 'full'}): {len(instances)} instances")
    print(corpus_module.summary(instances))
    print(f"methods: {len(methods)} -> {', '.join(m.name for m in methods)}\n")

    started = time.time()
    rows = run(instances, methods, verbose=not args.quiet,
               certify=args.certify)
    elapsed = time.time() - started

    print("\n== per method ==")
    print(per_method_table(rows))
    print("\n== ablations, against the full method ==")
    print(ablation_table(rows))
    print("\n== certificate ==")
    print(certificate_table(rows))
    if args.certify:
        print("\n== rotation certificate ==")
        print(rotation_table(rows))

    meta = {
        "quick": quick,
        "with_reference": args.with_reference,
        "certified": args.certify,
        "instances": len(instances),
        "methods": [m.name for m in methods],
        "seconds": round(elapsed, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    stem = args.out or ("quick" if quick else "full")
    path = write_outputs(rows, meta, stem)
    print(f"\n{len(rows)} rows in {elapsed:.0f}s -> {path}")
    return 0


if __name__ == "__main__":       # pragma: no cover - entry point
    raise SystemExit(main())
