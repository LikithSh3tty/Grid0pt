// The two certificates answer different questions and must not be run together
// into one "how good is this" number. The rotation one bounds the COMPLETE
// count from above over every angle, by an argument that holds for any shape.
// The partial one bounds the PARTIAL count from below by a covering argument
// that can fail to apply -- and says so, via `certified`.
function rotationVerdict(stats) {
  if (stats.rotation_optimal) {
    return {
      tone: "proven",
      headline: "Proven optimal",
      detail: `No placement of this grid on this region fits more than ${stats.rotation_bound} cells, at any angle.`,
    };
  }
  if (!stats.rotation_exhausted) {
    return {
      tone: "open",
      headline: "Not proven",
      detail: `The search stopped at its budget after ${stats.rotation_nodes} angle windows. At most ${stats.rotation_bound} cells are possible, so this is within ${stats.rotation_gap} of the best there could be.`,
    };
  }
  return {
    tone: "open",
    headline: `Within ${stats.rotation_gap} of optimal`,
    detail: `Every angle was ruled out down to a bound of ${stats.rotation_bound} cells, which this placement does not reach.`,
  };
}

function partialVerdict(stats) {
  if (!stats.certified) {
    return {
      tone: "open",
      headline: "Partial count not bounded here",
      detail:
        "This region puts more boundary inside a single cell than that cell's diagonal, which is the assumption the partial-cell floor rests on. No floor is quoted rather than one that does not hold.",
    };
  }
  if (stats.optimality_gap === 0) {
    return {
      tone: "proven",
      headline: "Fewest partial cells possible",
      detail: `${stats.partial} partial cells, and no placement of this grid on this region has fewer.`,
    };
  }
  return {
    tone: "open",
    headline: `Within ${stats.optimality_gap} partial cells of the floor`,
    detail: `${stats.partial} partial cells against a floor of ${stats.partial_floor}, of which ${stats.irreducible} are features too small for any placement to rescue.`,
  };
}

function Verdict({ verdict }) {
  return (
    <div className={`verdict ${verdict.tone}`}>
      <strong>{verdict.headline}</strong>
      <span>{verdict.detail}</span>
    </div>
  );
}

export default function ResultView({ result }) {
  if (!result) return null;

  const { shape, obstacles, complete_cells, partial_cells, stats } = result;

  const allPoints = [...shape, ...obstacles.flat(), ...complete_cells.flat(), ...partial_cells.flat()];
  const xs = allPoints.map((p) => p[0]);
  const ys = allPoints.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = maxX - minX;
  const height = maxY - minY;
  const pad = Math.max(width, height) * 0.05 || 1;

  const flipY = (y) => minY + maxY - y;
  const toPoints = (pts) => pts.map(([x, y]) => `${x},${flipY(y)}`).join(" ");

  return (
    <div className="result-view">
      <svg
        viewBox={`${minX - pad} ${minY - pad} ${width + pad * 2} ${height + pad * 2}`}
        className="result-svg"
      >
        <polygon points={toPoints(shape)} className="shape-outline" />
        {obstacles.map((ob, i) => (
          <polygon key={`ob-${i}`} points={toPoints(ob)} className="obstacle" />
        ))}
        {partial_cells.map((c, i) => (
          <polygon key={`pc-${i}`} points={toPoints(c)} className="partial-cell" />
        ))}
        {complete_cells.map((c, i) => (
          <polygon key={`cc-${i}`} points={toPoints(c)} className="complete-cell" />
        ))}
      </svg>
      <div className="stats-strip">
        <span>Complete: {stats.complete}</span>
        <span>Partial: {stats.partial}</span>
        <span>Coverage: {(stats.coverage * 100).toFixed(1)}%</span>
        <span>Offset: ({stats.dx.toFixed(2)}, {stats.dy.toFixed(2)})</span>
        <span>Angle: {stats.angle.toFixed(0)}°</span>
        {stats.evaluations !== undefined && (
          <span>Placements evaluated: {stats.evaluations}</span>
        )}
        {stats.rotation_nodes !== undefined && (
          <span>Angle windows: {stats.rotation_nodes}</span>
        )}
      </div>

      {stats.rotation_bound !== undefined && (
        <Verdict verdict={rotationVerdict(stats)} />
      )}
      {stats.partial_floor !== undefined && (
        <Verdict verdict={partialVerdict(stats)} />
      )}

      {stats.rotation_bound === undefined && (
        // Said plainly rather than left blank. Translation is exact on every
        // request, so the honest gap is about the ANGLE, and a reader who is
        // not told that will assume the result is proven when it is not.
        <p className="verdict-note">
          {stats.angle === 0
            ? "Offsets are solved exactly, so this is the best placement at this angle. Enable rotation to search angles as well."
            : "Offsets are solved exactly, so this is the best placement at this angle. The angle itself was read off the walls rather than proven — tick “Prove optimal” to settle it."}
        </p>
      )}
    </div>
  );
}
