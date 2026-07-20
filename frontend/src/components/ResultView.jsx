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
      </div>
    </div>
  );
}
