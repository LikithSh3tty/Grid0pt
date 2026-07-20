import { useRef, useState } from "react";

const VIEW_SIZE = 100;

function pointFromEvent(svgEl, evt) {
  const pt = svgEl.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const loc = pt.matrixTransform(svgEl.getScreenCTM().inverse());
  return [Math.round(loc.x * 10) / 10, Math.round(loc.y * 10) / 10];
}

function polygonsToText(polygons) {
  return polygons.map((pts) => pts.map(([x, y]) => `${x},${y}`).join(" ")).join("\n");
}

function textToPolygons(text) {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const polygons = [];
  for (const line of lines) {
    const pts = line.split(/\s+/).map((tok) => {
      const [xs, ys] = tok.split(",");
      const x = parseFloat(xs);
      const y = parseFloat(ys);
      if (Number.isNaN(x) || Number.isNaN(y)) {
        throw new Error(`invalid point "${tok}"`);
      }
      return [x, y];
    });
    if (pts.length < 3) {
      throw new Error("each polygon needs at least 3 points");
    }
    polygons.push(pts);
  }
  return polygons;
}

export default function DrawCanvas({ polygons, onPolygonsChange }) {
  const svgRef = useRef(null);
  const [current, setCurrent] = useState([]);
  const [text, setText] = useState(polygonsToText(polygons));
  const [textError, setTextError] = useState(null);

  function syncText(next) {
    setText(polygonsToText(next));
    setTextError(null);
  }

  function handleCanvasClick(evt) {
    const [x, y] = pointFromEvent(svgRef.current, evt);
    setCurrent((prev) => [...prev, [x, y]]);
  }

  function closePolygon() {
    if (current.length < 3) return;
    const next = [...polygons, current];
    onPolygonsChange(next);
    syncText(next);
    setCurrent([]);
  }

  function undoPoint() {
    setCurrent((prev) => prev.slice(0, -1));
  }

  function clearAll() {
    setCurrent([]);
    onPolygonsChange([]);
    syncText([]);
  }

  function handleTextChange(evt) {
    const value = evt.target.value;
    setText(value);
    try {
      const parsed = textToPolygons(value);
      setTextError(null);
      onPolygonsChange(parsed);
    } catch (err) {
      setTextError(err.message);
    }
  }

  const allDrawn = [...polygons, current].filter((p) => p.length > 0);
  const xs = allDrawn.flat().map((p) => p[0]);
  const ys = allDrawn.flat().map((p) => p[1]);
  const minX = Math.min(0, ...xs);
  const minY = Math.min(0, ...ys);
  const maxX = Math.max(VIEW_SIZE, ...xs);
  const maxY = Math.max(VIEW_SIZE, ...ys);

  return (
    <div className="draw-panel">
      <svg
        ref={svgRef}
        viewBox={`${minX} ${minY} ${maxX - minX} ${maxY - minY}`}
        className="draw-svg"
        onClick={handleCanvasClick}
      >
        {polygons.map((pts, i) => (
          <polygon
            key={i}
            points={pts.map((p) => p.join(",")).join(" ")}
            className={i === 0 ? "draw-shape" : "draw-obstacle"}
          />
        ))}
        {current.length > 0 && (
          <polyline
            points={current.map((p) => p.join(",")).join(" ")}
            className="draw-current"
          />
        )}
        {current.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={1} className="draw-point" />
        ))}
      </svg>

      <div className="draw-toolbar">
        <button type="button" onClick={closePolygon} disabled={current.length < 3}>
          Close polygon
        </button>
        <button type="button" onClick={undoPoint} disabled={current.length === 0}>
          Undo point
        </button>
        <button type="button" onClick={clearAll}>Clear</button>
        <span className="draw-hint">
          {polygons.length === 0
            ? "Draw the shape first"
            : `${polygons.length - 1} obstacle(s) drawn`}
        </span>
      </div>

      <textarea
        className="draw-text"
        value={text}
        onChange={handleTextChange}
        rows={4}
        placeholder={"shape: x,y x,y x,y ...\nobstacle: x,y x,y x,y ..."}
      />
      {textError && <div className="text-error">{textError}</div>}
    </div>
  );
}
