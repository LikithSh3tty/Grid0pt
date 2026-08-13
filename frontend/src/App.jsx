import { useState } from "react";
import ImageInput from "./components/ImageInput";
import DrawCanvas from "./components/DrawCanvas";
import ResultView from "./components/ResultView";
import { packImage, packPolygon } from "./api";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState("image");
  const [file, setFile] = useState(null);
  const [polygons, setPolygons] = useState([]);
  const [cellWidth, setCellWidth] = useState(40);
  const [cellHeight, setCellHeight] = useState(40);
  const [rotate, setRotate] = useState(false);
  const [certify, setCertify] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // The proof only exists over angles, so asking for it without rotation would
  // buy nothing and cost tens of seconds. The checkbox below is disabled in
  // that state; this makes the request agree with what the box shows, so a
  // stale tick left over from a previous run cannot quietly pay for a proof
  // that would not be produced.
  const proving = rotate && certify;

  async function handleRun() {
    setError(null);
    setLoading(true);
    try {
      let data;
      if (mode === "image") {
        if (!file) throw new Error("Choose an image first.");
        data = await packImage(file, Number(cellWidth), Number(cellHeight),
                               rotate, proving);
      } else {
        if (polygons.length === 0) throw new Error("Draw or enter a shape first.");
        const [shape, ...obstacles] = polygons;
        data = await packPolygon(shape, obstacles, Number(cellWidth),
                                 Number(cellHeight), rotate, proving);
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <h1>Grid Packer</h1>

      <div className="tabs">
        <button
          type="button"
          className={mode === "image" ? "tab active" : "tab"}
          onClick={() => setMode("image")}
        >
          Image
        </button>
        <button
          type="button"
          className={mode === "draw" ? "tab active" : "tab"}
          onClick={() => setMode("draw")}
        >
          Draw
        </button>
      </div>

      <div className="controls">
        <label>
          Cell width
          <input type="number" min="0.01" step="any" value={cellWidth}
                 onChange={(e) => setCellWidth(e.target.value)} />
        </label>
        <label>
          Cell height
          <input type="number" min="0.01" step="any" value={cellHeight}
                 onChange={(e) => setCellHeight(e.target.value)} />
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={rotate}
                 onChange={(e) => setRotate(e.target.checked)} />
          Allow rotation
        </label>
        <label
          className={rotate ? "checkbox-label" : "checkbox-label disabled"}
          title={
            rotate
              ? "Prove no placement at any angle fits more cells. Takes tens of seconds."
              : "Needs rotation: without it there is no angle to prove, and the offsets are already solved exactly."
          }
        >
          <input type="checkbox" checked={proving} disabled={!rotate}
                 onChange={(e) => setCertify(e.target.checked)} />
          Prove optimal
        </label>
        <button onClick={handleRun} disabled={loading}>
          {loading ? (proving ? "Proving..." : "Running...") : "Run"}
        </button>
      </div>

      {proving && !loading && (
        <p className="hint">
          Proving searches every angle rather than trusting the one the walls
          voted for. Expect tens of seconds; curved outlines take longest.
        </p>
      )}

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button type="button" className="error-dismiss" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {mode === "image" ? (
        <ImageInput onFileSelected={setFile} />
      ) : (
        <DrawCanvas polygons={polygons} onPolygonsChange={setPolygons} />
      )}

      <ResultView result={result} />
    </div>
  );
}
