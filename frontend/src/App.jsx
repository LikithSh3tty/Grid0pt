import { useState } from "react";
import ImageInput from "./components/ImageInput";
import ResultView from "./components/ResultView";
import { packImage } from "./api";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [cellWidth, setCellWidth] = useState(40);
  const [cellHeight, setCellHeight] = useState(40);
  const [rotate, setRotate] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleRun() {
    setError(null);
    if (!file) {
      setError("Choose an image first.");
      return;
    }
    setLoading(true);
    try {
      const data = await packImage(file, Number(cellWidth), Number(cellHeight), rotate);
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
        <button onClick={handleRun} disabled={loading}>
          {loading ? "Running..." : "Run"}
        </button>
      </div>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button type="button" className="error-dismiss" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      <ImageInput onFileSelected={setFile} />

      <ResultView result={result} />
    </div>
  );
}
