import { useRef, useState } from "react";

/**
 * Upload a plan, and optionally mark obstacles on it.
 *
 * Tracing finds what the plan draws as a hole. It cannot find a pillar the
 * survey missed, a stairwell added since, or an area someone has decided to
 * keep clear -- so those are marked here, by clicking their corners on the
 * image itself.
 *
 * THE Y AXIS IS THE THING TO GET RIGHT. Clicks arrive in display pixels with y
 * down; the tracer flips y while building the shape, so the backend wants the
 * same space it hands back. Both conversions happen at the moment of the click,
 * against the image's natural size, so nothing downstream has to remember which
 * way up a coordinate is.
 */
export default function ImageInput({ onFileSelected, onObstaclesChange }) {
  const [preview, setPreview] = useState(null);
  const [marking, setMarking] = useState(false);
  const [current, setCurrent] = useState([]);
  const [rings, setRings] = useState([]);
  const imgRef = useRef(null);

  function handleFile(file) {
    if (!file) return;
    onFileSelected(file);
    setPreview(URL.createObjectURL(file));
    // A new plan invalidates marks made on the old one.
    reset();
  }

  function reset() {
    setCurrent([]);
    setRings([]);
    onObstaclesChange?.([]);
  }

  function toImageSpace(event) {
    const img = imgRef.current;
    if (!img) return null;
    const box = img.getBoundingClientRect();
    // Displayed size differs from natural size whenever the preview is scaled
    // to fit, so a click has to be rescaled before it means anything.
    const x = ((event.clientX - box.left) / box.width) * img.naturalWidth;
    const yDown = ((event.clientY - box.top) / box.height) * img.naturalHeight;
    return [x, img.naturalHeight - yDown];
  }

  function addPoint(event) {
    if (!marking) return;
    event.preventDefault();
    const point = toImageSpace(event);
    if (point) setCurrent((points) => [...points, point]);
  }

  function closeRing() {
    if (current.length < 3) return;
    const next = [...rings, current];
    setRings(next);
    setCurrent([]);
    onObstaclesChange?.(next);
  }

  // Both of these exist because the only remedies used to be destructive:
  // close a shape you did not want, or clear everything and start again. A
  // mis-click is the ordinary case when placing corners by eye, so taking one
  // back should not cost the rest of the work.
  function undoPoint() {
    setCurrent((points) => points.slice(0, -1));
  }

  function removeLastRing() {
    const next = rings.slice(0, -1);
    setRings(next);
    onObstaclesChange?.(next);
  }

  const display = (point) => {
    const img = imgRef.current;
    if (!img) return { cx: 0, cy: 0 };
    return {
      cx: (point[0] / img.naturalWidth) * 100,
      cy: ((img.naturalHeight - point[1]) / img.naturalHeight) * 100,
    };
  };

  return (
    <div className="image-input">
      <label className="dropzone">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {preview ? (
          <span className="image-stage">
            <img
              ref={imgRef}
              src={preview}
              alt="Selected upload preview"
              className="image-preview"
              onClick={marking ? addPoint : undefined}
            />
            {(rings.length > 0 || current.length > 0) && (
              <svg className="image-overlay" viewBox="0 0 100 100"
                   preserveAspectRatio="none">
                {rings.map((ring, i) => (
                  <polygon
                    key={`ring-${i}`}
                    className="draw-obstacle"
                    points={ring.map((p) => {
                      const { cx, cy } = display(p);
                      return `${cx},${cy}`;
                    }).join(" ")}
                  />
                ))}
                {current.map((p, i) => {
                  const { cx, cy } = display(p);
                  return <circle key={`p-${i}`} cx={cx} cy={cy} r="0.8"
                                 className="draw-point" />;
                })}
              </svg>
            )}
          </span>
        ) : (
          <span>Click or drop an image</span>
        )}
      </label>

      {preview && (
        <div className="draw-toolbar">
          <button type="button" onClick={() => setMarking((on) => !on)}>
            {marking ? "Done marking" : "Mark obstacles"}
          </button>
          <button type="button" onClick={closeRing} disabled={current.length < 3}>
            Close obstacle
          </button>
          <button type="button" onClick={undoPoint} disabled={!current.length}>
            Undo point
          </button>
          <button type="button" onClick={removeLastRing} disabled={!rings.length}>
            Remove last
          </button>
          <button type="button" onClick={reset}
                  disabled={!rings.length && !current.length}>
            Clear
          </button>
          <span className="draw-hint">
            {marking
              ? "Click the corners of an area to keep clear, then close it."
              : `${rings.length} obstacle(s) marked`}
          </span>
        </div>
      )}
    </div>
  );
}
