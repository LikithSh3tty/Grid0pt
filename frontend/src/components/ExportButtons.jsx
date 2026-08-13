import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "";

// The formats the backend declares. Kept as a list so adding one here is the
// only change the UI needs.
const FORMATS = [
  ["csv", "CSV", "Cell centres and corners, for a spreadsheet or a script"],
  ["dxf", "DXF", "Layered drawing, for a CAD package"],
];

/**
 * Download the packed layout.
 *
 * Takes the REQUEST that produced the result on screen rather than the result
 * itself, and asks the server to pack it again in the wanted format. That
 * sounds wasteful and is not: the service caches by request, so the second ask
 * costs the serialisation only -- and it guarantees the file describes the same
 * placement as the drawing, which rebuilding a CSV here from the response would
 * not, since it would be a second implementation free to disagree.
 */
export default function ExportButtons({ request }) {
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null);

  if (!request) return null;

  async function download(format) {
    setError(null);
    setBusy(format);
    try {
      const res = await fetch(`${API_BASE}/api/export/polygon`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...request, format }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || "Export failed");
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `grid-layout.${format}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      // Surfaced rather than swallowed: a download that silently does nothing
      // is indistinguishable from one the browser blocked.
      setError(err.message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="export-row">
      <span className="export-label">Export</span>
      {FORMATS.map(([format, label, title]) => (
        <button
          key={format}
          type="button"
          className="export-button"
          title={title}
          disabled={busy !== null}
          onClick={() => download(format)}
        >
          {busy === format ? `${label}...` : label}
        </button>
      ))}
      {error && <span className="export-error">{error}</span>}
    </div>
  );
}
