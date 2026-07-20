const API_BASE = import.meta.env.VITE_API_BASE || "";

export async function packPolygon(shape, obstacles, cellWidth, cellHeight, rotate) {
  const res = await fetch(`${API_BASE}/api/pack/polygon`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      shape,
      obstacles,
      cell_width: cellWidth,
      cell_height: cellHeight,
      rotate,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Packing failed");
  }
  return data;
}

export async function packImage(file, cellWidth, cellHeight, rotate) {
  const form = new FormData();
  form.append("file", file);
  form.append("cell_width", String(cellWidth));
  form.append("cell_height", String(cellHeight));
  form.append("rotate", String(rotate));

  const res = await fetch(`${API_BASE}/api/pack/image`, {
    method: "POST",
    body: form,
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Packing failed");
  }
  return data;
}
