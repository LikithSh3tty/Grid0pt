const API_BASE = import.meta.env.VITE_API_BASE || "";

// `certify` asks the backend to PROVE the angle rather than vote for it, which
// adds the rotation_* statistics to stats. It costs tens of seconds against well
// under one, and only does anything alongside `rotate`, so callers default it
// off -- see packer_service._solve.
export async function packPolygon(shape, obstacles, cellWidth, cellHeight, rotate,
                                  certify = false) {
  const res = await fetch(`${API_BASE}/api/pack/polygon`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      shape,
      obstacles,
      cell_width: cellWidth,
      cell_height: cellHeight,
      rotate,
      certify,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Packing failed");
  }
  return data;
}

export async function packImage(file, cellWidth, cellHeight, rotate,
                                certify = false) {
  const form = new FormData();
  form.append("file", file);
  form.append("cell_width", String(cellWidth));
  form.append("cell_height", String(cellHeight));
  form.append("rotate", String(rotate));
  form.append("certify", String(certify));

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
