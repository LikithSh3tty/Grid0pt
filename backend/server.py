"""
server.py
=========

FastAPI app exposing the grid packer over HTTP, and serving the built React
frontend (../frontend/dist) as static files.

Run (from the backend/ folder): python server.py
Then open http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from packer_service import run_export, run_packing, run_packing_from_image

app = FastAPI(title="Grid Packer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PolygonRequest(BaseModel):
    shape: List[Tuple[float, float]]
    obstacles: List[List[Tuple[float, float]]] = []
    cell_width: float
    cell_height: float
    rotate: bool = False
    #: Prove the angle rather than vote for it. Adds the rotation_* statistics
    #: and costs tens of seconds, so it defaults off; see packer_service._solve.
    certify: bool = False


@app.post("/api/pack/polygon")
def pack_polygon(req: PolygonRequest):
    try:
        return run_packing(
            req.shape, req.obstacles, req.cell_width, req.cell_height,
            req.rotate, req.certify,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ExportRequest(PolygonRequest):
    #: "csv" for a spreadsheet or a script, "dxf" for a drawing package.
    format: str = "csv"


@app.post("/api/export/polygon")
def export_polygon(req: ExportRequest):
    """The packed layout as a file, rather than as JSON to redraw from.

    Served as an attachment: inline it opens as a wall of text in the browser,
    and the point of it is a file that goes to CAD.
    """
    try:
        text, media_type, filename = run_export(
            req.shape, req.obstacles, req.cell_width, req.cell_height,
            req.rotate, req.format, req.certify,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(
        content=text,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/pack/image")
async def pack_image(
    file: UploadFile = File(...),
    cell_width: float = Form(...),
    cell_height: float = Form(...),
    rotate: bool = Form(False),
    certify: bool = Form(False),
):
    image_bytes = await file.read()
    try:
        return run_packing_from_image(image_bytes, cell_width, cell_height,
                                      rotate, certify)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
