from __future__ import annotations

import tempfile
from typing import Dict

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

from app.analysis import analyze_shape

app = FastAPI(title="STEP File Analysis API", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze_step(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.lower().endswith((".step", ".stp")):
        return JSONResponse(status_code=400, content={"error": "Expected a .step or .stp file"})

    with tempfile.NamedTemporaryFile(suffix=".step") as tmp:
        tmp.write(file.file.read())
        tmp.flush()

        reader = STEPControl_Reader()
        status = reader.ReadFile(tmp.name)
        if status != IFSelect_RetDone:
            return JSONResponse(status_code=400, content={"error": "Unable to read STEP file"})

        reader.TransferRoots()
        shape = reader.OneShape()

    result = analyze_shape(shape)
    return JSONResponse(content={"file_name": file.filename, **result})
