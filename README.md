# STEP File Analysis API (OCCT)

## Co-planning checklist
This implementation is scoped to an API-only service for STEP ingestion and geometric extraction. Confirm the following before expanding feature fidelity:

1. **Units**: STEP files are assumed to be millimeters and converted to inches. If your data uses inches, set the conversion to 1.0.
2. **Feature detection tolerance**: Heuristic hole/pocket detection is used. If you need certified feature recognition, we should add a dedicated feature recognition pipeline.
3. **Tool accessibility**: Curvature sampling is used as a proxy. If you require voxelization or ray-casting, we can add that pass and GPU support.

## Feasibility notes (unfiltered)
- **Thin wall thickness**: Reliable for parallel planar faces; curved/organic thin walls require a signed distance field or medial axis computation.
- **Hole detection**: Cylindrical faces are robustly identified; complex hole blends and cross-holes need deeper topology analysis.
- **Pocket detection**: Current implementation flags planar faces that are inward-facing, which is a proxy. True cavity extraction requires boolean subtractions or feature graph analysis.

## Running locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API usage
POST `/analyze` with a `.step` or `.stp` file.

Example response structure:
```json
{
  "file_name": "part.step",
  "metrics": [
    {
      "key": "bounding_box",
      "display_name": "Bounding Box Dimensions",
      "description": "Length × Width × Height and min/max coordinates",
      "unit": "in",
      "category": "Geometry",
      "occt_extraction": "Bnd_Box from BRepBndLib::Add()",
      "feeds": ["stock sizing", "fixturing", "stock volume estimation"],
      "value": {
        "min": [0, 0, 0],
        "max": [1, 2, 3],
        "dimensions": [1, 2, 3]
      }
    }
  ],
  "units": {
    "length": "in",
    "area": "in^2",
    "volume": "in^3"
  },
  "notes": [
    "Units assume STEP geometry is authored in millimeters; converted to inches.",
    "Pocket and hole detection are heuristic and should be validated per part."
  ]
}
```

## Render.com deployment
This repo includes `render.yaml` and `requirements.txt`. Connect the GitHub repo to Render and deploy the web service. The OCCT Python bindings are installed from the upstream GitHub repo (https://github.com/tpaviot/pythonocc-core.git) as requested.

```
startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
