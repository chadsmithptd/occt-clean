import tempfile
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceType,
    GeomAbs_Torus,
)
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shape, topods_Face
from OCC.Core.gp import gp_Dir, gp_Pnt

app = FastAPI(title="STEP File Analysis API", version="0.1.0")

INCH_PER_MM = 0.0393701


def _shape_from_step(data: bytes) -> TopoDS_Shape:
    with tempfile.NamedTemporaryFile(suffix=".step") as temp_file:
        temp_file.write(data)
        temp_file.flush()
        reader = STEPControl_Reader()
        status = reader.ReadFile(temp_file.name)
        if status != IFSelect_RetDone:
            raise ValueError("Failed to read STEP data")
        reader.TransferRoots()
        return reader.OneShape()


def _count_topology(shape: TopoDS_Shape) -> Dict[str, int]:
    counts = {"faces": 0, "edges": 0, "vertices": 0}
    explorer = TopExp_Explorer()
    explorer.Init(shape, TopAbs_FACE)
    while explorer.More():
        counts["faces"] += 1
        explorer.Next()
    explorer.Init(shape, TopAbs_EDGE)
    while explorer.More():
        counts["edges"] += 1
        explorer.Next()
    explorer.Init(shape, TopAbs_VERTEX)
    while explorer.More():
        counts["vertices"] += 1
        explorer.Next()
    return counts


def _bounding_box(shape: TopoDS_Shape) -> Dict[str, Any]:
    box = Bnd_Box()
    brepbndlib_Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return {
        "min": {"x": xmin * INCH_PER_MM, "y": ymin * INCH_PER_MM, "z": zmin * INCH_PER_MM},
        "max": {"x": xmax * INCH_PER_MM, "y": ymax * INCH_PER_MM, "z": zmax * INCH_PER_MM},
        "length": (xmax - xmin) * INCH_PER_MM,
        "width": (ymax - ymin) * INCH_PER_MM,
        "height": (zmax - zmin) * INCH_PER_MM,
    }


def _volume(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    return props.Mass() * (INCH_PER_MM ** 3)


def _surface_area(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    brepgprop_SurfaceProperties(shape, props)
    return props.Mass() * (INCH_PER_MM ** 2)


def _surface_area_by_type(shape: TopoDS_Shape) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        surface_type = adaptor.GetType()
        props = GProp_GProps()
        brepgprop_SurfaceProperties(face, props)
        area = props.Mass() * (INCH_PER_MM ** 2)
        key = _surface_type_label(surface_type)
        totals[key] = totals.get(key, 0.0) + area
        explorer.Next()
    return totals


def _surface_type_label(surface_type: GeomAbs_SurfaceType) -> str:
    mapping = {
        GeomAbs_Plane: "planar",
        GeomAbs_Cylinder: "cylindrical",
        GeomAbs_Cone: "conical",
        GeomAbs_Sphere: "spherical",
        GeomAbs_Torus: "toroidal",
        GeomAbs_BSplineSurface: "bspline",
        GeomAbs_BezierSurface: "bezier",
    }
    return mapping.get(surface_type, "other")


def _face_type_distribution(shape: TopoDS_Shape) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        surface_type = adaptor.GetType()
        key = _surface_type_label(surface_type)
        counts[key] = counts.get(key, 0) + 1
        explorer.Next()
    return counts


def _curvature_statistics(shape: TopoDS_Shape) -> Dict[str, float]:
    curvatures: List[float] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        umin, umax, vmin, vmax = breptools_UVBounds(face)
        u = (umin + umax) / 2.0
        v = (vmin + vmax) / 2.0
        props = BRepLProp_SLProps(adaptor, u, v, 2, 1e-6)
        if props.IsCurvatureDefined():
            curvatures.append(abs(props.MaxCurvature()))
            curvatures.append(abs(props.MinCurvature()))
        explorer.Next()
    if not curvatures:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    scale = 1 / INCH_PER_MM
    return {
        "min": min(curvatures) * scale,
        "max": max(curvatures) * scale,
        "avg": (sum(curvatures) / len(curvatures)) * scale,
    }


def _thin_wall_thickness(shape: TopoDS_Shape) -> float:
    faces: List[Tuple[gp_Dir, float, Tuple[float, float, float, float, float, float]]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        if adaptor.GetType() == GeomAbs_Plane:
            plane = adaptor.Plane()
            normal = plane.Axis().Direction()
            point = plane.Location()
            d = normal.X() * point.X() + normal.Y() * point.Y() + normal.Z() * point.Z()
            box = Bnd_Box()
            brepbndlib_Add(face, box)
            faces.append((normal, d, box.Get()))
        explorer.Next()
    min_dist = None
    for i, (n1, d1, b1) in enumerate(faces):
        for n2, d2, b2 in faces[i + 1 :]:
            if n1.IsParallel(n2, 1e-3) or n1.IsOpposite(n2, 1e-3):
                dist = abs(d1 - d2)
                if _bboxes_overlap(b1, b2):
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
    if min_dist is None:
        return 0.0
    return min_dist * INCH_PER_MM


def _bboxes_overlap(
    b1: Tuple[float, float, float, float, float, float],
    b2: Tuple[float, float, float, float, float, float],
) -> bool:
    return not (
        b1[3] < b2[0]
        or b1[0] > b2[3]
        or b1[4] < b2[1]
        or b1[1] > b2[4]
        or b1[5] < b2[2]
        or b1[2] > b2[5]
    )


def _detect_holes(shape: TopoDS_Shape) -> List[Dict[str, Any]]:
    holes: List[Dict[str, Any]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        if adaptor.GetType() == GeomAbs_Cylinder:
            cylinder = adaptor.Cylinder()
            radius = cylinder.Radius()
            axis = cylinder.Axis()
            umin, umax, vmin, vmax = breptools_UVBounds(face)
            depth = abs(vmax - vmin) * INCH_PER_MM
            holes.append(
                {
                    "diameter": 2 * radius * INCH_PER_MM,
                    "depth": depth,
                    "axis_direction": {
                        "x": axis.Direction().X(),
                        "y": axis.Direction().Y(),
                        "z": axis.Direction().Z(),
                    },
                    "centroid": {
                        "x": axis.Location().X() * INCH_PER_MM,
                        "y": axis.Location().Y() * INCH_PER_MM,
                        "z": axis.Location().Z() * INCH_PER_MM,
                    },
                    "is_through": None,
                }
            )
        explorer.Next()
    return holes


def _hole_diameters(holes: List[Dict[str, Any]]) -> List[float]:
    diameters = sorted({round(hole["diameter"], 4) for hole in holes})
    return diameters


def _feature_size_accessibility(shape: TopoDS_Shape) -> Dict[str, float]:
    stats = _curvature_statistics(shape)
    max_curv = stats["max"]
    if max_curv <= 0:
        return {">0.5in": 1.0, "0.25-0.5in": 0.0, "<0.25in": 0.0}
    min_radius_in = 1 / max_curv
    if min_radius_in >= 0.5:
        return {">0.5in": 1.0, "0.25-0.5in": 0.0, "<0.25in": 0.0}
    if min_radius_in >= 0.25:
        return {">0.5in": 0.0, "0.25-0.5in": 1.0, "<0.25in": 0.0}
    return {">0.5in": 0.0, "0.25-0.5in": 0.0, "<0.25in": 1.0}


def _detect_pockets(shape: TopoDS_Shape) -> List[Dict[str, Any]]:
    pockets: List[Dict[str, Any]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        if adaptor.GetType() == GeomAbs_Plane:
            umin, umax, vmin, vmax = breptools_UVBounds(face)
            u = (umin + umax) / 2.0
            v = (vmin + vmax) / 2.0
            props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
            if props.IsNormalDefined():
                normal = props.Normal()
                point = props.Value()
                offset_point = gp_Pnt(
                    point.X() + normal.X() * 1.0,
                    point.Y() + normal.Y() * 1.0,
                    point.Z() + normal.Z() * 1.0,
                )
                classifier = BRepClass3d_SolidClassifier(shape, offset_point, 1e-6)
                if classifier.State() == TopAbs_IN:
                    box = Bnd_Box()
                    brepbndlib_Add(face, box)
                    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
                    pockets.append(
                        {
                            "volume": None,
                            "depth": None,
                            "opening_area": (xmax - xmin) * (ymax - ymin) * (INCH_PER_MM ** 2),
                            "bounds": {
                                "length": (xmax - xmin) * INCH_PER_MM,
                                "width": (ymax - ymin) * INCH_PER_MM,
                                "height": (zmax - zmin) * INCH_PER_MM,
                            },
                        }
                    )
        explorer.Next()
    return pockets


def _metric_entry(
    key: str,
    display_name: str,
    description: str,
    unit: str,
    category: str,
    occt_extraction: str,
    feeds: List[str],
    value: Any,
) -> Dict[str, Any]:
    return {
        "key": key,
        "display_name": display_name,
        "description": description,
        "unit": unit,
        "category": category,
        "occt_extraction": occt_extraction,
        "feeds": feeds,
        "value": value,
    }


@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.lower().endswith((".step", ".stp")):
        raise HTTPException(status_code=400, detail="File must be a STEP file")
    data = await file.read()
    try:
        shape = _shape_from_step(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    counts = _count_topology(shape)
    bounding = _bounding_box(shape)
    volume = _volume(shape)
    area = _surface_area(shape)
    area_by_type = _surface_area_by_type(shape)
    face_types = _face_type_distribution(shape)
    curvature = _curvature_statistics(shape)
    holes = _detect_holes(shape)
    pockets = _detect_pockets(shape)
    thin_wall = _thin_wall_thickness(shape)
    feature_access = _feature_size_accessibility(shape)

    metrics = [
        _metric_entry(
            "bounding_box",
            "Bounding Box Dimensions",
            "Length × Width × Height and min/max coordinates",
            "in",
            "Geometry",
            "Bnd_Box from BRepBndLib::Add()",
            ["stock sizing", "fixturing", "stock volume estimation"],
            bounding,
        ),
        _metric_entry(
            "part_volume",
            "Part Volume",
            "Solid volume of the final part",
            "in^3",
            "Geometry",
            "BRepGProp::VolumeProperties",
            ["material estimation", "machining time"],
            volume,
        ),
        _metric_entry(
            "total_surface_area",
            "Total Surface Area",
            "Overall external + internal surface area",
            "in^2",
            "Geometry",
            "BRepGProp::SurfaceProperties",
            ["finishing", "coating"],
            area,
        ),
        _metric_entry(
            "surface_area_by_type",
            "Surface Area Breakdown",
            "Surface area per face type",
            "in^2",
            "Geometry",
            "BRepAdaptor_Surface per face",
            ["machining strategy"],
            area_by_type,
        ),
        _metric_entry(
            "topology_counts",
            "Number of Faces/Edges/Vertices",
            "Topological complexity metrics",
            "count",
            "Topology",
            "TopExp_Explorer",
            ["complexity scoring"],
            counts,
        ),
        _metric_entry(
            "face_type_distribution",
            "Face Type Distribution",
            "Count of planar vs sculptured/freeform surfaces",
            "count",
            "Topology",
            "BRepAdaptor_Surface::GetType",
            ["geometry classification"],
            face_types,
        ),
        _metric_entry(
            "curvature_statistics",
            "Curvature Statistics",
            "Max/average/min curvature sampled at face midpoints",
            "1/in",
            "Geometry",
            "BRepLProp_SLProps",
            ["surface finish", "classification"],
            curvature,
        ),
        _metric_entry(
            "thin_wall_thickness",
            "Thin Wall Thickness",
            "Minimum distance between parallel planar faces",
            "in",
            "Manufacturing",
            "Plane comparisons with Bnd_Box overlap",
            ["deflection risk", "fixturing"],
            thin_wall,
        ),
        _metric_entry(
            "feature_accessibility",
            "Feature Size / Tool Accessibility Distribution",
            "Heuristic distribution based on minimum curvature radius",
            "%",
            "Manufacturing",
            "Curvature-derived radius bins",
            ["tool selection"],
            feature_access,
        ),
        _metric_entry(
            "hole_diameters",
            "Hole Diameters",
            "Distinct diameters for detected holes",
            "in",
            "Manufacturing",
            "Cylindrical faces from BRepAdaptor_Surface",
            ["drilling"],
            _hole_diameters(holes),
        ),
        _metric_entry(
            "detected_holes",
            "Detected Holes",
            "Per-hole diameter, depth, through/blind, axis direction, centroid",
            "in",
            "Manufacturing",
            "Cylinder surface bounds + axis",
            ["drilling", "inspection"],
            {"count": len(holes), "items": holes},
        ),
        _metric_entry(
            "detected_pockets",
            "Detected Pockets / Cavities",
            "Count, per-pocket volume/depth/opening area/bounds",
            "in",
            "Manufacturing",
            "Planar faces classified with BRepClass3d_SolidClassifier",
            ["machining", "fixture"],
            {"count": len(pockets), "items": pockets},
        ),
    ]

    return JSONResponse({"metrics": metrics})


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
