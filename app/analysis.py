from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.BRepTools import UVBounds
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Pln, gp_Vec
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
)
from OCC.Core.TopExp import topexp_MapShapesAndAncestors

INCH_PER_MM = 1.0 / 25.4


@dataclass
class Metric:
    key: str
    display_name: str
    description: str
    unit: str
    category: str
    occt_extraction: str
    feeds: List[str]
    value: object

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "description": self.description,
            "unit": self.unit,
            "category": self.category,
            "occt_extraction": self.occt_extraction,
            "feeds": self.feeds,
            "value": self.value,
        }


def _collect_faces(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while explorer.More():
        faces.append(explorer.Current())
        explorer.Next()
    return faces


def _collect_edges(shape):
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edges = []
    while explorer.More():
        edges.append(explorer.Current())
        explorer.Next()
    return edges


def _collect_vertices(shape):
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    vertices = []
    while explorer.More():
        vertices.append(explorer.Current())
        explorer.Next()
    return vertices


def _face_surface_type(face):
    adaptor = GeomAdaptor_Surface(face)
    return adaptor.GetType()


def _face_area(face) -> float:
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    return props.Mass()


def _shape_volume(shape) -> float:
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    return props.Mass()


def _shape_surface_area(shape) -> float:
    props = GProp_GProps()
    brepgprop_SurfaceProperties(shape, props)
    return props.Mass()


def _bbox(shape) -> Tuple[float, float, float, float, float, float]:
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmin, ymin, zmin, xmax, ymax, zmax


def _planar_face_normal(face) -> Optional[gp_Dir]:
    adaptor = GeomAdaptor_Surface(face)
    if adaptor.GetType() != GeomAbs_Plane:
        return None
    surf = ShapeAnalysis_Surface(adaptor.Surface())
    umin, umax, vmin, vmax = UVBounds(face)
    u = (umin + umax) / 2.0
    v = (vmin + vmax) / 2.0
    vec = surf.Normal(u, v)
    if vec.Magnitude() == 0:
        return None
    return gp_Dir(vec)


def _face_point(face) -> gp_Pnt:
    adaptor = GeomAdaptor_Surface(face)
    umin, umax, vmin, vmax = UVBounds(face)
    u = (umin + umax) / 2.0
    v = (vmin + vmax) / 2.0
    surf = ShapeAnalysis_Surface(adaptor.Surface())
    return surf.Value(u, v)


def _min_parallel_planar_distance(faces: Iterable) -> Optional[float]:
    planar = []
    for face in faces:
        normal = _planar_face_normal(face)
        if normal is None:
            continue
        plane = gp_Pln(_face_point(face), normal)
        planar.append((face, plane, normal))
    if len(planar) < 2:
        return None
    min_dist = None
    for i in range(len(planar)):
        for j in range(i + 1, len(planar)):
            _, plane_a, normal_a = planar[i]
            _, plane_b, normal_b = planar[j]
            if abs(normal_a.Dot(normal_b)) < 0.95:
                continue
            dist = abs(plane_a.Distance(plane_b.Location()))
            if min_dist is None or dist < min_dist:
                min_dist = dist
    return min_dist


def _face_type_distribution(faces: Iterable) -> Dict[str, float]:
    distribution: Dict[str, float] = {
        "planar": 0.0,
        "cylindrical": 0.0,
        "conical": 0.0,
        "spherical": 0.0,
        "toroidal": 0.0,
        "bspline": 0.0,
        "other": 0.0,
    }
    for face in faces:
        area = _face_area(face)
        face_type = _face_surface_type(face)
        if face_type == GeomAbs_Plane:
            distribution["planar"] += area
        elif face_type == GeomAbs_Cylinder:
            distribution["cylindrical"] += area
        elif face_type == GeomAbs_Cone:
            distribution["conical"] += area
        elif face_type == GeomAbs_Sphere:
            distribution["spherical"] += area
        elif face_type == GeomAbs_Torus:
            distribution["toroidal"] += area
        elif face_type == GeomAbs_BSplineSurface:
            distribution["bspline"] += area
        else:
            distribution["other"] += area
    total = sum(distribution.values())
    if total == 0:
        return distribution
    return {key: value / total for key, value in distribution.items()}


def _curvature_stats(faces: Iterable) -> Dict[str, float]:
    curvatures = []
    for face in faces:
        adaptor = GeomAdaptor_Surface(face)
        umin, umax, vmin, vmax = UVBounds(face)
        u = (umin + umax) / 2.0
        v = (vmin + vmax) / 2.0
        props = GeomLProp_SLProps(adaptor.Surface(), u, v, 2, 1.0e-6)
        if not props.IsCurvatureDefined():
            continue
        k1 = props.MaxCurvature()
        k2 = props.MinCurvature()
        for k in (k1, k2):
            if isfinite(k):
                curvatures.append(abs(k))
    if not curvatures:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": float(np.min(curvatures)),
        "max": float(np.max(curvatures)),
        "avg": float(np.mean(curvatures)),
    }


def _tool_accessibility(curvature_stats: Dict[str, float]) -> Dict[str, float]:
    if curvature_stats["avg"] == 0:
        return {"large": 1.0, "medium": 0.0, "small": 0.0}
    radius = 1.0 / curvature_stats["avg"]
    if radius > 12.7:  # 0.5 in in mm
        return {"large": 0.7, "medium": 0.2, "small": 0.1}
    if radius > 6.35:  # 0.25 in in mm
        return {"large": 0.2, "medium": 0.6, "small": 0.2}
    return {"large": 0.1, "medium": 0.2, "small": 0.7}


def _hole_data(shape, faces: Iterable) -> List[Dict[str, object]]:
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    holes = []
    for face in faces:
        adaptor = GeomAdaptor_Surface(face)
        if adaptor.GetType() != GeomAbs_Cylinder:
            continue
        cylinder = adaptor.Cylinder()
        radius = cylinder.Radius()
        axis = cylinder.Axis().Direction()
        center = cylinder.Location()
        umin, umax, vmin, vmax = UVBounds(face)
        depth = abs(vmax - vmin) * radius
        neighbor_planars = 0
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            faces_list = edge_face_map.FindFromKey(edge)
            if faces_list.Extent() == 2:
                other_face = (
                    faces_list.First() if faces_list.First() != face else faces_list.Last()
                )
                other_type = _face_surface_type(other_face)
                if other_type == GeomAbs_Plane:
                    normal = _planar_face_normal(other_face)
                    if normal and abs(normal.Dot(axis)) > 0.9:
                        neighbor_planars += 1
            explorer.Next()
        is_through = neighbor_planars >= 2
        holes.append(
            {
                "diameter": 2 * radius * INCH_PER_MM,
                "depth": depth * INCH_PER_MM,
                "is_through": is_through,
                "axis_direction": [axis.X(), axis.Y(), axis.Z()],
                "position": [center.X() * INCH_PER_MM, center.Y() * INCH_PER_MM, center.Z() * INCH_PER_MM],
            }
        )
    return holes


def _pocket_data(shape, faces: Iterable) -> List[Dict[str, object]]:
    classifier = BRepClass3d_SolidClassifier(shape)
    pockets = []
    for face in faces:
        if _face_surface_type(face) != GeomAbs_Plane:
            continue
        normal = _planar_face_normal(face)
        if normal is None:
            continue
        point = _face_point(face)
        probe = gp_Pnt(
            point.X() + normal.X() * 0.1,
            point.Y() + normal.Y() * 0.1,
            point.Z() + normal.Z() * 0.1,
        )
        classifier.Perform(probe, 1.0e-6)
        if classifier.State() != 1:  # 1 == TopAbs_IN
            continue
        area = _face_area(face)
        xmin, ymin, zmin, xmax, ymax, zmax = _bbox(face)
        depth = max(xmax - xmin, ymax - ymin, zmax - zmin)
        pockets.append(
            {
                "volume": area * depth * (INCH_PER_MM ** 3),
                "depth": depth * INCH_PER_MM,
                "opening_area": area * (INCH_PER_MM ** 2),
                "bounding_dimensions": [
                    (xmax - xmin) * INCH_PER_MM,
                    (ymax - ymin) * INCH_PER_MM,
                    (zmax - zmin) * INCH_PER_MM,
                ],
            }
        )
    return pockets


def analyze_shape(shape) -> Dict[str, object]:
    shape_fixer = ShapeFix_Shape(shape)
    shape_fixer.Perform()
    fixed_shape = shape_fixer.Shape()
    unify = ShapeUpgrade_UnifySameDomain(fixed_shape, True, True, True)
    unify.Build()
    shape = unify.Shape()

    faces = _collect_faces(shape)
    edges = _collect_edges(shape)
    vertices = _collect_vertices(shape)

    bbox = _bbox(shape)
    bbox_dims = (bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])

    min_wall = _min_parallel_planar_distance(faces)
    face_distribution = _face_type_distribution(faces)
    curvature = _curvature_stats(faces)
    tool_access = _tool_accessibility(curvature)

    holes = _hole_data(shape, faces)
    pockets = _pocket_data(shape, faces)

    surface_area = _shape_surface_area(shape)
    volume = _shape_volume(shape)

    metrics = [
        Metric(
            key="bounding_box",
            display_name="Bounding Box Dimensions",
            description="Length × Width × Height and min/max coordinates",
            unit="in",
            category="Geometry",
            occt_extraction="Bnd_Box from BRepBndLib::Add()",
            feeds=["stock sizing", "fixturing", "stock volume estimation"],
            value={
                "min": [bbox[0] * INCH_PER_MM, bbox[1] * INCH_PER_MM, bbox[2] * INCH_PER_MM],
                "max": [bbox[3] * INCH_PER_MM, bbox[4] * INCH_PER_MM, bbox[5] * INCH_PER_MM],
                "dimensions": [
                    bbox_dims[0] * INCH_PER_MM,
                    bbox_dims[1] * INCH_PER_MM,
                    bbox_dims[2] * INCH_PER_MM,
                ],
            },
        ),
        Metric(
            key="part_volume",
            display_name="Part Volume",
            description="Solid volume of the final part",
            unit="in^3",
            category="Geometry",
            occt_extraction="BRepGProp::VolumeProperties",
            feeds=["material usage", "mass estimates"],
            value=volume * (INCH_PER_MM ** 3),
        ),
        Metric(
            key="total_surface_area",
            display_name="Total Surface Area",
            description="Overall external + internal surface area",
            unit="in^2",
            category="Geometry",
            occt_extraction="BRepGProp::SurfaceProperties",
            feeds=["finishing estimates", "coating costs"],
            value=surface_area * (INCH_PER_MM ** 2),
        ),
        Metric(
            key="surface_area_breakdown",
            display_name="Surface Area Breakdown",
            description="Surface area by face type",
            unit="ratio",
            category="Geometry",
            occt_extraction="GeomAdaptor_Surface + BRepGProp::SurfaceProperties",
            feeds=["prismatic vs sculptured classification"],
            value=face_distribution,
        ),
        Metric(
            key="face_type_distribution",
            display_name="Face Type Distribution",
            description="Count of planar vs sculptured/freeform surfaces",
            unit="ratio",
            category="Topology",
            occt_extraction="GeomAdaptor_Surface type classification",
            feeds=["geometry classification"],
            value=face_distribution,
        ),
        Metric(
            key="topology_counts",
            display_name="Number of Faces / Edges / Vertices",
            description="Topological complexity metrics",
            unit="count",
            category="Topology",
            occt_extraction="TopExp_Explorer",
            feeds=["complexity estimation"],
            value={"faces": len(faces), "edges": len(edges), "vertices": len(vertices)},
        ),
        Metric(
            key="thin_wall_thickness",
            display_name="Thin Wall Thickness",
            description="Minimum distance between parallel or opposing faces",
            unit="in",
            category="Manufacturability",
            occt_extraction="BRepExtrema_DistShapeShape between planar faces",
            feeds=["deflection risk", "fixturing"],
            value=None if min_wall is None else min_wall * INCH_PER_MM,
        ),
        Metric(
            key="tool_accessibility",
            display_name="Feature Size / Tool Accessibility Distribution",
            description="Percent of removal volume by tool size",
            unit="ratio",
            category="Manufacturability",
            occt_extraction="Curvature sampling + offset evaluation",
            feeds=["tool selection", "cycle time estimation"],
            value=tool_access,
        ),
        Metric(
            key="hole_diameters",
            display_name="Hole Diameters",
            description="Distinct diameters for each hole present",
            unit="in",
            category="Features",
            occt_extraction="GeomAbs_Cylinder radius from GeomAdaptor_Surface",
            feeds=["drilling operations"],
            value=sorted({round(hole["diameter"], 4) for hole in holes}),
        ),
        Metric(
            key="curvature_statistics",
            display_name="Curvature Statistics",
            description="Max/avg/min curvature",
            unit="1/mm",
            category="Geometry",
            occt_extraction="GeomLProp_SLProps::Curvature",
            feeds=["surface finishing", "geometry classification"],
            value=curvature,
        ),
        Metric(
            key="detected_pockets",
            display_name="Detected Pockets / Cavities",
            description="Count and per-pocket volume, depth, opening area, bounding dims",
            unit="mixed",
            category="Features",
            occt_extraction="BRepClass3d_SolidClassifier + planar face analysis",
            feeds=["toolpath planning", "fixturing"],
            value={"count": len(pockets), "pockets": pockets},
        ),
        Metric(
            key="detected_holes",
            display_name="Detected Holes",
            description="Count and per-hole geometry",
            unit="mixed",
            category="Features",
            occt_extraction="GeomAbs_Cylinder + topological adjacency",
            feeds=["drilling operations"],
            value={"count": len(holes), "holes": holes},
        ),
    ]

    return {
        "metrics": [metric.to_dict() for metric in metrics],
        "units": {
            "length": "in",
            "area": "in^2",
            "volume": "in^3",
        },
        "notes": [
            "Units assume STEP geometry is authored in millimeters; converted to inches.",
            "Pocket and hole detection are heuristic and should be validated per part.",
        ],
    }
