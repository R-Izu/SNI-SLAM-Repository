"""Loaders (adapters) that turn different on-disk formats into ``LabeledCloud``.

Each loader is an adapter: it absorbs format specifics and emits the same
source-agnostic ``LabeledCloud``. Adding IFC support later means adding one
``load_ifc_cloud`` branch here; nothing downstream changes.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import open3d as o3d
import trimesh

from .labels import (
    CLASS_NAMES,
    NAME_TO_ID,
    LabeledCloud,
    build_replica_to_six,
    color_to_label,
)


def load_source_cloud(cfg: Dict) -> LabeledCloud:
    """Dispatch on ``cfg['source']['type']``. Currently only ``slam_mesh``."""
    spec = cfg["source"]
    if spec["type"] == "slam_mesh":
        return _load_slam_mesh(spec)
    raise ValueError(f"unknown source type: {spec['type']}")


def load_reference_cloud(cfg: Dict) -> LabeledCloud:
    """Dispatch on ``cfg['reference']['type']``.

    ``replica_gt`` treats the Replica semantic mesh as a BIM/IFC surrogate.
    A future ``ifc`` branch returns the same ``LabeledCloud`` type.
    """
    spec = cfg["reference"]
    if spec["type"] == "replica_gt":
        return _load_replica_reference(spec, cfg["classes"])
    if spec["type"] == "ifc":  # extension point (out of scope for this task)
        raise NotImplementedError("load_ifc_cloud not yet implemented")
    raise ValueError(f"unknown reference type: {spec['type']}")


# --------------------------------------------------------------------------- #
# SLAM side: colours encode the 6-class label (decode_segmap palette).
# --------------------------------------------------------------------------- #
def _load_slam_mesh(spec: Dict) -> LabeledCloud:
    mesh = o3d.io.read_triangle_mesh(spec["mesh_path"])
    if not mesh.has_vertex_colors():
        raise ValueError(f"SLAM mesh has no vertex colours: {spec['mesh_path']}")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=int(spec["n_points"]),
                                        use_triangle_normal=True)
    points = np.asarray(pcd.points)
    labels = color_to_label(np.asarray(pcd.colors))
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    return LabeledCloud(points=points, labels=labels, normals=normals,
                        meta={"source": "slam_mesh", "path": spec["mesh_path"]})


# --------------------------------------------------------------------------- #
# Reference side: Replica semantic mesh -> structural-only labelled cloud.
# --------------------------------------------------------------------------- #
def _load_replica_reference(spec: Dict, classes_cfg: Dict) -> LabeledCloud:
    from plyfile import PlyData  # local import: only needed for Replica GT

    ply = PlyData.read(spec["mesh_path"])
    v = ply["vertex"]
    vertices = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    faces_raw = ply["face"]
    vertex_indices = np.asarray(faces_raw.data["vertex_indices"])
    object_id = np.asarray(faces_raw.data["object_id"]).astype(np.int64)
    faces = np.stack(vertex_indices).astype(np.int64)

    with open(spec["info_path"]) as f:
        info = json.load(f)
    id_to_label: List[int] = info["id_to_label"]            # instance id -> Replica class id
    replica_to_six = build_replica_to_six(classes_cfg["replica_to_six"])

    # Per-face 6-class id: object_id -> Replica class -> 6-class (default background).
    n_inst = len(id_to_label)
    face_six = np.zeros(len(faces), dtype=np.int64)
    valid = (object_id >= 0) & (object_id < n_inst)
    rep_cls = np.full(len(faces), -1, dtype=np.int64)
    rep_cls[valid] = np.asarray(id_to_label, dtype=np.int64)[object_id[valid]]
    for replica_id, six_id in replica_to_six.items():
        face_six[rep_cls == replica_id] = six_id

    # Keep only structural classes (IFC has no furniture/background).
    keep_ids = {NAME_TO_ID[n] for n in spec["keep_classes"]}
    keep_face = np.isin(face_six, list(keep_ids))
    faces_k = faces[keep_face]            # (M, 4) quads
    face_six_k = face_six[keep_face]
    if len(faces_k) == 0:
        raise ValueError("no structural faces found in reference mesh")

    # Replica faces are quads; fan-triangulate ourselves so labels stay aligned
    # (trimesh's own triangulation would re-index faces and break the mapping).
    tris = np.concatenate([faces_k[:, [0, 1, 2]], faces_k[:, [0, 2, 3]]], axis=0)
    tri_labels = np.concatenate([face_six_k, face_six_k], axis=0)
    face_six_k = tri_labels

    tm = trimesh.Trimesh(vertices=vertices, faces=tris, process=False)
    n_points = int(spec["n_points"])
    samples, face_index = trimesh.sample.sample_surface(tm, n_points)
    labels = face_six_k[face_index]
    normals = tm.face_normals[face_index]
    return LabeledCloud(points=np.asarray(samples), labels=labels,
                        normals=np.asarray(normals),
                        meta={"source": "replica_gt", "path": spec["mesh_path"]})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_o3d(cloud: LabeledCloud) -> o3d.geometry.PointCloud:
    """Convert to an Open3D point cloud (points + normals, no labels)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points)
    if cloud.normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(cloud.normals)
    return pcd


def class_counts(cloud: LabeledCloud) -> Dict[str, int]:
    """Human-readable per-class point counts (handy for sanity checks)."""
    out: Dict[str, int] = {}
    for cid in cloud.present_classes():
        out[CLASS_NAMES[cid]] = int((cloud.labels == cid).sum())
    return out
