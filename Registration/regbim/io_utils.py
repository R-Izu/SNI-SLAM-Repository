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
        cloud = _load_replica_reference(spec, cfg["classes"])
    elif spec["type"] == "ifc":
        cloud = _load_ifc_reference(spec, cfg["classes"])
    else:
        raise ValueError(f"unknown reference type: {spec['type']}")
    # Partial-coverage study (R3 T3). Absent in every existing config, and a
    # no-op when absent, so previous results are unaffected.
    from .clip import clip_reference
    return clip_reference(cloud, spec.get("clip"))


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
# Reference side: IFC -> structural-only labelled cloud (R3 T1).
# --------------------------------------------------------------------------- #
def _load_ifc_reference(spec: Dict, classes_cfg: Dict) -> LabeledCloud:
    """Load an IFC-derived reference cloud, building it if ifcopenshell is here.

    The registration pipeline runs under the ``sni-slam`` env, which is Python
    3.7 and therefore cannot host ifcopenshell (>=3.9). So the IFC is turned
    into points/labels/normals by ``Registration/scripts/ifc_export.py`` under
    the ``bim-ifc`` env, and this loader reads that ``.npz``. When ifcopenshell
    *is* importable the same code path builds the cloud in-process, so nothing
    silently depends on a stale cache.
    """
    import os

    cache = spec.get("cache_path")
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        meta["cache_path"] = cache
        _check_ifc_cache_matches(spec, meta, cache)
        if "is_inner" in z.files:
            # R5 Q6: room-facing side of each solid face. Carried for failure
            # analysis only -- no stage of the method reads it.
            meta["is_inner"] = z["is_inner"]
        return LabeledCloud(points=z["points"], labels=z["labels"],
                            normals=z["normals"], meta=meta)

    try:
        import ifcopenshell  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "IFC reference needs either a prebuilt cache or ifcopenshell.\n"
            "  cache_path: %s (not found)\n"
            "  build it with:\n"
            "    conda run -n bim-ifc python Registration/scripts/ifc_export.py \\\n"
            "        --ifc %s --config <this config> \\\n"
            "        --out %s%s"
            % (cache, spec.get("path"), cache,
               "" if not spec.get("spaces") else
               " \\\n        --spaces " + " ".join(str(s) for s in spec["spaces"])))

    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from ifc_export import build_ifc_cloud       # noqa: E402

    res = build_ifc_cloud(
        spec["path"], classes_cfg["ifc_class_map"], int(spec["n_points"]),
        spaces=spec.get("spaces"), storeys=spec.get("storeys"),
        space_margin_m=float(spec.get("space_margin_m", 0.35)),
        seed=int(spec.get("seed", 0)), keep_classes=spec.get("keep_classes"))
    meta = dict(res["meta"])
    meta["is_inner"] = res["is_inner"]
    return LabeledCloud(points=res["points"], labels=res["labels"],
                        normals=res["normals"], meta=meta)


def _check_ifc_cache_matches(spec: Dict, meta: Dict, cache: str) -> None:
    """Fail loudly if the cache was built for a different request.

    A cache that silently disagrees with the config would make the room-level
    experiments (E1 vs E2: reference = 410 vs 411) compare the wrong things.
    """
    want = [str(s) for s in (spec.get("spaces") or [])]
    got = [str(s) for s in (meta.get("spaces_requested") or [])]
    if sorted(want) != sorted(got):
        raise ValueError(
            "IFC cache %s was built for spaces=%s but the config asks for %s. "
            "Rebuild it with ifc_export.py --spaces." % (cache, got or None, want or None))
    if spec.get("keep_classes") and meta.get("keep_classes") != spec["keep_classes"]:
        raise ValueError(
            "IFC cache %s keep_classes=%s != config %s"
            % (cache, meta.get("keep_classes"), spec["keep_classes"]))
    # Storey matters as much as spaces here: this model carries a second ceiling
    # 0.6 m above the real one, belonging to another storey.
    if [str(s) for s in (spec.get("storeys") or [])] != \
            [str(s) for s in (meta.get("storeys_requested") or [])]:
        raise ValueError(
            "IFC cache %s was built for storeys=%s but the config asks for %s"
            % (cache, meta.get("storeys_requested"), spec.get("storeys")))


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
