"""Class vocabulary and the source-agnostic ``LabeledCloud`` abstraction.

The 6-class system and palette are the single source of truth shared with the
SLAM side: they MUST stay identical to ``decode_segmap`` in
``src/utils/Mesher.py`` (id 0..5 = background, wall, door, floor, window,
ceiling). If that palette changes, label decoding from SLAM meshes breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# id -> name, identical ordering to src/utils/Mesher.py decode_segmap.
CLASS_NAMES: List[str] = ["background", "wall", "door", "floor", "window", "ceiling"]
NAME_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}

# RGB(0-255) palette matching decode_segmap. Order follows CLASS_NAMES (id 0..5).
LABEL_COLORS: np.ndarray = np.array(
    [
        (128, 128, 128),  # 0 background
        (255, 64, 64),    # 1 wall
        (255, 200, 64),   # 2 door
        (180, 220, 255),  # 3 floor
        (64, 200, 255),   # 4 window
        (200, 100, 255),  # 5 ceiling
    ],
    dtype=np.float64,
)


@dataclass
class LabeledCloud:
    """A point cloud with per-point 6-class labels (and optional normals).

    This is the only type the registration pipeline depends on, regardless of
    whether the points came from a SLAM mesh, Replica GT, or an IFC export.
    """

    points: np.ndarray                     # (N, 3) float
    labels: np.ndarray                     # (N,) int in [0, 5]
    normals: Optional[np.ndarray] = None   # (N, 3) float or None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if len(self.points) != len(self.labels):
            raise ValueError(
                f"points/labels length mismatch: {len(self.points)} vs {len(self.labels)}"
            )
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64).reshape(-1, 3)
            if len(self.normals) != len(self.points):
                raise ValueError("normals length mismatch with points")

    def __len__(self) -> int:
        return len(self.points)

    def class_mask(self, name: str) -> np.ndarray:
        """Boolean mask of points belonging to class ``name``."""
        return self.labels == NAME_TO_ID[name]

    def subset(self, mask: np.ndarray) -> "LabeledCloud":
        """Return a new cloud restricted to ``mask`` (carries normals/meta)."""
        mask = np.asarray(mask)
        return LabeledCloud(
            points=self.points[mask],
            labels=self.labels[mask],
            normals=None if self.normals is None else self.normals[mask],
            meta=dict(self.meta),
        )

    def present_classes(self) -> List[int]:
        """Sorted list of class ids actually present in the cloud."""
        return sorted(int(c) for c in np.unique(self.labels))


def color_to_label(colors: np.ndarray) -> np.ndarray:
    """Map RGB colors to 6-class ids by nearest palette colour.

    ``colors`` may be 0-1 floats (Open3D) or 0-255. Nearest-neighbour matching
    absorbs the tiny rounding introduced when colours round-trip through a mesh.
    """
    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    if colors.max() <= 1.0 + 1e-6:
        colors = colors * 255.0
    # (N, 1, 3) - (1, 6, 3) -> (N, 6) squared distances
    d2 = ((colors[:, None, :] - LABEL_COLORS[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(np.int64)


def build_replica_to_six(mapping_name_keyed: Dict, ) -> Dict[int, int]:
    """Build a Replica-class-id -> 6-class-id table from a name-keyed config map.

    Config gives e.g. ``{93: 'wall', 40: 'floor', ...}``; unlisted Replica ids
    fall back to background (0).
    """
    out: Dict[int, int] = {}
    for replica_id, six_name in mapping_name_keyed.items():
        out[int(replica_id)] = NAME_TO_ID[str(six_name)]
    return out
