"""regbim — semantic registration of SLAM point clouds to a BIM-surrogate reference.

The package is built around a single source-agnostic abstraction, ``LabeledCloud``
(points + 6-class labels + optional normals). Every downstream component
(rotation, semantic_icp, methods, metrics) depends only on that abstraction, so
swapping the Replica reference for an IFC-derived point cloud is a loader change,
not a pipeline change.
"""

from .labels import LabeledCloud  # noqa: F401
