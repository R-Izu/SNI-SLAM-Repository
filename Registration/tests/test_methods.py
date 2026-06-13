import numpy as np
from synthetic import make_room

from regbim.labels import LabeledCloud
from regbim.methods import available_methods, get_method
from regbim.metrics import apply_sim3, make_sim3, sim3_errors


def test_committed_methods_registered():
    methods = available_methods()
    assert "proposed" in methods
    assert "baseline_open3d" in methods


def test_get_method_unknown_raises():
    try:
        get_method("does_not_exist")
    except KeyError:
        return
    assert False, "expected KeyError for unknown method"


def test_proposed_registers_translation_scale():
    # End-to-end on synthetic data: pure translation+scale (no rotation) must be
    # recovered by the proposed pipeline.
    room = make_room(seed=5)
    cfg = {
        "preprocess": {"voxel_size": 0.06, "normal_radius": 0.2, "normal_max_nn": 30},
        "rotation": {"ransac_iters": 500, "ransac_normal_thresh_deg": 5.0,
                     "yaw_bins": 360, "yaw_use_kmeans": False},
        "semantic_icp": {"max_iter": 60, "max_corr_dist": 0.5, "tukey_c": 0.5,
                         "with_scaling": True, "rotation_fixed": False,
                         "convergence_delta": 1e-7},
        "classes": {"structural": ["wall", "floor", "ceiling"],
                    "match_classes": ["wall", "floor", "ceiling", "door", "window"]},
    }
    s, t = 1.15, np.array([0.5, -0.4, 0.3])
    T_true = make_sim3(np.eye(3), t, s)
    moved = LabeledCloud(points=apply_sim3(T_true, room.points),
                         labels=room.labels.copy(), normals=room.normals.copy())
    T = get_method("proposed").register(room, moved, cfg)
    err = sim3_errors(T, T_true)
    assert err["rot_deg"] < 3.0
    assert err["trans"] < 0.1
    assert err["scale_ratio"] < 0.05
