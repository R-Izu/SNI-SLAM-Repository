import numpy as np
from synthetic import axis_angle, make_room

from regbim.labels import LabeledCloud
from regbim.metrics import apply_sim3, decompose_sim3, make_sim3, sim3_errors
from regbim.scale import umeyama
from regbim.semantic_icp import semantic_icp

ICP_CFG = {
    "semantic_icp": {"max_iter": 60, "max_corr_dist": 0.5, "tukey_c": 0.5,
                     "with_scaling": True, "rotation_fixed": False,
                     "convergence_delta": 1e-7},
    "classes": {"match_classes": ["wall", "floor", "ceiling", "door", "window"]},
}


def test_umeyama_recovers_known_sim3():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(200, 3))
    R = axis_angle([0.3, 1, 0.2], 40.0)
    t = np.array([1.5, -2.0, 0.7])
    s = 1.4
    dst = s * (src @ R.T) + t
    Re, te, se = umeyama(src, dst, with_scaling=True)
    assert np.linalg.norm(Re - R) < 1e-6
    assert np.linalg.norm(te - t) < 1e-6
    assert abs(se - s) < 1e-6


def test_semantic_icp_recovers_translation_scale():
    room = make_room(seed=2)
    s, t = 1.2, np.array([0.4, -0.3, 0.2])
    T_true = make_sim3(np.eye(3), t, s)
    moved = LabeledCloud(points=apply_sim3(T_true, room.points),
                         labels=room.labels.copy(), normals=room.normals.copy())
    # rotation fixed to identity; recover scale + translation
    T = semantic_icp(room, moved, make_sim3(np.eye(3), np.zeros(3), 1.0),
                     ICP_CFG, rotation_fixed=True)
    err = sim3_errors(T, T_true)
    assert err["trans"] < 0.02
    assert err["scale_ratio"] < 0.02


def test_semantic_icp_recovers_full_sim3_from_close_init():
    room = make_room(seed=3)
    R = axis_angle([0, 0, 1], 8.0)
    s, t = 1.1, np.array([0.2, 0.2, -0.1])
    T_true = make_sim3(R, t, s)
    moved = LabeledCloud(points=apply_sim3(T_true, room.points),
                         labels=room.labels.copy(), normals=room.normals.copy())
    T = semantic_icp(room, moved, make_sim3(np.eye(3), np.zeros(3), 1.0), ICP_CFG)
    err = sim3_errors(T, T_true)
    assert err["rot_deg"] < 2.0
    assert err["trans"] < 0.05
    assert err["scale_ratio"] < 0.03
