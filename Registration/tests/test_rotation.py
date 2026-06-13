import numpy as np
from synthetic import ROT_CFG, axis_angle, make_room

from regbim import rotation
from regbim.metrics import angle_between, rotation_error_deg


def _rotate(cloud, R):
    c = cloud.subset(np.ones(len(cloud), bool))
    c.points = cloud.points @ R.T
    c.normals = cloud.normals @ R.T
    return c


def test_gravity_axis_recovered_within_2deg():
    room = make_room()
    up = rotation.estimate_gravity_axis(room, ROT_CFG)
    assert angle_between(up, np.array([0, 0, 1.0])) < 2.0


def test_gravity_axis_under_known_tilt():
    room = make_room()
    R = axis_angle([1, 0.3, 0], 12.0)
    tilted = _rotate(room, R)
    up = rotation.estimate_gravity_axis(tilted, ROT_CFG)
    assert angle_between(up, R @ np.array([0, 0, 1.0])) < 2.0


def test_relative_rotation_recovered():
    room = make_room()
    R = axis_angle([0.2, 0.1, 1.0], 25.0)        # mostly yaw + slight tilt
    moved = _rotate(room, R)
    # align moved -> original should recover R^T (within Manhattan yaw ambiguity)
    cands = rotation.relative_rotation_candidates(moved, room, ROT_CFG)
    best = min(rotation_error_deg(C, R.T) for C in cands)
    assert best < 2.0
