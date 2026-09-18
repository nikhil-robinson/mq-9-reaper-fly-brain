import numpy as np
from fly_brain.course import (
    bearing_and_distance,
    bearing_and_distance_from_body_frame_delta,
    advance_checkpoint,
    CHECKPOINTS,
    CHECKPOINT_RADIUS,
)


def test_bearing_zero_when_checkpoint_dead_ahead():
    bearing_err, distance = bearing_and_distance((0, 0, 0), heading=0.0, checkpoint=(10, 0, 0))
    assert abs(bearing_err) < 1e-9
    assert distance == 10.0


def test_bearing_positive_when_checkpoint_to_the_left():
    bearing_err, _ = bearing_and_distance((0, 0, 0), heading=0.0, checkpoint=(0, 10, 0))
    assert bearing_err > 0


def test_bearing_wraps_into_minus_pi_to_pi():
    bearing_err, _ = bearing_and_distance((0, 0, 0), heading=-3.0, checkpoint=(-10, -0.01, 0))
    assert -np.pi < bearing_err <= np.pi


def test_distance_ignores_altitude():
    _, distance = bearing_and_distance((0, 0, 0), heading=0.0, checkpoint=(3, 4, 999))
    assert distance == 5.0


def test_advance_checkpoint_stays_when_far():
    assert advance_checkpoint(0, distance=50.0, checkpoints=CHECKPOINTS, radius=CHECKPOINT_RADIUS) == 0


def test_advance_checkpoint_moves_on_when_close():
    assert advance_checkpoint(0, distance=1.0, checkpoints=CHECKPOINTS, radius=CHECKPOINT_RADIUS) == 1


def test_advance_checkpoint_clamps_at_last():
    last = len(CHECKPOINTS) - 1
    assert advance_checkpoint(last, distance=0.0, checkpoints=CHECKPOINTS, radius=CHECKPOINT_RADIUS) == last


def test_body_frame_delta_zero_bearing_when_target_dead_ahead():
    bearing_err, distance = bearing_and_distance_from_body_frame_delta((10.0, 0.0, 2.0))
    assert abs(bearing_err) < 1e-9
    assert distance == 10.0


def test_body_frame_delta_positive_bearing_when_target_to_the_left():
    bearing_err, _ = bearing_and_distance_from_body_frame_delta((0.0, 10.0, 0.0))
    assert bearing_err > 0


def test_body_frame_delta_ignores_altitude_component():
    _, distance = bearing_and_distance_from_body_frame_delta((3.0, 4.0, 999.0))
    assert distance == 5.0
