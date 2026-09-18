import math
import numpy as np
from fly_brain.course import (
    bearing_and_distance,
    bearing_and_distance_from_body_frame_delta,
    advance_checkpoint,
    latlon_to_local_xy,
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


def test_latlon_same_point_returns_origin():
    x, y = latlon_to_local_xy(37.6, -122.4, 37.6, -122.4)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6


def test_latlon_one_degree_north_is_about_111km():
    x, y = latlon_to_local_xy(38.6, -122.4, 37.6, -122.4)
    assert abs(x) < 1.0
    assert 110_000 < y < 112_000


def test_latlon_east_offset_scales_with_cos_latitude():
    origin_lat, origin_lon = 37.6, -122.4
    x, y = latlon_to_local_xy(origin_lat, origin_lon + 1.0, origin_lat, origin_lon)
    expected_x = math.radians(1.0) * 6371000.0 * math.cos(math.radians(origin_lat))
    assert abs(x - expected_x) < 1.0
    assert abs(y) < 1.0


def test_latlon_south_and_west_are_negative():
    x, y = latlon_to_local_xy(36.6, -123.4, 37.6, -122.4)
    assert x < 0
    assert y < 0
