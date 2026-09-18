"""Checkpoint course bearing/distance and advancement (PLAN.md Sections 4-5)."""

import math
import numpy as np


def bearing_and_distance(position, heading, checkpoint):
    dx = checkpoint[0] - position[0]
    dy = checkpoint[1] - position[1]
    distance = float(np.hypot(dx, dy))
    target_heading = float(np.arctan2(dy, dx))
    bearing_error = (target_heading - heading + np.pi) % (2 * np.pi) - np.pi
    return bearing_error, distance


CHECKPOINTS = [
    (50, 0, 30),
    (50, 50, 35),
    (0, 50, 30),
    (0, 0, 25),
]
CHECKPOINT_RADIUS = 8.0


def advance_checkpoint(current_index, distance, checkpoints, radius=CHECKPOINT_RADIUS):
    if distance < radius:
        return min(current_index + 1, len(checkpoints) - 1)
    return current_index


def bearing_and_distance_from_body_frame_delta(target_delta):
    """target_delta: (x, y, z) body-frame vector to target (x=forward, y=left/right),
    e.g. PyFlyt's obs['target_deltas'][i]. Returns (bearing_error, distance)."""
    dx, dy = target_delta[0], target_delta[1]
    bearing_error = float(np.arctan2(dy, dx))
    distance = float(np.hypot(dx, dy))
    return bearing_error, distance


_EARTH_RADIUS_M = 6371000.0


def latlon_to_local_xy(lat_deg, lon_deg, origin_lat_deg, origin_lon_deg):
    """Equirectangular approximation: fine over a mission-scale area of a
    few km, not a real geodesic calculation. x = east offset, y = north
    offset, both meters, from the origin to (lat_deg, lon_deg)."""
    lat_rad = math.radians(lat_deg)
    origin_lat_rad = math.radians(origin_lat_deg)
    dlat = math.radians(lat_deg - origin_lat_deg)
    dlon = math.radians(lon_deg - origin_lon_deg)

    x = dlon * _EARTH_RADIUS_M * math.cos(origin_lat_rad)
    y = dlat * _EARTH_RADIUS_M
    return x, y
