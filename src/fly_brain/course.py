"""Checkpoint course bearing/distance and advancement (PLAN.md Sections 4-5)."""

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
