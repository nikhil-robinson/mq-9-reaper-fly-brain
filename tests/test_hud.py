"""FlightHUD must never crash the sim loop, headless or with a GUI."""

import numpy as np
import gymnasium as gym
import PyFlyt.gym_envs  # noqa: F401 (registers PyFlyt/* env ids)

from fly_brain.hud import FlightHUD


def test_hud_noop_headless():
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v3", render_mode=None)
    env.reset(seed=0)

    hud = FlightHUD(env)
    assert hud.client is None  # no GUI connection -> no-op path

    hud.update(
        position=(0.0, 0.0, 10.0),
        bearing_error=0.2,
        distance=42.1,
        rates={"left_steer": 0.31, "right_steer": 0.08, "pitch": 0.0, "loom": 0.0},
        action=np.array([0.23, 0.0, 0.11, 0.6]),
    )

    env.close()
