"""Standalone demo exercising FlightHUD in a real PyBullet GUI loop.

Run: .venv/bin/python scripts/demo_hud.py
Uses a constant [0,0,0,0.7] action; bearing/distance are computed for real
from obs["target_deltas"][0], rates are a fixed dummy dict just to exercise
the HUD text every step (no fly brain wired in here).
"""

import time

import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401 (registers PyFlyt/* env ids)

from fly_brain.course import bearing_and_distance_from_body_frame_delta
from fly_brain.hud import FlightHUD

ENV_ID = "PyFlyt/Fixedwing-Waypoints-v3"
STEPS = 150
ACTION = np.array([0.0, 0.0, 0.0, 0.7], dtype=np.float64)
DUMMY_RATES = {"left_steer": 0.31, "right_steer": 0.08, "pitch": 0.0, "loom": 0.0}


def main():
    env = gym.make(ENV_ID, render_mode="human")
    obs, info = env.reset(seed=0)
    hud = FlightHUD(env)

    for step in range(STEPS):
        bearing, distance = bearing_and_distance_from_body_frame_delta(obs["target_deltas"][0])
        position = obs["attitude"][9:12]  # lin_pos: (ang_vel, ang_pos, lin_vel, lin_pos, quaternion, ...)
        hud.update(position, bearing, distance, DUMMY_RATES, ACTION)
        if step == 0:
            print(f"HUD debug text id: {hud._text_id}")

        obs, reward, terminated, truncated, info = env.step(ACTION)
        time.sleep(1 / 30)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
