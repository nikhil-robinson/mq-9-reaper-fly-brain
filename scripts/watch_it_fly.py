"""Opens a PyBullet window and flies the fixed-wing airframe so you can watch it.

Run: .venv/bin/python scripts/watch_it_fly.py

Uses a hardcoded constant controller (no fly brain / connectome needed yet —
see scripts/hardcoded_flight_check.py for the headless version of this same
check). Once you have real connectome data, run docs/files/fly_drone_sim.py
instead to watch the actual brain fly it (add render_mode="human" there too).
"""

import time

import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401 (registers the PyFlyt/* env ids)

ENV_ID = "PyFlyt/Fixedwing-Waypoints-v3"
STEPS = 1800  # ~60s at agent_hz=30
HARDCODED_ACTION = np.array([0.0, 0.0, 0.0, 0.7], dtype=np.float64)


def main():
    env = gym.make(ENV_ID, render_mode="human")
    obs, info = env.reset(seed=0)

    for step in range(STEPS):
        obs, reward, terminated, truncated, info = env.step(HARDCODED_ACTION)
        time.sleep(1 / 30)  # agent_hz=30, real-time playback
        if terminated or truncated:
            print(f"episode ended at step {step}: {info}")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
