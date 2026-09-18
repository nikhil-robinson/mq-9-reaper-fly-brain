"""Standalone visual check for terrain polish. Run with a DISPLAY set:

    timeout 8 .venv/bin/python scripts/demo_terrain.py

Opens a PyBullet GUI window, applies apply_terrain_polish, and flies a
constant [0, 0, 0, 0.7] action for 200 steps so a human can look at it.
"""
import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401  (registers the PyFlyt/* env ids)

from fly_brain.terrain import apply_terrain_polish

env = gym.make("PyFlyt/Fixedwing-Waypoints-v3", render_mode="human")
env.reset()
apply_terrain_polish(env)

action = np.array([0, 0, 0, 0.7], dtype=np.float32)
for _ in range(200):
    _, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        env.reset()

env.close()
