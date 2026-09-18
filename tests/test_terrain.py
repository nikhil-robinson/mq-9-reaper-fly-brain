"""Minimal smoke test: apply_terrain_polish must not crash a headless env."""
import gymnasium as gym
import PyFlyt.gym_envs  # noqa: F401

from fly_brain.terrain import apply_terrain_polish


def test_apply_terrain_polish_headless_no_crash():
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v3", render_mode=None)
    env.reset()
    apply_terrain_polish(env)  # must not raise, even without a GUI connection
    env.close()
