"""Opens a PyBullet window and flies the fixed-wing airframe so you can watch it.

Run: .venv/bin/python scripts/watch_it_fly.py

Uses a hardcoded constant controller (no fly brain / connectome needed yet —
see scripts/hardcoded_flight_check.py for the headless version of this same
check). Once you have real connectome data, run docs/files/fly_drone_sim.py
instead to watch the actual brain fly it.

Includes: MQ-9 cosmetic skin mount (falls back to PyFlyt's stock mesh until
a real skin file is dropped into assets/vehicles/mq9/), terrain/camera
polish, and a live on-screen HUD (bearing/distance/spike-rates/controls).

`FixedwingWaypointsEnv` doesn't expose `drone_options` through `gym.make()`
or its own `.reset()` (verified directly against its source — TypeError if
you try), so mounting a custom skin means driving `begin_reset`/`end_reset`
ourselves instead of the normal `env.reset()`/`env.step()` — see
scripts/demo_mq9_skin.py for the original verification of this workaround.
"""

import time

import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401 (registers the PyFlyt/* env ids)

from fly_brain.course import bearing_and_distance_from_body_frame_delta
from fly_brain.hud import FlightHUD
from fly_brain.mq9_skin import mq9_drone_options
from fly_brain.terrain import apply_terrain_polish

ENV_ID = "PyFlyt/Fixedwing-Waypoints-v3"
STEPS = 1800  # ~60s at agent_hz=30
HARDCODED_ACTION = np.array([0.0, 0.0, 0.0, 0.7], dtype=np.float64)

# attitude layout (fixedwing_waypoints_env.py): ang_vel(3) quat(4) lin_vel(3) lin_pos(3) prev_action(4) aux(6)
LIN_POS_SLICE = slice(10, 13)


def _reset(unwrapped, drone_options, seed=0):
    unwrapped.begin_reset(seed=seed, options={}, drone_options=drone_options)
    unwrapped.waypoints.reset(unwrapped.env, unwrapped.np_random)
    unwrapped.info["num_targets_reached"] = 0
    unwrapped.end_reset()
    return unwrapped.state


def main():
    drone_options = mq9_drone_options()
    print(f"MQ-9 skin: {'mounted (' + drone_options['model_dir'] + ')' if drone_options else 'not found, using PyFlyt stock mesh'}")

    env = gym.make(ENV_ID, render_mode="human", disable_env_checker=True)
    unwrapped = env.unwrapped
    state = _reset(unwrapped, drone_options)

    apply_terrain_polish(env)
    hud = FlightHUD(env)

    for step in range(STEPS):
        _, _, terminated, truncated, info = unwrapped.step(HARDCODED_ACTION)
        state = unwrapped.state
        position = state["attitude"][LIN_POS_SLICE]
        # No brain in this hardcoded demo — bearing/distance is real (from the
        # env's own target deltas), spike rates are all zero since there's no
        # brain driving anything yet. See fly_drone_sim.py for the real loop.
        bearing_err, dist = bearing_and_distance_from_body_frame_delta(state["target_deltas"][0])
        hud.update(
            position=position,
            bearing_error=bearing_err,
            distance=dist,
            rates={"left_steer": 0.0, "right_steer": 0.0, "pitch": 0.0, "loom": 0.0},
            action=HARDCODED_ACTION,
        )
        time.sleep(1 / 30)  # agent_hz=30, real-time playback
        if terminated or truncated:
            print(f"episode ended at step {step}: {info}")
            state = _reset(unwrapped, drone_options)

    unwrapped.close()


if __name__ == "__main__":
    main()
