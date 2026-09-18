"""Sanity check: does the PyFlyt fixed-wing airframe fly with a dumb constant
controller, before the fly brain is anywhere near the loop? (PLAN.md Section 3)

Run: .venv/bin/python scripts/hardcoded_flight_check.py
"""

import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401 (registers the PyFlyt/* env ids)

ENV_ID = "PyFlyt/Fixedwing-Waypoints-v3"
# The plane spawns already airborne and cruising (~10m altitude, ~20m/s forward),
# so "success" here means holding level flight without stalling/crashing, not
# gaining altitude from zero. Dome radius is 100m by default, so a straight
# cruise will eventually trip out_of_bounds on its own — that's expected, not
# a failure, once we've held past MIN_AIRBORNE_STEPS.
STEPS = 60
MIN_AIRBORNE_STEPS = 60
MIN_ALTITUDE = 2.0
# [aileron, elevator, rudder, throttle], each in [-1, 1] (throttle in [0, 1])
HARDCODED_ACTION = np.array([0.0, 0.0, 0.0, 0.7], dtype=np.float64)

# attitude layout (fixedwing_waypoints_env.py): ang_vel(3) quat(4) lin_vel(3) lin_pos(3) prev_action(4) aux(6)
ALTITUDE_INDEX = 12


def main():
    env = gym.make(ENV_ID, render_mode=None)
    obs, info = env.reset(seed=0)

    min_altitude = obs["attitude"][ALTITUDE_INDEX]
    terminated = truncated = False
    step = 0

    for step in range(STEPS):
        obs, reward, terminated, truncated, info = env.step(HARDCODED_ACTION)
        altitude = obs["attitude"][ALTITUDE_INDEX]
        min_altitude = min(min_altitude, altitude)
        if info.get("collision") or (terminated and step < MIN_AIRBORNE_STEPS):
            break

    env.close()

    print(f"ran {step + 1}/{STEPS} steps")
    print(f"min altitude held: {min_altitude:.2f} m")
    print(f"ended: terminated={terminated} truncated={truncated}")
    print(f"final info: {info}")

    held_airborne = step + 1 >= MIN_AIRBORNE_STEPS and min_altitude > MIN_ALTITUDE
    if held_airborne and not info.get("collision"):
        print("PASS: airframe held level flight under constant control.")
    else:
        print("FAIL: airframe stalled/crashed early — tune HARDCODED_ACTION.")


if __name__ == "__main__":
    main()
