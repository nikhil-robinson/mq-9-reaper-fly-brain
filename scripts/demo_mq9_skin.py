"""Standalone demo: fly Fixedwing-Waypoints-v3 with the MQ-9 skin mounted.

`FixedwingBaseEnv.begin_reset()` (fixedwing_base_env.py) takes a
`drone_options` dict and forwards it straight to `Aviary(drone_options=...)`,
which is how PyFlyt resolves `<model_dir>/fixedwing/fixedwing.{urdf,yaml}`.
That plumbing is real and confirmed working here.

HOWEVER: `FixedwingWaypointsEnv.__init__()` and its `.reset()` do NOT expose
a `drone_options` parameter (checked its source directly) - they hardcode
`begin_reset(seed, options)` with no drone_options, so plain
`gym.make("PyFlyt/Fixedwing-Waypoints-v3", drone_options=...)` raises
`TypeError: unexpected keyword argument 'drone_options'`. So we go through
`gym.make(...).unwrapped` and call `begin_reset`/`end_reset` ourselves,
passing drone_options directly - this is the "whichever genuinely works"
path and is what's verified below (no exception, correct model_dir loaded).
"""

import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401  (registers PyFlyt/* env ids)

from fly_brain.mq9_skin import mq9_drone_options


def main():
    drone_options = mq9_drone_options()
    print(f"drone_options: {drone_options}")
    assert "model_dir" in drone_options, "expected the MQ-9 skin dir to be found"

    env = gym.make(
        "PyFlyt/Fixedwing-Waypoints-v3",
        render_mode=None,
        disable_env_checker=True,
    )
    unwrapped = env.unwrapped

    # Reconfigure with our drone_options (mirrors what
    # FixedwingWaypointsEnv.reset() does, but with drone_options threaded
    # through since the env class itself doesn't expose that parameter).
    unwrapped.begin_reset(seed=0, options={}, drone_options=drone_options)
    unwrapped.waypoints.reset(unwrapped.env, unwrapped.np_random)
    unwrapped.info["num_targets_reached"] = 0
    unwrapped.end_reset()

    # Confirm the underlying Aviary/drone actually recorded our model_dir.
    actual_model_dir = str(unwrapped.env.drones[0].param_path).replace(
        "/fixedwing.yaml", ""
    )
    print(f"drone param_path directory in use: {actual_model_dir}")
    assert drone_options["model_dir"] in str(unwrapped.env.drones[0].param_path), (
        "env did not load the custom model_dir"
    )

    action = np.array([0.0, 0.0, 0.0, 0.7])
    for step in range(30):
        # step the unwrapped env directly (we bypassed gym's reset() above,
        # so the OrderEnforcing wrapper's reset-tracking would reject
        # env.step()).
        _, _, terminated, truncated, _ = unwrapped.step(action)
        if terminated or truncated:
            unwrapped.begin_reset(seed=0, options={}, drone_options=drone_options)
            unwrapped.waypoints.reset(unwrapped.env, unwrapped.np_random)
            unwrapped.info["num_targets_reached"] = 0
            unwrapped.end_reset()
    unwrapped.close()

    print("OK: ran 30 steps with the MQ-9 skin's model_dir mounted, no errors.")


if __name__ == "__main__":
    main()
