"""
Fly brain -> MQ-9 (PyFlyt) checkpoint flight.

PyFlyt's own Fixedwing-Waypoints-v3 env generates and auto-advances its own
waypoints and already reports target deltas in body frame (x=forward,
y=left/right, z=up/down relative to the aircraft) — so this loop reads
bearing/distance straight from the observation instead of tracking world
positions itself (fly_brain.course's CHECKPOINTS/bearing_and_distance are for
a manual/synthetic checkpoint course; this concrete env doesn't need them).

`FixedwingWaypointsEnv` doesn't expose `drone_options` through `gym.make()`
or its own `.reset()`, so mounting the MQ-9 cosmetic skin means driving
`begin_reset`/`end_reset` ourselves instead of `env.reset()`/`env.step()`
(see scripts/demo_mq9_skin.py for the original verification of this).

Pipeline per step:
    obs['target_deltas'][0] -> encode_visual() -> inject into LIF -> lif_step()
        -> decode_controls() from descending-neuron spike rates
        -> env.step(action) -> PyFlyt auto-advances its own waypoints
"""

import gymnasium as gym
import PyFlyt.gym_envs  # noqa: F401 (registers the PyFlyt/* env ids)
import torch

from fly_brain.brain import LIFBrain
from fly_brain.connectome import load_connectome, load_neuron_indices
from fly_brain.course import bearing_and_distance_from_body_frame_delta
from fly_brain.encoding import encode_visual
from fly_brain.hud import FlightHUD
from fly_brain.motor import RateTracker, decode_controls
from fly_brain.mq9_skin import mq9_drone_options
from fly_brain.terrain import apply_terrain_polish

ENV_ID = "PyFlyt/Fixedwing-Waypoints-v3"

# attitude layout (fixedwing_waypoints_env.py): ang_vel(3) quat(4) lin_vel(3) lin_pos(3) prev_action(4) aux(6)
LIN_POS_SLICE = slice(10, 13)


def _reset(unwrapped, drone_options, seed=None):
    unwrapped.begin_reset(seed=seed, options={}, drone_options=drone_options)
    unwrapped.waypoints.reset(unwrapped.env, unwrapped.np_random)
    unwrapped.info["num_targets_reached"] = 0
    unwrapped.end_reset()
    return unwrapped.state


def main(render=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Real neuPrint export (PLAN.md Section 1) — a curated visual-to-descending
    # subnetwork (R1-R6 photoreceptors, medulla/loom-detector interneurons,
    # DNg13/DNp01 descending neurons), not the full 166K-neuron connectome.
    # No mapped "pitch" descending neuron was found in this subnetwork —
    # decode_controls() defaults pitch's rate to 0.0 (rates.get("pitch", 0.0)),
    # so elevator gets no brain-driven signal yet; that's a real gap in DN
    # coverage, not a crash risk. See PLAN.md Section 6 for hand-picking more
    # DN candidates once you're ready to chase that down.
    connectivity_path = "data/connectivity.parquet"
    annotations_path = "data/annotations.parquet"
    try:
        W, n_neurons = load_connectome(connectivity_path, device=device)
        photoreceptor_idx, descending_idx = load_neuron_indices(annotations_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Run `.venv/bin/python scripts/pull_connectome.py` first "
            "(needs NEU_API_TOKEN in .env) to cache real connectome data, "
            "or point connectivity_path/annotations_path at your own files."
        ) from exc

    brain = LIFBrain(W, n_neurons, device=device)
    rate_tracker = RateTracker(descending_idx)

    drone_options = mq9_drone_options()
    env = gym.make(ENV_ID, render_mode="human" if render else None, disable_env_checker=True)
    unwrapped = env.unwrapped
    state = _reset(unwrapped, drone_options)

    if render:
        apply_terrain_polish(env, camera_target=state["attitude"][LIN_POS_SLICE])
    hud = FlightHUD(env)

    max_steps = 20000

    for step in range(max_steps):
        bearing_err, dist = bearing_and_distance_from_body_frame_delta(state["target_deltas"][0])

        ext_current = encode_visual(bearing_err, dist, photoreceptor_idx, n_neurons, device)
        S = brain.step(external_current=ext_current)
        rates = rate_tracker.update(S)
        action = decode_controls(rates)

        _, reward, terminated, truncated, info = unwrapped.step(action)
        state = unwrapped.state

        if render:
            hud.update(
                position=state["attitude"][LIN_POS_SLICE],
                bearing_error=bearing_err,
                distance=dist,
                rates=rates,
                action=action,
            )

        if step % 30 == 0:
            log_for_dashboard(S, rates, action)  # TODO: write to file/socket for the overlay

        if terminated or truncated:
            break

    unwrapped.close()


def log_for_dashboard(S, rates, action):
    """TODO: write spike/rate/action snapshot for the neural-activity overlay to read."""
    pass


if __name__ == "__main__":
    main()
