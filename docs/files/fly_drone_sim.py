"""
Fly brain -> MQ-9 (PyFlyt) checkpoint flight.

PyFlyt's own Fixedwing-Waypoints-v3 env generates and auto-advances its own
waypoints and already reports target deltas in body frame (x=forward,
y=left/right, z=up/down relative to the aircraft) — so this loop reads
bearing/distance straight from the observation instead of tracking world
positions itself (fly_brain.course's CHECKPOINTS/bearing_and_distance are for
a manual/synthetic checkpoint course; this concrete env doesn't need them).

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
from fly_brain.motor import RateTracker, decode_controls

ENV_ID = "PyFlyt/Fixedwing-Waypoints-v3"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: point these at your real neuPrint export once pulled (PLAN.md Section 1)
    connectivity_path = "male_cns_connectivity.parquet"
    annotations_path = "male_cns_annotations.parquet"
    try:
        W, n_neurons = load_connectome(connectivity_path, device=device)
        photoreceptor_idx, descending_idx = load_neuron_indices(annotations_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Real connectome data isn't pulled yet — see PLAN.md Section 1 "
            "for the neuPrint export steps, or point connectivity_path/annotations_path "
            "at your own cached files."
        ) from exc

    brain = LIFBrain(W, n_neurons, device=device)
    rate_tracker = RateTracker(descending_idx)

    env = gym.make(ENV_ID, render_mode=None)
    obs, info = env.reset()

    max_steps = 20000

    for step in range(max_steps):
        bearing_err, dist = bearing_and_distance_from_body_frame_delta(obs["target_deltas"][0])

        ext_current = encode_visual(bearing_err, dist, photoreceptor_idx, n_neurons, device)
        S = brain.step(external_current=ext_current)
        rates = rate_tracker.update(S)
        action = decode_controls(rates)

        obs, reward, terminated, truncated, info = env.step(action)

        if step % 30 == 0:
            log_for_dashboard(S, rates, action)  # TODO: write to file/socket for the overlay

        if terminated or truncated:
            break

    env.close()


def log_for_dashboard(S, rates, action):
    """TODO: write spike/rate/action snapshot for the neural-activity overlay to read."""
    pass


if __name__ == "__main__":
    main()
