"""
Fly brain -> MQ-9 (PyFlyt) checkpoint flight — starter skeleton.

This is scaffolding, not a finished simulation. Fill in the TODOs.
See PLAN.md for the full walkthrough of each section.

Pipeline per step:
    checkpoint state -> encode_visual() -> inject into LIF -> lif_step()
        -> decode_controls() from descending-neuron spike rates
        -> env.step(action) -> advance checkpoint if close enough
"""

import torch

from fly_brain.brain import LIFBrain
from fly_brain.connectome import load_connectome, load_neuron_indices
from fly_brain.course import CHECKPOINT_RADIUS, CHECKPOINTS, advance_checkpoint, bearing_and_distance
from fly_brain.encoding import encode_visual
from fly_brain.motor import RateTracker, decode_controls


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: point these at your actual cached files from Section 1
    W, n_neurons = load_connectome("male_cns_connectivity.parquet", device=device)
    photoreceptor_idx, descending_idx = load_neuron_indices("male_cns_annotations.parquet")

    brain = LIFBrain(W, n_neurons, device=device)
    rate_tracker = RateTracker(descending_idx)

    # TODO: replace with the actual PyFlyt fixed-wing env of your choice
    # import PyFlyt.gym_envs  # noqa
    # import gymnasium as gym
    # env = gym.make("PyFlyt/Fixedwing-v0", render_mode=None)
    # obs, info = env.reset()
    env = None  # placeholder until PyFlyt env is wired in

    current_cp = 0
    max_steps = 20000

    for step in range(max_steps):
        if env is None:
            break  # remove once PyFlyt env is wired in

        position, heading = get_pose(env)  # TODO: pull from obs
        bearing_err, dist = bearing_and_distance(position, heading, CHECKPOINTS[current_cp])

        ext_current = encode_visual(bearing_err, dist, photoreceptor_idx, n_neurons, device)
        S = brain.step(external_current=ext_current)
        rates = rate_tracker.update(S)
        action = decode_controls(rates)

        obs, reward, terminated, truncated, info = env.step(action)

        current_cp = advance_checkpoint(current_cp, dist, CHECKPOINTS, CHECKPOINT_RADIUS)

        if step % 30 == 0:
            log_for_dashboard(S, rates, action)  # TODO: write to file/socket for the overlay

        if terminated or truncated:
            break


def get_pose(env):
    """TODO: extract (x, y, z) position and yaw heading from env observation."""
    raise NotImplementedError


def log_for_dashboard(S, rates, action):
    """TODO: write spike/rate/action snapshot for the neural-activity overlay to read."""
    pass


if __name__ == "__main__":
    main()
