"""Checkpoint bearing/distance -> spike-driving current (PLAN.md Section 5)."""

import numpy as np
import torch


def encode_visual(bearing_error, distance, photoreceptor_idx, n_neurons, device="cpu"):
    current = torch.zeros(n_neurons, device=device)
    if len(photoreceptor_idx) == 0:
        return current

    closeness = 1.0 / (1.0 + distance)
    centering = max(0.0, 1.0 - abs(bearing_error) / np.pi)
    drive = float(closeness * centering)

    current[photoreceptor_idx] = drive * 2.0
    return current
