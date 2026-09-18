"""Descending-neuron spike rates -> control surfaces (PLAN.md Section 6)."""

import numpy as np


class RateTracker:
    """Rolling spike-rate estimate per tracked neuron index."""

    def __init__(self, indices: dict, window=15, decay=0.85):
        self.indices = indices
        self.rates = {k: 0.0 for k in indices}
        self.decay = decay

    def update(self, S):
        for name, idx in self.indices.items():
            fired = float(S[idx].item()) if idx is not None else 0.0
            self.rates[name] = self.rates[name] * self.decay + fired * (1 - self.decay)
        return self.rates


def decode_controls(rates, gains=None):
    gains = gains or dict(roll=1.0, pitch=1.0, loom_thrust=0.5, base_throttle=0.6)

    left = rates.get("left_steer", 0.0)
    right = rates.get("right_steer", 0.0)
    pitch = rates.get("pitch", 0.0)
    loom = rates.get("loom", 0.0)

    aileron = np.clip((left - right) * gains["roll"], -1, 1)
    elevator = np.clip(pitch * gains["pitch"], -1, 1)
    rudder = np.clip((left - right) * 0.5, -1, 1)
    throttle = np.clip(gains["base_throttle"] - loom * gains["loom_thrust"], 0, 1)

    return np.array([aileron, elevator, rudder, throttle], dtype=np.float32)
