import numpy as np
import torch
from fly_brain.motor import RateTracker, decode_controls, decode_weapons_trigger


def test_rate_tracker_zero_when_never_fired():
    tracker = RateTracker({"left_steer": 0, "right_steer": 1})
    rates = tracker.update(torch.tensor([0.0, 0.0]))
    assert rates == {"left_steer": 0.0, "right_steer": 0.0}


def test_rate_tracker_increases_when_neuron_fires():
    tracker = RateTracker({"left_steer": 0}, decay=0.85)
    rates = tracker.update(torch.tensor([1.0]))
    assert rates["left_steer"] > 0.0
    assert rates["left_steer"] == 1.0 * (1 - 0.85)


def test_rate_tracker_handles_none_index_as_never_firing():
    tracker = RateTracker({"loom": None})
    rates = tracker.update(torch.tensor([1.0, 1.0]))
    assert rates["loom"] == 0.0


def test_decode_controls_symmetric_rates_give_zero_steering():
    action = decode_controls({"left_steer": 0.5, "right_steer": 0.5, "pitch": 0.0, "loom": 0.0})
    aileron, elevator, rudder, throttle = action
    assert aileron == 0.0
    assert rudder == 0.0


def test_decode_controls_left_biased_rate_steers_left():
    action = decode_controls({"left_steer": 0.8, "right_steer": 0.1, "pitch": 0.0, "loom": 0.0})
    aileron, elevator, rudder, throttle = action
    assert aileron > 0.0
    assert rudder > 0.0


def test_decode_controls_clips_to_valid_ranges():
    action = decode_controls(
        {"left_steer": 100.0, "right_steer": 0.0, "pitch": 100.0, "loom": 100.0},
        gains=dict(roll=1.0, pitch=1.0, loom_thrust=0.5, base_throttle=0.6),
    )
    aileron, elevator, rudder, throttle = action
    assert -1.0 <= aileron <= 1.0
    assert -1.0 <= elevator <= 1.0
    assert -1.0 <= rudder <= 1.0
    assert 0.0 <= throttle <= 1.0
    assert isinstance(action, np.ndarray)
    assert action.dtype == np.float32


def test_weapons_trigger_fires_above_threshold():
    assert decode_weapons_trigger({"loom": 0.8}, threshold=0.5) is True


def test_weapons_trigger_does_not_fire_below_threshold():
    assert decode_weapons_trigger({"loom": 0.3}, threshold=0.5) is False


def test_weapons_trigger_fires_exactly_at_threshold():
    assert decode_weapons_trigger({"loom": 0.5}, threshold=0.5) is True


def test_weapons_trigger_defaults_to_no_fire_when_loom_missing():
    assert decode_weapons_trigger({}, threshold=0.5) is False


def test_weapons_trigger_uses_default_threshold():
    assert decode_weapons_trigger({"loom": 0.6}) is True
    assert decode_weapons_trigger({"loom": 0.4}) is False
