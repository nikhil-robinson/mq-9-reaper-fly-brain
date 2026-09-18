import numpy as np
import torch
from fly_brain.encoding import encode_visual


def test_no_photoreceptors_returns_zero_current():
    current = encode_visual(0.0, 10.0, np.array([], dtype=np.int64), n_neurons=5)
    assert torch.equal(current, torch.zeros(5))


def test_drive_only_nonzero_at_photoreceptor_indices():
    idx = np.array([1, 3], dtype=np.int64)
    current = encode_visual(0.0, 10.0, idx, n_neurons=5)
    assert current[0].item() == 0.0
    assert current[2].item() == 0.0
    assert current[4].item() == 0.0
    assert current[1].item() > 0.0
    assert current[3].item() > 0.0


def test_closer_checkpoint_drives_harder_than_farther():
    idx = np.array([0], dtype=np.int64)
    near = encode_visual(0.0, 1.0, idx, n_neurons=1)
    far = encode_visual(0.0, 100.0, idx, n_neurons=1)
    assert near[0].item() > far[0].item()


def test_centered_bearing_drives_harder_than_off_axis():
    idx = np.array([0], dtype=np.int64)
    centered = encode_visual(0.0, 10.0, idx, n_neurons=1)
    off_axis = encode_visual(3.0, 10.0, idx, n_neurons=1)
    assert centered[0].item() > off_axis[0].item()


def test_bearing_beyond_pi_clamps_to_zero_centering():
    idx = np.array([0], dtype=np.int64)
    behind = encode_visual(np.pi, 10.0, idx, n_neurons=1)
    assert behind[0].item() == 0.0
