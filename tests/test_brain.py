import torch
from fly_brain.brain import LIFBrain


def _ring_weights(n):
    """Each neuron excites the next one around a ring, weight 2.0."""
    idx = torch.tensor([[(i + 1) % n for i in range(n)], list(range(n))], dtype=torch.long)
    vals = torch.full((n,), 2.0)
    return torch.sparse_coo_tensor(idx, vals, size=(n, n)).coalesce()


def test_step_returns_zero_spikes_with_no_input():
    n = 4
    brain = LIFBrain(_ring_weights(n), n, device="cpu", decay=0.9, threshold=1.0)
    S = brain.step()
    assert S.shape == (n,)
    assert torch.equal(S, torch.zeros(n))


def test_external_current_above_threshold_fires_next_step():
    n = 3
    brain = LIFBrain(_ring_weights(n), n, device="cpu", decay=0.9, threshold=1.0)
    ext = torch.tensor([1.5, 0.0, 0.0])
    S = brain.step(external_current=ext)
    assert S.tolist() == [1.0, 0.0, 0.0]


def test_fired_neuron_resets_membrane_potential():
    n = 3
    brain = LIFBrain(_ring_weights(n), n, device="cpu", decay=0.9, threshold=1.0)
    brain.step(external_current=torch.tensor([1.5, 0.0, 0.0]))
    assert brain.V[0].item() == 0.0


def test_spike_propagates_through_sparse_weights_next_step():
    n = 3
    brain = LIFBrain(_ring_weights(n), n, device="cpu", decay=0.9, threshold=1.0)
    brain.step(external_current=torch.tensor([1.5, 0.0, 0.0]))  # neuron 0 fires
    S2 = brain.step()  # neuron 0's spike (weight 2.0) should drive neuron 1 over threshold
    assert S2.tolist() == [0.0, 1.0, 0.0]
