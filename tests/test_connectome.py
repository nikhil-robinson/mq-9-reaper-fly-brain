import torch
from fly_brain.brain import LIFBrain
from fly_brain.connectome import load_connectome, load_neuron_indices

CONN_PATH = "tests/fixtures/connectivity.parquet"
ANNOT_PATH = "tests/fixtures/annotations.parquet"


def test_load_connectome_returns_correct_shape_and_count():
    W, n_neurons = load_connectome(CONN_PATH, device="cpu")
    assert n_neurons == 4
    assert W.shape == (4, 4)
    assert W.is_sparse
    assert W.is_coalesced()


def test_load_connectome_preserves_weights():
    W, n_neurons = load_connectome(CONN_PATH, device="cpu")
    dense = W.to_dense()
    assert dense[1, 0].item() == 1.5
    assert dense[2, 1].item() == -2.0


def test_load_neuron_indices_finds_photoreceptors():
    photoreceptor_idx, descending_idx = load_neuron_indices(ANNOT_PATH)
    assert list(photoreceptor_idx) == [0]


def test_load_neuron_indices_maps_descending_neurons():
    photoreceptor_idx, descending_idx = load_neuron_indices(ANNOT_PATH)
    assert descending_idx["left_steer"] == 1
    assert descending_idx["right_steer"] == 2
    assert descending_idx["pitch"] == 3
    assert "loom" not in descending_idx


def test_load_neuron_indices_ignores_unmapped_cell_types():
    photoreceptor_idx, descending_idx = load_neuron_indices(ANNOT_PATH)
    assert 4 not in descending_idx.values()


def test_connectome_spike_propagates_through_lif_brain():
    W, n_neurons = load_connectome(CONN_PATH, device="cpu")
    brain = LIFBrain(W, n_neurons, device="cpu", decay=0.9, threshold=1.0)
    ext = torch.zeros(n_neurons)
    ext[0] = 1.5  # drive neuron 0 (pre_id=0) above threshold
    S = brain.step(external_current=ext)
    assert S[0].item() == 1.0
    S2 = brain.step()  # neuron 0's spike (weight 1.5, edge 0->1) should drive neuron 1 over threshold
    assert S2[1].item() == 1.0
