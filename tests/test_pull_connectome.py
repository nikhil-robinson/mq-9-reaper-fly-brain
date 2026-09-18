import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from pull_connectome import filter_connected_annotations  # noqa: E402


def test_drops_annotated_neuron_with_no_synapses():
    # neuron_id 2 is annotated but never appears as pre_id or post_id below —
    # exactly the case that caused a real IndexError in encode_visual() when
    # such a neuron happened to be a mapped photoreceptor/descending type.
    annot_out = pd.DataFrame({"neuron_id": [0, 1, 2], "cell_type": ["R1-R6", "DNg13_left", "R1-R6"]})
    conn_out = pd.DataFrame({"pre_id": [0], "post_id": [1], "weight": [1.0]})

    filtered = filter_connected_annotations(annot_out, conn_out)

    assert sorted(filtered["neuron_id"]) == [0, 1]


def test_keeps_all_annotations_when_all_neurons_are_connected():
    annot_out = pd.DataFrame({"neuron_id": [0, 1], "cell_type": ["R1-R6", "DNg13_left"]})
    conn_out = pd.DataFrame({"pre_id": [0], "post_id": [1], "weight": [1.0]})

    filtered = filter_connected_annotations(annot_out, conn_out)

    assert sorted(filtered["neuron_id"]) == [0, 1]
