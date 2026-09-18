"""Connectome connectivity + cell-type annotation loading (PLAN.md Section 1)."""

import numpy as np
import pandas as pd
import torch

_DESCENDING_NAME_MAP = {
    "DNg13_left": "left_steer",
    "DNg13_right": "right_steer",
    "DNp_pitch": "pitch",
    "DNp_loom": "loom",
}


def load_connectome(parquet_path, device="cpu"):
    """Load connectivity into a sparse W with W[post_id, pre_id] = weight
    (row=target, col=source), matching LIFBrain.step's torch.sparse.mm(W, S)
    convention.

    Assumes neuron ids are dense and 0-based (0..n_neurons-1, no gaps); a
    real neuPrint export with non-contiguous body IDs will need an
    id-to-index remap before calling this (follow-up work).
    """
    conn_df = pd.read_parquet(parquet_path)
    n_neurons = int(max(conn_df["pre_id"].max(), conn_df["post_id"].max()) + 1)

    idx = torch.tensor(
        np.vstack([conn_df["post_id"].values, conn_df["pre_id"].values]),
        dtype=torch.long,
    )
    vals = torch.tensor(conn_df["weight"].values, dtype=torch.float32)

    W = torch.sparse_coo_tensor(idx, vals, size=(n_neurons, n_neurons)).to(device)
    W = W.coalesce()

    return W, n_neurons


def load_neuron_indices(annotation_path):
    annot_df = pd.read_parquet(annotation_path)

    is_photoreceptor = annot_df["cell_type"].str.match(r"^R1-R6")
    photoreceptor_idx = annot_df.loc[is_photoreceptor, "neuron_id"].to_numpy(dtype=np.int64).copy()

    descending_idx = {}
    for _, row in annot_df.iterrows():
        name = _DESCENDING_NAME_MAP.get(row["cell_type"])
        if name is not None:
            descending_idx[name] = int(row["neuron_id"])

    return photoreceptor_idx, descending_idx
