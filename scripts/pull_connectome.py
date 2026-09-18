"""Pull a real visual-to-descending subnetwork from neuPrint MaleCNS v1.0.

Follows PLAN.md Section 1: fetch R1-R6 photoreceptors, a curated set of
downstream visual interneurons (medulla/lobula/lobula-plate types feeding the
loom-detection pathway), and known descending neurons (DNg13 steering,
DNp01 "Giant Fiber" loom/escape) restricted to status='Traced'.

Real neuPrint body IDs are large, sparse integers -- NOT the dense 0-based
ids that src/fly_brain/connectome.py assumes. This script remaps body IDs to
dense 0-based indices before writing the cached parquet files, and re-uses
that same mapping when writing annotations so index spaces agree.

Cell-type name reality check (see report for details):
- Real photoreceptor type name in this dataset is exactly "R1-R6", which
  already matches load_neuron_indices's `^R1-R6` regex -- no relabeling
  needed.
- Real DNg13 neurons exist (2: one per `somaSide` L/R). We relabel their
  cell_type to "DNg13_left"/"DNg13_right" to match _DESCENDING_NAME_MAP's
  exact keys in connectome.py (which we do not modify).
- There is no neuron in this dataset literally named for "pitch" or "loom"
  function. We use DNp01 ("Giant Fiber"), the well-documented looming/escape
  descending neuron, relabeled "DNp_loom". We found no defensible real
  candidate for "DNp_pitch" and leave it unmapped rather than fabricate one
  (see report Concerns).
"""

import os
import re
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Curated visual-to-descending pathway subnetwork (PLAN.md: "start narrow").
VISUAL_TYPES = [
    "R1-R6",
    "Mi1", "Mi4", "Mi9",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm9",
    "LC4", "LC6", "LC10a", "LC11", "LPLC1", "LPLC2",
]
DESCENDING_TYPES = ["DNg13", "DNp01"]


def read_token(env_path):
    with open(env_path) as f:
        for line in f:
            m = re.match(r'^\s*NEU_API_TOKEN\s*=\s*"?([^"\n]+?)"?\s*$', line)
            if m:
                return m.group(1)
    raise RuntimeError(f"NEU_API_TOKEN not found in {env_path}")


def filter_connected_annotations(annot_out, conn_out):
    """Drop annotated neurons with zero synapses in conn_out.

    load_connectome() infers n_neurons as max(pre_id, post_id) + 1 from
    connectivity alone (it has no other way to know about annotation-only
    ids), so any annotated neuron with no edges here would get an index past
    the end of W -> IndexError at sim time. Such a neuron can't affect the
    LIF sim anyway (no synapses in = no synapses out here), so it's dropped
    rather than inflating n_neurons for a permanently-silent row/col.
    """
    connected_ids = set(conn_out["pre_id"]) | set(conn_out["post_id"])
    return annot_out[annot_out["neuron_id"].isin(connected_ids)]


def relabel_cell_type(row):
    """Map real neuPrint type/somaSide to the labels connectome.py expects."""
    t = row["type"]
    if t == "DNg13":
        side = row.get("somaSide")
        if side == "L":
            return "DNg13_left"
        if side == "R":
            return "DNg13_right"
        return None  # unknown side, don't guess
    if t == "DNp01":
        return "DNp_loom"
    return t  # R1-R6 and all visual interneuron types pass through unchanged


def main():
    token = read_token(os.path.join(REPO_ROOT, ".env"))

    from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC

    client = Client("neuprint.janelia.org", dataset="male-cns:v1.0", token=token)
    print(f"connected to neuPrint (client {client.fetch_version()})", file=sys.stderr)

    all_types = VISUAL_TYPES + DESCENDING_TYPES
    neurons_df, _ = fetch_neurons(NC(type=all_types, status="Traced"))
    if neurons_df.empty:
        raise RuntimeError("fetch_neurons returned 0 neurons for requested types")

    body_ids = neurons_df["bodyId"].tolist()
    _, conn_df = fetch_adjacencies(
        NC(bodyId=body_ids, status="Traced"),
        NC(bodyId=body_ids, status="Traced"),
    )
    if conn_df.empty:
        raise RuntimeError("fetch_adjacencies returned 0 synapses for the fetched neuron set")

    # Dense 0-based remap, built from the traced neuron set actually returned.
    id_to_idx = {bid: i for i, bid in enumerate(sorted(neurons_df["bodyId"].unique()))}

    conn_out = pd.DataFrame({
        "pre_id": conn_df["bodyId_pre"].map(id_to_idx),
        "post_id": conn_df["bodyId_post"].map(id_to_idx),
        "weight": conn_df["weight"].astype(float),
    }).dropna()
    conn_out["pre_id"] = conn_out["pre_id"].astype("int64")
    conn_out["post_id"] = conn_out["post_id"].astype("int64")

    neurons_df["cell_type"] = neurons_df.apply(relabel_cell_type, axis=1)
    annot_out = pd.DataFrame({
        "neuron_id": neurons_df["bodyId"].map(id_to_idx).astype("int64"),
        "cell_type": neurons_df["cell_type"],
    }).dropna(subset=["neuron_id"])

    n_before = len(annot_out)
    annot_out = filter_connected_annotations(annot_out, conn_out)
    n_dropped = n_before - len(annot_out)
    if n_dropped:
        print(f"dropped {n_dropped} annotated neuron(s) with no synapses in this subnetwork", file=sys.stderr)

    os.makedirs(DATA_DIR, exist_ok=True)
    conn_out.to_parquet(os.path.join(DATA_DIR, "connectivity.parquet"), index=False)
    annot_out.to_parquet(os.path.join(DATA_DIR, "annotations.parquet"), index=False)

    n_photo = (annot_out["cell_type"] == "R1-R6").sum()
    n_left = (annot_out["cell_type"] == "DNg13_left").sum()
    n_right = (annot_out["cell_type"] == "DNg13_right").sum()
    n_loom = (annot_out["cell_type"] == "DNp_loom").sum()

    print("== pull_connectome summary ==")
    print(f"neurons: {len(id_to_idx)}")
    print(f"edges:   {len(conn_out)}")
    print(f"photoreceptors (R1-R6): {n_photo}")
    print(f"descending DNg13_left:  {n_left}")
    print(f"descending DNg13_right: {n_right}")
    print(f"descending DNp_loom:    {n_loom}")
    print("descending DNp_pitch:   0 (no real candidate found, left unmapped)")


if __name__ == "__main__":
    main()
