"""Build the FULL traced MaleCNS v1.0 connectome from Janelia's bulk export.

Earlier versions of this script queried neuPrint's API in batches (first a
curated neuron-type subset, then the full 165K-neuron graph via
fetch_adjacencies). Both were far slower than necessary: MaleCNS v1.0 ships
a public bulk export specifically so people can skip the query API --
https://male-cns.janelia.org/download/, files under
gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/. Downloading the
two feather files (~1.05GB weights, ~14MB annotations) and joining them
locally took minutes; the query-API approach was still on batch 9 of 83
after 30+ minutes for the same 165,122 traced neurons.

Real neuPrint body IDs are large, sparse integers -- NOT the dense 0-based
ids that src/fly_brain/connectome.py assumes. This script remaps body IDs
to dense 0-based indices before writing the cached parquet files, and
reuses that same mapping when writing annotations so index spaces agree.

Cell-type name reality check (unchanged from earlier versions):
- Real photoreceptor type name in this dataset is exactly "R1-R6", which
  already matches load_neuron_indices's `^R1-R6` regex -- no relabeling
  needed.
- Real DNg13 neurons exist (2: one per `somaSide` L/R). We relabel their
  type to "DNg13_left"/"DNg13_right" to match _DESCENDING_NAME_MAP's
  exact keys in connectome.py (which we do not modify).
- There is no neuron in this dataset literally named for "pitch" function.
  We use DNp01 ("Giant Fiber"), the well-documented looming/escape
  descending neuron, relabeled "DNp_loom". No defensible real candidate
  for "DNp_pitch" was found; it stays unmapped rather than fabricated.

Run: .venv/bin/python scripts/pull_connectome.py
(downloads data/raw/*.feather on first run if not already present)
"""

import os
import sys
import urllib.request

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

_BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
_WEIGHTS_URL = f"{_BUCKET}/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
_ANNOTATIONS_URL = f"{_BUCKET}/body-annotations-male-cns-v1.0-minconf-0.5.feather"
_WEIGHTS_PATH = os.path.join(RAW_DIR, "connectome-weights.feather")
_ANNOTATIONS_PATH = os.path.join(RAW_DIR, "body-annotations.feather")


def download_if_missing(url, path):
    if os.path.exists(path):
        return
    print(f"downloading {url} -> {path}...", file=sys.stderr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)


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
    return t  # R1-R6 and everything else passes through unchanged


def main():
    download_if_missing(_WEIGHTS_URL, _WEIGHTS_PATH)
    download_if_missing(_ANNOTATIONS_URL, _ANNOTATIONS_PATH)

    print("loading bulk connectome files...", file=sys.stderr)
    weights_df = pd.read_feather(_WEIGHTS_PATH)
    annot_df = pd.read_feather(_ANNOTATIONS_PATH)

    traced_df = annot_df[annot_df["status"] == "Traced"]
    print(f"  {len(traced_df)} traced neurons found", file=sys.stderr)
    traced_ids = set(traced_df["bodyId"])

    conn_df = weights_df[
        weights_df["body_pre"].isin(traced_ids) & weights_df["body_post"].isin(traced_ids)
    ]
    print(f"  {len(conn_df)} synapse connections among traced neurons", file=sys.stderr)

    id_to_idx = {bid: i for i, bid in enumerate(sorted(traced_ids))}

    conn_out = pd.DataFrame(
        {
            "pre_id": conn_df["body_pre"].map(id_to_idx).astype("int64"),
            "post_id": conn_df["body_post"].map(id_to_idx).astype("int64"),
            "weight": conn_df["weight"].astype(float),
        }
    )

    traced_df = traced_df.copy()
    traced_df["cell_type"] = traced_df.apply(relabel_cell_type, axis=1)
    annot_out = pd.DataFrame(
        {
            "neuron_id": traced_df["bodyId"].map(id_to_idx).astype("int64"),
            "cell_type": traced_df["cell_type"],
        }
    ).dropna(subset=["neuron_id"])

    n_before = len(annot_out)
    annot_out = filter_connected_annotations(annot_out, conn_out)
    n_dropped = n_before - len(annot_out)
    if n_dropped:
        print(f"dropped {n_dropped} annotated neuron(s) with no synapses at all", file=sys.stderr)

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
