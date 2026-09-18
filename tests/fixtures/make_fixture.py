"""Generates tiny local parquet fixtures for connectome loader tests.

Run once: `.venv/bin/python tests/fixtures/make_fixture.py`
Never fetches from neuPrint — this is synthetic data shaped like the real
export so load_connectome/load_neuron_indices can be tested without a
network call or API token.
"""

import pandas as pd

CONN_PATH = "tests/fixtures/connectivity.parquet"
ANNOT_PATH = "tests/fixtures/annotations.parquet"


def main():
    conn_df = pd.DataFrame(
        {
            "pre_id": [0, 1, 2, 3],
            "post_id": [1, 2, 3, 0],
            "weight": [1.5, -2.0, 0.5, 3.0],
        }
    )
    conn_df.to_parquet(CONN_PATH)

    annot_df = pd.DataFrame(
        {
            "neuron_id": [0, 1, 2, 3, 4],
            "cell_type": ["R1-R6.a", "DNg13_left", "DNg13_right", "DNp_pitch", "other"],
        }
    )
    annot_df.to_parquet(ANNOT_PATH)


if __name__ == "__main__":
    main()
