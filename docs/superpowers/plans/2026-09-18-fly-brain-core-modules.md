# Fly-Brain Core Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `docs/files/fly_drone_sim.py` skeleton into a tested `fly_brain` package covering every piece of the pipeline that does NOT require a live neuPrint API token, a real connectome download, or an installed PyFlyt/gym environment: the LIF brain step, the visual encoder, the motor decoder, the checkpoint-course logic, and a connectome loader tested against a small local fixture file (not a live API pull).

**Architecture:** A single `fly_brain` Python package under `src/fly_brain/`, one module per pipeline stage (mirrors the section boundaries in `docs/files/PLAN.md`), each with pure functions/classes that take tensors/arrays in and return them out — no global state, no I/O inside the compute functions. `pytest` tests live in `tests/`, one test file per module, using small synthetic data (a handful of neurons, not 166K) so tests run in milliseconds on CPU.

**Tech Stack:** Python 3, PyTorch (CPU, tests never require CUDA), NumPy, pandas + pyarrow (for the connectome fixture), pytest. A project-local virtualenv already exists at `.venv/` (Debian's system Python is externally-managed) — every command in this plan runs through `.venv/bin/python` / `.venv/bin/pytest`.

**Spec:** `docs/files/PLAN.md` (design/spec narrative) and `docs/files/fly_drone_sim.py` (existing skeleton — the plan extracts and hardens its logic; do not treat the skeleton as throwaway, its function signatures are the starting contract).

## Global Constraints

- Package name: `fly_brain`, installed in editable mode from `src/fly_brain/`.
- All new/modified Python files pass `.venv/bin/python -m py_compile` (no syntax errors) and their tests pass under `.venv/bin/pytest`.
- No task in this plan touches the network, requires a GPU, or requires PyFlyt/neuPrint to be installed. If a task's implementer finds itself needing either, that is a plan defect — stop and flag it rather than adding a fake/mocked network call.
- Tensor dtype convention: spike vectors and membrane potentials are `torch.float32`. Index arrays (`photoreceptor_idx`, `descending_idx` values) are Python `int` / `np.int64`.
- Keep functions pure: no module-level mutable state, no `print`/logging inside compute functions (tests assert on return values, not stdout).
- Follow the existing skeleton's function names and signatures from `docs/files/fly_drone_sim.py` unless a task explicitly changes one — later tasks and the main-loop wiring (out of scope here, but future work) depend on these names staying stable: `LIFBrain.step`, `encode_visual`, `RateTracker.update`, `decode_controls`, `bearing_and_distance`, `load_connectome`, `load_neuron_indices`.
- Out of scope for this plan (do not implement): real neuPrint API calls, PyFlyt env wiring, `get_pose`, `log_for_dashboard`, the `main()` loop itself. These stay as TODOs in the skeleton until a follow-up plan covers them.

---

### Task 1: Project scaffold + LIF brain step

**Files:**
- Create: `pyproject.toml`
- Create: `src/fly_brain/__init__.py`
- Create: `src/fly_brain/brain.py`
- Test: `tests/test_brain.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `fly_brain.brain.LIFBrain` — constructor `LIFBrain(W: torch.Tensor, n_neurons: int, device: str = "cpu", decay: float = 0.9, threshold: float = 1.0)`, method `step(self, external_current: torch.Tensor | None = None) -> torch.Tensor` returning the new spike vector `S` (shape `(n_neurons,)`, dtype float32, values 0.0/1.0). Instance attributes `V` and `S` (both `(n_neurons,)` float32 tensors) are updated in place each call, matching `docs/files/fly_drone_sim.py:60-79`.

- [ ] **Step 1: Create the package scaffold**

Create `pyproject.toml`:

```toml
[project]
name = "fly-brain"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "numpy",
    "pandas",
    "pyarrow",
]

[tool.setuptools.packages.find]
where = ["src"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

Create `src/fly_brain/__init__.py` (empty file).

Install the package in editable mode into the existing venv:

Run: `.venv/bin/pip install -e .`
Expected: succeeds, prints "Successfully installed fly-brain-0.1.0" (or similar).

- [ ] **Step 2: Write the failing test for a single LIF step**

Create `tests/test_brain.py`:

```python
import torch
from fly_brain.brain import LIFBrain


def _ring_weights(n):
    """Each neuron excites the next one around a ring, weight 2.0."""
    idx = torch.tensor([list(range(n)), [(i + 1) % n for i in range(n)]], dtype=torch.long)
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_brain.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'fly_brain.brain'` (or `LIFBrain` not defined).

- [ ] **Step 4: Implement `LIFBrain`**

Create `src/fly_brain/brain.py`:

```python
"""Minimal leaky-integrate-and-fire connectome sim (PLAN.md Section 2)."""

import torch


class LIFBrain:
    def __init__(self, W, n_neurons, device="cpu", decay=0.9, threshold=1.0):
        self.W = W
        self.n = n_neurons
        self.device = device
        self.decay = decay
        self.threshold = threshold
        self.V = torch.zeros(n_neurons, device=device)
        self.S = torch.zeros(n_neurons, device=device)

    @torch.no_grad()
    def step(self, external_current=None):
        input_current = torch.sparse.mm(self.W, self.S.unsqueeze(1)).squeeze(1)
        if external_current is not None:
            input_current = input_current + external_current

        self.V = self.V * self.decay + input_current
        self.S = (self.V > self.threshold).float()
        self.V = self.V * (1 - self.S)
        return self.S
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_brain.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/fly_brain/__init__.py src/fly_brain/brain.py tests/test_brain.py
git commit -m "feat: add fly_brain package scaffold and LIF brain step"
```

---

### Task 2: Visual encoder

**Files:**
- Create: `src/fly_brain/encoding.py`
- Test: `tests/test_encoding.py`

**Interfaces:**
- Consumes: nothing new (only `torch`, `numpy`).
- Produces: `fly_brain.encoding.encode_visual(bearing_error: float, distance: float, photoreceptor_idx: "np.ndarray", n_neurons: int, device: str = "cpu") -> torch.Tensor`, returning a `(n_neurons,)` float32 current tensor, matching `docs/files/fly_drone_sim.py:86-106`. Later course-wiring work (out of scope) will pass this straight into `LIFBrain.step(external_current=...)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_encoding.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_encoding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fly_brain.encoding'`.

- [ ] **Step 3: Implement `encode_visual`**

Create `src/fly_brain/encoding.py`:

```python
"""Checkpoint bearing/distance -> spike-driving current (PLAN.md Section 5)."""

import numpy as np
import torch


def encode_visual(bearing_error, distance, photoreceptor_idx, n_neurons, device="cpu"):
    current = torch.zeros(n_neurons, device=device)
    if len(photoreceptor_idx) == 0:
        return current

    closeness = 1.0 / (1.0 + distance)
    centering = max(0.0, 1.0 - abs(bearing_error) / np.pi)
    drive = float(closeness * centering)

    current[photoreceptor_idx] = drive * 2.0
    return current
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_encoding.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fly_brain/encoding.py tests/test_encoding.py
git commit -m "feat: add visual encoder for checkpoint bearing/distance"
```

---

### Task 3: Motor decoder (rate tracker + control mapping)

**Files:**
- Create: `src/fly_brain/motor.py`
- Test: `tests/test_motor.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `fly_brain.motor.RateTracker` (constructor `RateTracker(indices: dict[str, int | None], window=15, decay=0.85)`, method `update(self, S: torch.Tensor) -> dict[str, float]`), and `fly_brain.motor.decode_controls(rates: dict[str, float], gains: dict | None = None) -> "np.ndarray"` returning `[aileron, elevator, rudder, throttle]` each clipped to their documented range, matching `docs/files/fly_drone_sim.py:113-145`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_motor.py`:

```python
import numpy as np
import torch
from fly_brain.motor import RateTracker, decode_controls


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_motor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fly_brain.motor'`.

- [ ] **Step 3: Implement `RateTracker` and `decode_controls`**

Create `src/fly_brain/motor.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_motor.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fly_brain/motor.py tests/test_motor.py
git commit -m "feat: add motor decoder (rate tracker + control mapping)"
```

---

### Task 4: Checkpoint course logic

**Files:**
- Create: `src/fly_brain/course.py`
- Test: `tests/test_course.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `fly_brain.course.bearing_and_distance(position: tuple[float, float, float], heading: float, checkpoint: tuple[float, float, float]) -> tuple[float, float]` (bearing_error radians in `(-pi, pi]`, distance), `fly_brain.course.CHECKPOINTS: list[tuple[float, float, float]]`, `fly_brain.course.CHECKPOINT_RADIUS: float`, and `fly_brain.course.advance_checkpoint(current_index: int, distance: float, checkpoints: list, radius: float = CHECKPOINT_RADIUS) -> int` (returns the next index, clamped to the last checkpoint — extracted from the inline logic at `docs/files/fly_drone_sim.py:212-213` so it's independently testable).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_course.py`:

```python
import numpy as np
from fly_brain.course import bearing_and_distance, advance_checkpoint, CHECKPOINTS, CHECKPOINT_RADIUS


def test_bearing_zero_when_checkpoint_dead_ahead():
    bearing_err, distance = bearing_and_distance((0, 0, 0), heading=0.0, checkpoint=(10, 0, 0))
    assert abs(bearing_err) < 1e-9
    assert distance == 10.0


def test_bearing_positive_when_checkpoint_to_the_left():
    bearing_err, _ = bearing_and_distance((0, 0, 0), heading=0.0, checkpoint=(0, 10, 0))
    assert bearing_err > 0


def test_bearing_wraps_into_minus_pi_to_pi():
    bearing_err, _ = bearing_and_distance((0, 0, 0), heading=-3.0, checkpoint=(-10, -0.01, 0))
    assert -np.pi < bearing_err <= np.pi


def test_distance_ignores_altitude():
    _, distance = bearing_and_distance((0, 0, 0), heading=0.0, checkpoint=(3, 4, 999))
    assert distance == 5.0


def test_advance_checkpoint_stays_when_far():
    assert advance_checkpoint(0, distance=50.0, checkpoints=CHECKPOINTS, radius=CHECKPOINT_RADIUS) == 0


def test_advance_checkpoint_moves_on_when_close():
    assert advance_checkpoint(0, distance=1.0, checkpoints=CHECKPOINTS, radius=CHECKPOINT_RADIUS) == 1


def test_advance_checkpoint_clamps_at_last():
    last = len(CHECKPOINTS) - 1
    assert advance_checkpoint(last, distance=0.0, checkpoints=CHECKPOINTS, radius=CHECKPOINT_RADIUS) == last
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_course.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fly_brain.course'`.

- [ ] **Step 3: Implement course logic**

Create `src/fly_brain/course.py`:

```python
"""Checkpoint course bearing/distance and advancement (PLAN.md Sections 4-5)."""

import numpy as np


def bearing_and_distance(position, heading, checkpoint):
    dx = checkpoint[0] - position[0]
    dy = checkpoint[1] - position[1]
    distance = float(np.hypot(dx, dy))
    target_heading = float(np.arctan2(dy, dx))
    bearing_error = (target_heading - heading + np.pi) % (2 * np.pi) - np.pi
    return bearing_error, distance


CHECKPOINTS = [
    (50, 0, 30),
    (50, 50, 35),
    (0, 50, 30),
    (0, 0, 25),
]
CHECKPOINT_RADIUS = 8.0


def advance_checkpoint(current_index, distance, checkpoints, radius=CHECKPOINT_RADIUS):
    if distance < radius:
        return min(current_index + 1, len(checkpoints) - 1)
    return current_index
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_course.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fly_brain/course.py tests/test_course.py
git commit -m "feat: add checkpoint course bearing/distance and advancement logic"
```

---

### Task 5: Connectome loader (tested against a local fixture, no network)

**Files:**
- Create: `src/fly_brain/connectome.py`
- Create: `tests/fixtures/make_fixture.py`
- Test: `tests/test_connectome.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `fly_brain.connectome.load_connectome(parquet_path: str, device: str = "cpu") -> tuple[torch.Tensor, int]` (coalesced sparse COO tensor + neuron count, matching `docs/files/fly_drone_sim.py:20-42`), and `fly_brain.connectome.load_neuron_indices(annotation_path: str) -> tuple["np.ndarray", dict]` reading a parquet with columns `neuron_id` (int) and `cell_type` (str): `photoreceptor_idx` is every `neuron_id` whose `cell_type` matches regex `^R1-R6`, and `descending_idx` is a dict built from rows whose `cell_type` is one of `DNg13_left -> "left_steer"`, `DNg13_right -> "right_steer"`, `DNp_pitch -> "pitch"`, `DNp_loom -> "loom"` (map `cell_type` to the fixed name via that lookup table; a `cell_type` not in the table is skipped).

- [ ] **Step 1: Write the fixture generator**

Create `tests/fixtures/make_fixture.py` (run manually to produce test fixtures, not itself a test):

```python
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
```

Run: `.venv/bin/python tests/fixtures/make_fixture.py`
Expected: creates `tests/fixtures/connectivity.parquet` and `tests/fixtures/annotations.parquet`, no errors.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_connectome.py`:

```python
import torch
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
    assert dense[0, 1].item() == 1.5
    assert dense[1, 2].item() == -2.0


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_connectome.py -v`
Expected: FAIL — fixtures missing (run Step 1's script first if not already run) and then `ModuleNotFoundError: No module named 'fly_brain.connectome'`.

- [ ] **Step 4: Implement the connectome loader**

Create `src/fly_brain/connectome.py`:

```python
"""Connectome connectivity + cell-type annotation loading (PLAN.md Section 1)."""

import re

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
    conn_df = pd.read_parquet(parquet_path)
    n_neurons = int(max(conn_df["pre_id"].max(), conn_df["post_id"].max()) + 1)

    idx = torch.tensor(
        np.vstack([conn_df["pre_id"].values, conn_df["post_id"].values]),
        dtype=torch.long,
    )
    vals = torch.tensor(conn_df["weight"].values, dtype=torch.float32)

    W = torch.sparse_coo_tensor(idx, vals, size=(n_neurons, n_neurons)).to(device)
    W = W.coalesce()

    return W, n_neurons


def load_neuron_indices(annotation_path):
    annot_df = pd.read_parquet(annotation_path)

    is_photoreceptor = annot_df["cell_type"].str.match(r"^R1-R6")
    photoreceptor_idx = annot_df.loc[is_photoreceptor, "neuron_id"].to_numpy(dtype=np.int64)

    descending_idx = {}
    for _, row in annot_df.iterrows():
        name = _DESCENDING_NAME_MAP.get(row["cell_type"])
        if name is not None:
            descending_idx[name] = int(row["neuron_id"])

    return photoreceptor_idx, descending_idx
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_connectome.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/fly_brain/connectome.py tests/fixtures/make_fixture.py tests/fixtures/connectivity.parquet tests/fixtures/annotations.parquet tests/test_connectome.py
git commit -m "feat: add connectome loader tested against local fixtures"
```

---

### Task 6: Wire the skeleton onto the new package

**Files:**
- Modify: `docs/files/fly_drone_sim.py`
- Test: (existing suite must still pass; no new test file — this task is a refactor, verified by the full suite)

**Interfaces:**
- Consumes: `fly_brain.brain.LIFBrain`, `fly_brain.encoding.encode_visual`, `fly_brain.motor.RateTracker`, `fly_brain.motor.decode_controls`, `fly_brain.course.bearing_and_distance`, `fly_brain.course.advance_checkpoint`, `fly_brain.course.CHECKPOINTS`, `fly_brain.course.CHECKPOINT_RADIUS`, `fly_brain.connectome.load_connectome`, `fly_brain.connectome.load_neuron_indices`.
- Produces: nothing new — this is the file that stays a thin skeleton/entry point per PLAN.md, now importing the tested package instead of duplicating logic inline.

- [ ] **Step 1: Replace inline definitions with imports**

Edit `docs/files/fly_drone_sim.py`: delete the bodies of `load_connectome`, `load_neuron_indices`, `LIFBrain`, `encode_visual`, `RateTracker`, `decode_controls`, `bearing_and_distance`, and the module-level `CHECKPOINTS`/`CHECKPOINT_RADIUS`, replacing them with imports from `fly_brain`, and update the checkpoint-advance call in `main()` to use `advance_checkpoint`. The file becomes:

```python
"""
Fly brain -> MQ-9 (PyFlyt) checkpoint flight — starter skeleton.

This is scaffolding, not a finished simulation. Fill in the TODOs.
See PLAN.md for the full walkthrough of each section.

Pipeline per step:
    checkpoint state -> encode_visual() -> inject into LIF -> lif_step()
        -> decode_controls() from descending-neuron spike rates
        -> env.step(action) -> advance checkpoint if close enough
"""

import torch

from fly_brain.brain import LIFBrain
from fly_brain.connectome import load_connectome, load_neuron_indices
from fly_brain.course import CHECKPOINT_RADIUS, CHECKPOINTS, advance_checkpoint, bearing_and_distance
from fly_brain.encoding import encode_visual
from fly_brain.motor import RateTracker, decode_controls


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: point these at your actual cached files from Section 1
    W, n_neurons = load_connectome("male_cns_connectivity.parquet", device=device)
    photoreceptor_idx, descending_idx = load_neuron_indices("male_cns_annotations.parquet")

    brain = LIFBrain(W, n_neurons, device=device)
    rate_tracker = RateTracker(descending_idx)

    # TODO: replace with the actual PyFlyt fixed-wing env of your choice
    # import PyFlyt.gym_envs  # noqa
    # import gymnasium as gym
    # env = gym.make("PyFlyt/Fixedwing-v0", render_mode=None)
    # obs, info = env.reset()
    env = None  # placeholder until PyFlyt env is wired in

    current_cp = 0
    max_steps = 20000

    for step in range(max_steps):
        if env is None:
            break  # remove once PyFlyt env is wired in

        position, heading = get_pose(env)  # TODO: pull from obs
        bearing_err, dist = bearing_and_distance(position, heading, CHECKPOINTS[current_cp])

        ext_current = encode_visual(bearing_err, dist, photoreceptor_idx, n_neurons, device)
        S = brain.step(external_current=ext_current)
        rates = rate_tracker.update(S)
        action = decode_controls(rates)

        obs, reward, terminated, truncated, info = env.step(action)

        current_cp = advance_checkpoint(current_cp, dist, CHECKPOINTS, CHECKPOINT_RADIUS)

        if step % 30 == 0:
            log_for_dashboard(S, rates, action)  # TODO: write to file/socket for the overlay

        if terminated or truncated:
            break


def get_pose(env):
    """TODO: extract (x, y, z) position and yaw heading from env observation."""
    raise NotImplementedError


def log_for_dashboard(S, rates, action):
    """TODO: write spike/rate/action snapshot for the neural-activity overlay to read."""
    pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file still imports cleanly**

Run: `.venv/bin/python -c "import ast; ast.parse(open('docs/files/fly_drone_sim.py').read())"`
Expected: no output, exit code 0 (syntax check — `main()` itself isn't invoked since it needs real fixture files and PyFlyt).

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

Run: `.venv/bin/pytest -v`
Expected: all tests from Tasks 1-5 still pass (27 passed).

- [ ] **Step 4: Commit**

```bash
git add docs/files/fly_drone_sim.py
git commit -m "refactor: wire fly_drone_sim.py skeleton onto tested fly_brain package"
```

---

## Out of Scope (follow-up work, not this plan)

- Real neuPrint API pull (`neuprint-python`, live API token) — Section 1 of `docs/files/PLAN.md`.
- PyFlyt environment installation and wiring, `get_pose()`, the actual `env.step()` loop — Section 3 and the body of `main()`.
- `log_for_dashboard()` real implementation (file/socket write) — Section 9.
- DN candidate selection and gain tuning against a real connectome and real flight — Section 6's "trial and error" tuning pass, and Section 7's full closed loop.

These all require either network/credentials, an installed simulator, or empirical tuning against a running environment — none of which fit a unit-test-driven task loop. A follow-up plan should cover them once PyFlyt is installed and a neuPrint token is available.
