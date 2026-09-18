# Fly Brain Flies an MQ-9 — Build Plan

**Goal:** take a real male fruit fly connectome (MaleCNS v1.0), simulate it as a
spiking network, and use its output to fly a PyFlyt fixed-wing UAV (reskinned
as an "MQ-9") through a checkpoint course — takeoff, steer through waypoints,
land or loiter.

No RL, no training loop. The brain's wiring is fixed; you are building the
sensory encoder and motor decoder around it, then watching what it does.

---

## 0. What you'll end up with

```
camera/checkpoint state --> spike encoding --> LIF connectome sim (PyTorch)
                                                        |
                                                        v
                                          DN spike rates --> control decode
                                                        |
                                                        v
                                              PyFlyt fixed-wing actuators
                                                        |
                                                        v
                                            physics step --> next observation
```

Everything except the middle box (the brain) is code you write and fully
control. The brain box is the one piece of real biology in the loop.

---

## 1. Get the connectome data

**Source:** HHMI Janelia FlyEM's `neuPrint` — the same data portal used for
the earlier female FlyWire/hemibrain releases, now also serving MaleCNS v1.0
(166,700 neurons, released Sept 3, 2026).

Steps:
1. Create a free neuPrint account and generate an API token from your
   account settings.
2. Install the Python client: `pip install neuprint-python`
3. Pull two things per neuron:
   - **Connectivity table**: `(pre_id, post_id, weight, sign)` for every
     synapse — this becomes your sparse adjacency matrix.
   - **Cell type annotations**: lets you find neurons by name instead of
     raw ID (e.g. search for `R1-R6`, `LC*` loom-detectors, `DNg*` descending
     neurons) — you need this to locate your input/output layer.
4. Cache both as local files (parquet or `.npz` for the sparse matrix) —
   you do NOT want to re-query the API every run.

```python
from neuprint import Client, fetch_adjacencies, NeuronCriteria as NC

client = Client('neuprint.janelia.org', dataset='male-cns', token=YOUR_TOKEN)

# example: locate photoreceptor + descending neuron populations
photoreceptors = NC(type='R1-R6.*', regex=True)
descending = NC(type='DN.*', regex=True)

# full connectivity for your chosen neuron set
neurons_df, conn_df = fetch_adjacencies(NC(status='Traced'), NC(status='Traced'))
conn_df.to_parquet('male_cns_connectivity.parquet')
```

Expect the full connectivity pull to take a while and be large — filter to
`Traced`/`Roughly traced` status neurons only, and consider pulling in
batches by region if the full 166K-neuron query is unwieldy.

---

## 2. Build the minimal LIF brain sim

Skip Eon's multi-backend benchmark suite entirely — you need one thing: a
single forward pass per simulated timestep, in PyTorch, on your GPU.

Core pieces:
- **State**: membrane potential vector `V` (166,700,), spike vector `S`
  (same shape, 0/1)
- **Weights**: sparse `torch.sparse_coo_tensor` built from the connectivity
  parquet (pre → post, signed weight)
- **Step**: standard LIF update
  ```python
  input_current = torch.sparse.mm(W, S.unsqueeze(1)).squeeze(1)
  V = V * decay + input_current
  S = (V > threshold).float()
  V = V * (1 - S)  # reset fired neurons
  ```
- **External input injection**: add your encoded sensory spikes directly
  into `V` (or as extra current) for the photoreceptor neuron indices each
  step, before the update above runs.

This is the whole brain box. No gradients, no backward pass — `torch.no_grad()`
throughout.

---

## 3. Set up PyFlyt

```bash
pip install PyFlyt
```

Use PyFlyt's fixed-wing environment as your base airframe. Steps:
1. Start from PyFlyt's fixed-wing example env (`Fixedwing-v0` or similar,
   check current PyFlyt docs for the exact env name — these change between
   releases).
2. Reskin it as your "MQ-9": swap the mesh/texture reference in its model
   config for a Reaper-shaped asset (cosmetic only — the flight physics
   underneath stays PyFlyt's fixed-wing model, which is intentionally close
   to a real slow, propeller-driven UAV rather than a fighter jet).
3. Run it headless first (`render=False`) while you're wiring the loop —
   only enable the viewer once brain → control mapping is working, to save
   overhead while debugging.
4. Confirm you can read the env's action space (expect something like
   `[aileron, elevator, rudder, throttle]`, normalized -1..1 or 0..1) and
   observation space (position, velocity, attitude, and optionally a camera
   image if you enable one) before touching the brain at all.

Sanity-check with a dumb hardcoded controller first (e.g. constant forward
throttle) to confirm takeoff and level flight work before the brain is in
the loop — this isolates "is my airframe/physics broken" from "is my brain
wiring broken."

---

## 4. Define the checkpoint course

Keep it simple and visual for the meme:
- 3–5 waypoints in 3D space, each a sphere/marker in the PyFlyt scene.
- Track "current target checkpoint" in your loop — advance to the next one
  when the drone gets within some radius of the current one.
- This is scripted bookkeeping, not brain logic — the brain never "knows"
  about checkpoints directly, it only ever sees visual input.

---

## 5. Wire the sensory encoder (checkpoint → optic lobe)

Since you don't have a literal camera feed by default in PyFlyt (unless you
enable one), two options:

**Option A — synthetic visual encoding (simpler, recommended to start):**
Compute the checkpoint's bearing and distance relative to the drone's
current heading each step. Render this as a simple synthetic "visual field"
— e.g. a small 2D grid where brightness peaks in the direction of the
checkpoint (bigger/brighter as distance closes, mimicking looming). Rate-code
that grid into spike probabilities for your R1-R6 photoreceptor neuron
indices (brighter pixel = higher instantaneous firing probability).

**Option B — real rendered camera feed:**
Enable PyFlyt's onboard camera, point it forward, and feed the actual
rendered frame through the same rate-coding scheme. More "real," slower,
and doesn't materially change the funniness of the result — start with A,
swap to B later if you want it more legit.

```python
def encode_visual(bearing_error, distance, grid_size=32):
    grid = torch.zeros(grid_size, grid_size)
    # map bearing_error to a horizontal position in the grid,
    # brightness/spread inversely related to distance (closer = bigger/brighter = "looming")
    ...
    return torch.bernoulli(grid.flatten())  # spike train for this step
```

---

## 6. Wire the motor decoder (DN spikes → control surfaces)

1. From your cell-type-annotated data, pick a handful of candidate
   descending neurons (DNs) — start with ones known from the connectome
   literature to sit on steering-related pathways (the Google/Janelia
   announcement specifically traces an R1-R6 → DNg13 pathway as one
   documented example — a reasonable first candidate).
2. Log their spike rate (rolling window, e.g. last 15ms) each step.
3. Hand-map rate → control signal, e.g.:
   ```python
   aileron = (dn_left_rate - dn_right_rate) * gain_roll
   elevator = (dn_pitch_rate - baseline) * gain_pitch
   throttle = base_throttle + dn_loom_rate * gain_thrust_pullback
   ```
4. Expect to spend real time here — this is trial and error, not a formula.
   Log rates for several candidate DNs during a test run, eyeball which ones
   visibly correlate with checkpoint bearing changing, and only wire in the
   ones that show a signal.

---

## 7. Main loop

```python
for step in range(max_steps):
    obs = env.get_observation()
    bearing_err, dist = compute_bearing_to_checkpoint(obs, current_checkpoint)
    visual_spikes = encode_visual(bearing_err, dist)

    inject_input(visual_spikes, photoreceptor_indices)
    V, S = lif_step(V, S, W)  # one brain timestep

    dn_rates.update(S, descending_indices)
    action = decode_controls(dn_rates)

    obs, reward, done, info = env.step(action)
    check_checkpoint_advance(obs, current_checkpoint)

    if step % raster_log_interval == 0:
        log_for_dashboard(S, dn_rates, action)  # feed your neural-activity overlay
```

Run the brain a few times per physics step if the physics step is coarser
than your LIF timestep — decide this once you've measured actual step time
on your GPU (Step 2 gives you that number quickly with a short profiling
run).

---

## 8. Realistic expectations going in

- Takeoff itself may need to stay scripted/assisted (constant throttle +
  fixed pitch until airborne) rather than brain-controlled from wheels-up —
  ground handling isn't something the fly's flight reflexes cover at all.
- Expect wandering, overcorrection, and missed checkpoints especially early
  on — tuning gains in step 6 is most of the real work.
- The honest, fun outcome is a UAV that visibly steers toward things using a
  real biological reflex, badly — not a competent autopilot. That mismatch
  is the bit.

---

## 9. Recording / dashboard

Feed `S` (spikes) and `dn_rates` from the main loop into the neural-activity
dashboard artifact already built — replace its simulated `stepControls()` /
`drawBrain()` values with real numbers written out each step (simplest: dump
to a small JSON file each step, have the dashboard poll/read it, or pipe
over a local WebSocket if you want it live rather than polled).

---

## Rough milestone order

1. Pull + cache connectome data (Section 1)
2. Standalone LIF sim running, confirm real spike output on test input (Section 2)
3. PyFlyt airframe flying with hardcoded controller, no brain yet (Section 3)
4. Checkpoint tracking + synthetic visual encoder wired to brain input (Sections 4–5)
5. DN candidate logging + hand-tuned control mapping (Section 6)
6. Full closed loop, tune gains until it survives one checkpoint (Section 7)
7. Record, overlay dashboard, post it
