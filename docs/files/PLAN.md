# Fly Brain Flies an F-16 — Build Plan (v2: FlightGear)

**Goal:** take a real male fruit fly connectome (MaleCNS v1.0), simulate it as a
spiking network, and use its output to fly FlightGear's community F-16
(NikolaiVChr/f16, real JSBSim aerodynamics, real weapons/radar-lock system)
through a mission: scripted takeoff, brain-driven cruise/steering, brain-driven
weapons trigger, scripted landing.

No RL, no training loop. The brain's wiring is fixed; you are building the
sensory encoder and motor decoder around it, then watching what it does.

**Why the rewrite (v1 was PyFlyt + a cosmetic MQ-9 mesh):** PyFlyt's debug
renderer (PyBullet's OpenGL viewer) has a hard visual ceiling — no real
terrain, no scenery, no proper HUD, aliased textures at any real viewing
distance. FlightGear is a full flight-sim application with real terrain,
scenery, camera modes, and HUD, and the community F-16 aircraft brings real
JSBSim flight dynamics and an actual working missile/radar-lock system
(Nasal-scripted Dynamic Launch Zone calculations, not cosmetic). The
trade-off: FlightGear is a separate ~2GB application process, not a Python
library — the brain loop now talks to it over a network socket instead of
calling a physics step in-process.

**Honest scope limit, carried over from v1's own Section 8 and unchanged by
the rewrite:** our real connectome pull (`data/connectivity.parquet`,
`data/annotations.parquet`, `scripts/pull_connectome.py`) currently has only
one descending neuron confirmed *connected* within the curated subnetwork
(DNp01 / "DNp_loom") — the DNg13 left/right steering neurons came back
isolated (zero synapses) in that fetch, so they produce no real signal today.
Full autonomous takeoff-through-landing from real spike output is not
realistic on this data; ground handling isn't something fly reflexes cover
at all regardless. This plan splits the mission accordingly:

- **Scripted** (ordinary Python/JSBSim autopilot properties, not the brain):
  takeoff roll, initial climb-out, final approach, landing, gear.
- **Brain-driven**: steering/cruise once airborne and stable (bearing/distance
  to a waypoint encoded as visual input, descending-neuron rates decoded to
  aileron/rudder), and the weapons trigger (loom-detector spike rate crossing
  a threshold fires — a fly's "something's closing fast, react" reflex
  repurposed as a fire-control trigger; thematically apt, and it's the one
  channel we actually have real signal on).
- Widening `VISUAL_TYPES`/`DESCENDING_TYPES` in `scripts/pull_connectome.py`
  to recover a working DNg13 steering signal is real follow-up work (Section
  6), not blocking this plan — cruise/steering can ship hand-tuned off
  whatever DN rates are actually live, same as v1's own philosophy.

---

## 0. What you'll end up with

```
FlightGear (F-16, real terrain/scenery/HUD, JSBSim physics)
        ^                                    |
        | control inputs (telnet/property)   | FDM telemetry (telnet/property)
        |                                    v
Python: checkpoint state --> spike encoding --> LIF connectome sim (PyTorch)
                                                        |
                                                        v
                                          DN spike rates --> control decode
                                                        |
                                                        v
                                    aileron/rudder + weapons-trigger properties
                                       sent back to FlightGear over telnet
```

Everything except the middle box (the brain) and the FlightGear/JSBSim
physics itself is code you write and fully control.

---

## 1. Get the connectome data — unchanged from v1, already done

Already pulled and cached: `data/connectivity.parquet` (237,601 edges),
`data/annotations.parquet` (16,742 remapped 0-based indices). Loader:
`src/fly_brain/connectome.py` (`load_connectome`, `load_neuron_indices`,
already tested against both real data and local fixtures). See
`scripts/pull_connectome.py` for the neuPrint query and the "drop annotated
neurons with zero synapses in this subnetwork" fix (this is why DNg13 L/R
show 0 count today — real, not a bug).

No changes needed here for this plan. Widening the curated type list to
recover DNg13 steering signal is optional follow-up (see Section 6).

---

## 2. The LIF brain sim — unchanged from v1, already done

`src/fly_brain/brain.py` (`LIFBrain`), `src/fly_brain/encoding.py`
(`encode_visual`), `src/fly_brain/motor.py` (`RateTracker`,
`decode_controls`) are already built and tested (37 tests passing as of this
rewrite). No changes needed for the brain box itself. What changes is what
feeds it (Section 4) and what it drives (Section 5) — both move from
PyFlyt's Python API to FlightGear's network protocol.

---

## 3. Install FlightGear + the F-16 aircraft

**Blocking step, needs your action (sudo, real download):**

```bash
sudo apt-get install -y flightgear
```

Installs the `flightgear` runtime (~46MB) plus `flightgear-data-base`
(~1.1GB), `flightgear-data-models` (~276MB), `flightgear-data-ai` (~525MB)
— roughly 2GB total. This is the base simulator with its default aircraft
set; it does NOT include the F-16.

**F-16 aircraft** (NikolaiVChr/f16, JSBSim, real weapons system):

```bash
git clone https://github.com/NikolaiVChr/f16.git ~/.fgfs/Aircraft/f16
```

(FlightGear's default per-user aircraft search path is `~/.fgfs/Aircraft` on
Linux; confirm this exact path once FlightGear is installed — `fgfs
--version` or the launcher's Settings show the actual `$FG_HOME` in use, and
`--aircraft-dir`/`--fg-aircraft` can point elsewhere if you'd rather keep it
inside this repo.)

**Sanity-check done, both blockers resolved:** FlightGear 2020.3.19 is
installed (`/usr/games/fgfs`, not on `$PATH` by default — use the full path
or add `/usr/games` to `PATH`), and `NikolaiVChr/f16` is cloned to
`~/.fgfs/Aircraft/f16` (FlightGear's real per-user aircraft search path on
this machine, confirmed). Launched headless-safe with:
```bash
/usr/games/fgfs --fg-aircraft=$HOME/.fgfs/Aircraft --aircraft=f16-simplified \
  --airport=KSFO --telnet=5501 --disable-real-weather-fetch \
  --disable-ai-traffic --prop:/sim/terrasync/enabled=false --timeofday=noon
```
Confirmed booting to a flyable (if unfueled/engine-off) state — logs show
"Initializing General Dynamics F-16 (Simplified FDM) systems" and the
property tree answering telnet queries. Two real things to carry into the
mission-state-machine work (Section 6): (1) **cold-and-dark start** — engine
`cutoff=true`, `throttle=0` at spawn, same as a real aircraft; TAKEOFF phase
needs an engine-start sequence, not just throttle-up. (2) KSFO's default
parking spot put the aircraft over water ("Aircraft hit Ocean" in the log
before any control input) — pick/verify a real runway-start position, not
just an airport ICAO, before scripting TAKEOFF.

**Startup takes ~30-45s** (scenery/systems init) before the telnet server
actually answers commands — a bridge/script that connects and sends
commands immediately after the port opens will hang; poll with a real
command (not just a TCP connect check) until it responds.

---

## 4. The FlightGear bridge (new — this is the core of the rewrite)

FlightGear exposes its live property tree over a simple line-based telnet
protocol (`--telnet=5501` at launch, then `get /path/to/prop` /
`set /path/to/prop <value>` over a plain TCP socket) — this is the simplest,
most robust integration path (vs. the binary `--native-fdm`/`--native-ctrls`
struct protocols the reference Udemy-course repo uses, which require
matching FlightGear's exact FDM struct version). Python's stdlib `telnetlib`
was removed in 3.13, so this repo's Python 3.13 venv needs a small
hand-rolled `socket`-based telnet client — confirmed genuinely simple:
`ls`/`get`/`set` are plain `\r\n`-terminated ASCII lines, no telnet
control-sequence negotiation needed for FlightGear's usage, and the server
prompts with `/> ` after each response (read until that suffix, not a fixed
byte count — response length varies).

**Confirmed real property paths (live telnet session against
`f16-simplified`, FlightGear 2020.3.19):**

Controls to write (all `double`, unindexed under `/controls/flight`):
- `/controls/flight/aileron`, `/controls/flight/elevator`,
  `/controls/flight/rudder` — range not separately confirmed but standard
  FlightGear convention is -1..1.
- `/controls/engines/engine[0]/throttle` (`double`) — per-engine, indexed;
  this airframe also exposes `.../cutoff` (`bool`, `true` at cold start —
  must be set `false` before throttle has any effect), `.../starter`
  (`bool`), `.../magnetos` (`int`).
- `/controls/armament/trigger` (`bool`) — the weapons fire control. Also
  present and relevant: `master-arm` (`bool`) and `master-arm-switch`
  (`int`) likely gate whether `trigger` does anything (unarmed weapons
  can't fire on a real aircraft either) — confirm the arm sequence during
  Section 6 implementation, not guessed here.
- `/controls/gear/gear-down` (`bool`) — `true` at spawn; LANDING phase
  needs this already down (it should be, from spawn) rather than a
  retraction/extension sequence to build for a first pass.

Telemetry to read:
- `/position/longitude-deg`, `/position/latitude-deg` (`double`, degrees —
  NOT what `fly_brain.course.bearing_and_distance` expects, which wants
  local (x, y, z) meters; converting lat/lon to a local tangent-plane
  x/y for bearing math is new code, see below), `/position/altitude-ft`
  (`double`, feet — convert to meters, `* 0.3048`).
- `/orientation/heading-deg`, `/orientation/pitch-deg`,
  `/orientation/roll-deg` (`double`, degrees).
- `/velocities/airspeed-kt`, `/velocities/groundspeed-kt` (`double`, knots).

**New, not in v1:** `bearing_and_distance` was built for a local flat x/y/z
checkpoint course (meters), but FlightGear reports position as lat/lon
(degrees) on a spherical Earth. Add
`fly_brain.course.latlon_to_local_xy(lat_deg, lon_deg, origin_lat_deg,
origin_lon_deg) -> (x_m, y_m)` (equirectangular approximation is plenty
accurate over a mission-scale area of a few km — do not reach for a full
geodesic library for this) before `bearing_and_distance` can be reused
as-is against FlightGear telemetry.

`src/fly_brain/flightgear_bridge.py`:
- `FlightGearBridge.__init__(host="localhost", port=5501)` — opens the
  socket, reads until the first `/> ` prompt (the connection banner).
- `._read_until_prompt() -> str` — internal: accumulate `recv()` chunks
  until the buffer ends with `/> `.
- `.get(path: str) -> str` — sends `get {path}\r\n`, parses FlightGear's
  `value (type)` reply format, returns the raw value string (caller casts).
- `.set(path: str, value)` — sends `set {path} {value}\r\n`.
- `.get_telemetry() -> dict` — reads the confirmed position/orientation/
  velocity properties above, returns floats with units already converted
  (meters, radians, m/s — pick ONE consistent unit system here so
  `encode_visual`/`bearing_and_distance` callers never see feet/knots/degrees).
- `.send_controls(aileron, elevator, rudder, throttle)` — writes the three
  flight-control properties plus engine throttle.
- `.trigger_weapons(fire: bool)` — writes `/controls/armament/trigger`.

---

## 5. Sensory encoding & motor decoding — same brain, new plumbing

`encode_visual` (Section 5 of v1, `src/fly_brain/encoding.py`) already takes
`(bearing_error, distance, photoreceptor_idx, n_neurons, device)` — this
doesn't change. What changes is where `bearing_error`/`distance` come from:
`FlightGearBridge.get_telemetry()`'s position/heading plus a scripted
waypoint list (reuse `fly_brain.course.bearing_and_distance` — this is
exactly the manual/world-frame checkpoint course it was originally built
for, unlike PyFlyt's env which had its own built-in body-frame waypoints).

`decode_controls` (`src/fly_brain/motor.py`) already returns
`[aileron, elevator, rudder, throttle]` — reused as-is, sent via
`FlightGearBridge.send_controls(*action)` instead of `env.step(action)`.

**New**: a weapons-trigger decode. Add `decode_weapons_trigger(rates,
threshold) -> bool` (new function, `src/fly_brain/motor.py`) — fires when
`rates.get("loom", 0.0)` crosses `threshold`. Threshold value is a Section 6
tuning question (see below), not something to hardcode blindly.

---

## 6. Mission state machine (new — replaces v1's simple checkpoint loop)

```python
class MissionPhase(Enum):
    TAKEOFF = auto()   # scripted
    CRUISE = auto()    # brain-driven steering + weapons
    LANDING = auto()   # scripted
```

- **TAKEOFF**: scripted constant/ramped throttle + pitch-up, matching v1's
  own Section 8 caveat almost verbatim — hold this phase until airspeed/
  altitude crosses a threshold (confirm realistic F-16 takeoff numbers
  against the aircraft's actual JSBSim performance once flying it manually
  in Section 3's sanity check, not textbook F-16 numbers which assume a
  loaded combat aircraft, not necessarily this model's exact mass/thrust
  config).
- **CRUISE**: hand-off to the brain loop (Sections 4-5 above) once stable
  and above a safe altitude. Advance through a scripted waypoint list
  (reuse `fly_brain.course.CHECKPOINTS`/`advance_checkpoint`, tuned to
  real-world-scale waypoints instead of the small synthetic ones) same as
  v1's own checkpoint philosophy — the brain never "knows" about waypoints,
  it only ever sees bearing/distance as visual drive.
- **LANDING**: scripted approach/flare/touchdown once the last waypoint is
  reached, or a fuel/time budget expires.

Expect real trial-and-error tuning here — same honest expectation as v1's
own Section 6 ("this is trial and error, not a formula") — now compounded
by JSBSim's much higher physical fidelity than PyFlyt's simplified model.
Log DN rates during a manual (non-brain) flight first to see what a real
cruise profile produces before wiring the decoder in, same diagnostic
order as v1.

---

## Rough milestone order

1. Install FlightGear + F-16, confirm manual flight works (Section 3)
2. Telnet property-tree discovery pass, confirm exact paths/units (Section 4)
3. `FlightGearBridge` built + tested against a running FlightGear instance
4. Scripted-only mission loop (TAKEOFF -> CRUISE -> LANDING, no brain,
   hardcoded cruise controls) — isolates "is the bridge/mission logic
   broken" from "is the brain wiring broken", same reasoning as v1's own
   hardcoded-controller-first approach
5. Brain wired into CRUISE phase (steering), tune against real DN rates
6. Weapons trigger wired to loom detector, tuned
7. Full run, record, revisit `docs/files/fly_drone_sim.py`'s dashboard
   hook (`log_for_dashboard`) for the neural-activity overlay

## Explicitly out of scope for this rewrite

- Widening the neuPrint pull to recover a working DNg13 steering signal
  (Section 1 follow-up, independent of this rewrite).
- Any PyFlyt/MQ9 code — `docs/files/fly_drone_sim.py`, `scripts/watch_it_fly.py`,
  `src/fly_brain/mq9_skin.py`, `src/fly_brain/terrain.py`, `src/fly_brain/hud.py`,
  `assets/vehicles/mq9/` stay as-is (a separate, working PyFlyt demo) — this
  is a new parallel track, not a deletion of the old one, until FlightGear is
  confirmed working end-to-end.
