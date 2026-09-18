# FlightGear F-16 Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pieces the fly-brain needs to control FlightGear's real F-16 over its telnet property-tree protocol: a local lat/lon-to-meters conversion, the telnet bridge itself, a weapons-trigger decoder, and a pure mission-phase state machine — each unit-tested in isolation (no live FlightGear process required for the automated test suite).

**Architecture:** One new module per responsibility, matching the existing `fly_brain` package's pattern (`course.py`/`motor.py` gain new functions; a new `flightgear_bridge.py` and `mission.py`). The bridge is tested against an in-process mock telnet server that replays FlightGear's exact confirmed wire format — real FlightGear is only needed for a final manual smoke run, not for the automated suite.

**Tech Stack:** Python 3, stdlib `socket`/`threading` only (no new pip dependency — the plan's own design doc ruled out a telnet-client package for a protocol this simple). Existing venv already has pytest/torch/numpy/pandas.

**Spec:** `docs/files/PLAN.md` (v2, FlightGear/F-16 rewrite) — Sections 4-6 specifically. This plan implements the confirmed property paths and wire format documented there from a live telnet discovery session; do not re-derive or second-guess those confirmed values.

## Global Constraints

- Package: add to the existing `fly_brain` package (`src/fly_brain/`), same conventions as existing modules — pure functions/classes, no module-level mutable state, no `print`/logging inside compute functions, float32/float consistency per existing style (this plan's new code deals in plain Python floats, not tensors, except where explicitly noted).
- All new/modified Python files pass `.venv/bin/python -m py_compile`; tests pass under `.venv/bin/pytest`.
- No task in this plan's automated tests may require a live FlightGear process, network access, or GPU. The mock telnet server (Task 2) runs in-process on `localhost` via a loopback socket — that's local IPC, not "the network" in the sense this constraint means to exclude (no external host, no real FlightGear instance).
- Confirmed FlightGear wire format (from a live telnet session against `f16-simplified`, FlightGear 2020.3.19 — do not deviate from this format in the mock server or the bridge's parser):
  - Every `get <path>` and `set <path> <value>` command gets a reply of the exact form `{path} = '{value}' ({type})\r\n` followed immediately by the prompt `/> ` (slash, greater-than, space, no trailing newline after the space).
  - `ls <path>` returns multiple `{name} = '{value}' ({type})\r\n` lines (one per child property, using the property's own bare name not the full path) followed by the same `/> ` prompt.
  - Commands are sent as `{command}\r\n`.
- Confirmed real property paths (do not guess others; only these are needed for this plan):
  - Controls (write): `/controls/flight/aileron`, `/controls/flight/elevator`, `/controls/flight/rudder` (all `double`, -1..1), `/controls/engines/engine[0]/throttle` (`double`), `/controls/engines/engine[0]/cutoff` (`bool`, must be set `false` before throttle works), `/controls/armament/trigger` (`bool`).
  - Telemetry (read): `/position/longitude-deg`, `/position/latitude-deg`, `/position/altitude-ft` (all `double`), `/orientation/heading-deg`, `/orientation/pitch-deg`, `/orientation/roll-deg` (all `double`), `/velocities/airspeed-kt` (`double`).
- Unit conversions this plan's code must apply (bridge returns SI units — meters, radians, m/s — never feet/degrees/knots to callers): altitude feet→meters `* 0.3048`; knots→m/s `* 0.514444`; degrees→radians `math.radians(...)`.

---

### Task 1: Local lat/lon to flat x/y conversion

**Files:**
- Modify: `src/fly_brain/course.py` (add a new function; existing `bearing_and_distance`, `bearing_and_distance_from_body_frame_delta`, `advance_checkpoint`, `CHECKPOINTS`, `CHECKPOINT_RADIUS` stay untouched)
- Test: `tests/test_course.py` (add new test functions; existing tests stay untouched)

**Interfaces:**
- Consumes: nothing new.
- Produces: `fly_brain.course.latlon_to_local_xy(lat_deg: float, lon_deg: float, origin_lat_deg: float, origin_lon_deg: float) -> tuple[float, float]` returning `(x_m, y_m)` — x is east-offset in meters, y is north-offset in meters, from `origin_lat_deg`/`origin_lon_deg` to `lat_deg`/`lon_deg`, using an equirectangular (flat-Earth) approximation. This is consumed by Task 4 (mission state machine) and the eventual live-FlightGear wiring (out of scope for this plan) to turn FlightGear's lat/lon telemetry into the `(x, y)` meters `bearing_and_distance` already expects.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_course.py`:

```python
from fly_brain.course import latlon_to_local_xy
import math


def test_latlon_same_point_returns_origin():
    x, y = latlon_to_local_xy(37.6, -122.4, 37.6, -122.4)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6


def test_latlon_one_degree_north_is_about_111km():
    x, y = latlon_to_local_xy(38.6, -122.4, 37.6, -122.4)
    assert abs(x) < 1.0
    assert 110_000 < y < 112_000


def test_latlon_east_offset_scales_with_cos_latitude():
    origin_lat, origin_lon = 37.6, -122.4
    x, y = latlon_to_local_xy(origin_lat, origin_lon + 1.0, origin_lat, origin_lon)
    expected_x = math.radians(1.0) * 6371000.0 * math.cos(math.radians(origin_lat))
    assert abs(x - expected_x) < 1.0
    assert abs(y) < 1.0


def test_latlon_south_and_west_are_negative():
    x, y = latlon_to_local_xy(36.6, -123.4, 37.6, -122.4)
    assert x < 0
    assert y < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_course.py -v -k latlon`
Expected: FAIL — `ImportError: cannot import name 'latlon_to_local_xy'`.

- [ ] **Step 3: Implement `latlon_to_local_xy`**

Add to `src/fly_brain/course.py` (append to the end of the file; keep the existing `import numpy as np` at the top, add `import math` alongside it):

```python
_EARTH_RADIUS_M = 6371000.0


def latlon_to_local_xy(lat_deg, lon_deg, origin_lat_deg, origin_lon_deg):
    """Equirectangular approximation: fine over a mission-scale area of a
    few km, not a real geodesic calculation. x = east offset, y = north
    offset, both meters, from the origin to (lat_deg, lon_deg)."""
    lat_rad = math.radians(lat_deg)
    origin_lat_rad = math.radians(origin_lat_deg)
    dlat = math.radians(lat_deg - origin_lat_deg)
    dlon = math.radians(lon_deg - origin_lon_deg)

    x = dlon * _EARTH_RADIUS_M * math.cos(origin_lat_rad)
    y = dlat * _EARTH_RADIUS_M
    return x, y
```

Also add `import math` at the top of `src/fly_brain/course.py` alongside the existing `import numpy as np`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_course.py -v -k latlon`
Expected: 4 passed.

- [ ] **Step 5: Run the full existing suite to confirm nothing broke**

Run: `.venv/bin/pytest -q`
Expected: 41 passed (37 existing + 4 new).

- [ ] **Step 6: Commit**

```bash
git add src/fly_brain/course.py tests/test_course.py
git commit -m "feat: add lat/lon to local flat x/y conversion for FlightGear telemetry"
```

---

### Task 2: FlightGear telnet bridge

**Files:**
- Create: `src/fly_brain/flightgear_bridge.py`
- Test: `tests/test_flightgear_bridge.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `fly_brain.flightgear_bridge.FlightGearBridge` — constructor `FlightGearBridge(host="localhost", port=5501, timeout=5.0)` opens a TCP socket and blocks until the initial `/> ` prompt is read. Methods: `.get(path: str) -> str` (returns the raw value string, e.g. `"0.5"` or `"true"`), `.set(path: str, value) -> None`, `.get_telemetry() -> dict` (keys: `"x_m"`, `"y_m"` are NOT computed here — this method returns raw telemetry only: `"lat_deg"`, `"lon_deg"`, `"altitude_m"`, `"heading_rad"`, `"pitch_rad"`, `"roll_rad"`, `"airspeed_mps"`, all floats, SI/radians per Global Constraints), `.send_controls(aileron: float, elevator: float, rudder: float, throttle: float) -> None`, `.trigger_weapons(fire: bool) -> None`, `.close() -> None`.

- [ ] **Step 1: Write the mock telnet server fixture and failing tests**

Create `tests/test_flightgear_bridge.py`:

```python
import socket
import threading

import pytest

from fly_brain.flightgear_bridge import FlightGearBridge

# In-memory property store the mock server answers from, mimicking
# FlightGear's confirmed wire format exactly (see plan Global Constraints):
# every get/set reply is "{path} = '{value}' ({type})\r\n/> ".
_MOCK_PROPS = {
    "/controls/flight/aileron": ("0", "double"),
    "/controls/flight/elevator": ("0", "double"),
    "/controls/flight/rudder": ("0", "double"),
    "/controls/engines/engine[0]/throttle": ("0", "double"),
    "/controls/engines/engine[0]/cutoff": ("true", "bool"),
    "/controls/armament/trigger": ("false", "bool"),
    "/position/longitude-deg": ("-122.3933734", "double"),
    "/position/latitude-deg": ("37.62870187", "double"),
    "/position/altitude-ft": ("1000.0", "double"),
    "/orientation/heading-deg": ("90.0", "double"),
    "/orientation/pitch-deg": ("2.0", "double"),
    "/orientation/roll-deg": ("-1.0", "double"),
    "/velocities/airspeed-kt": ("150.0", "double"),
}


def _run_mock_server(sock, props):
    conn, _ = sock.accept()
    conn.sendall(b"/> ")  # initial banner/prompt, same as real FlightGear
    buf = b""
    with conn:
        while True:
            try:
                chunk = conn.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\r\n" in buf:
                line, buf = buf.split(b"\r\n", 1)
                line = line.decode()
                parts = line.split(" ", 2)
                cmd = parts[0]
                if cmd == "get":
                    path = parts[1]
                    value, typ = props.get(path, ("", "none"))
                    conn.sendall(f"{path} = '{value}' ({typ})\r\n/> ".encode())
                elif cmd == "set":
                    path, value = parts[1], parts[2]
                    _, typ = props.get(path, ("", "double"))
                    props[path] = (value, typ)
                    conn.sendall(f"{path} = '{value}' ({typ})\r\n/> ".encode())
                else:
                    conn.sendall(b"\r\n/> ")


@pytest.fixture
def mock_fg_server():
    props = dict(_MOCK_PROPS)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    thread = threading.Thread(target=_run_mock_server, args=(sock, props), daemon=True)
    thread.start()
    yield port, props
    sock.close()


def test_connect_reads_initial_prompt(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    bridge.close()


def test_get_returns_raw_value_string(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    assert bridge.get("/controls/flight/aileron") == "0"
    bridge.close()


def test_set_then_get_round_trips(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    bridge.set("/controls/flight/aileron", 0.5)
    assert bridge.get("/controls/flight/aileron") == "0.5"
    bridge.close()


def test_get_telemetry_converts_units(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    telem = bridge.get_telemetry()
    assert telem["lat_deg"] == pytest.approx(37.62870187)
    assert telem["lon_deg"] == pytest.approx(-122.3933734)
    assert telem["altitude_m"] == pytest.approx(1000.0 * 0.3048)
    assert telem["heading_rad"] == pytest.approx(1.5707963, abs=1e-4)  # 90 deg
    assert telem["pitch_rad"] == pytest.approx(0.0349066, abs=1e-4)  # 2 deg
    assert telem["roll_rad"] == pytest.approx(-0.0174533, abs=1e-4)  # -1 deg
    assert telem["airspeed_mps"] == pytest.approx(150.0 * 0.514444, abs=1e-3)
    bridge.close()


def test_send_controls_writes_all_four_properties(mock_fg_server):
    port, props = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    bridge.send_controls(aileron=0.3, elevator=-0.2, rudder=0.1, throttle=0.8)
    assert bridge.get("/controls/flight/aileron") == "0.3"
    assert bridge.get("/controls/flight/elevator") == "-0.2"
    assert bridge.get("/controls/flight/rudder") == "0.1"
    assert bridge.get("/controls/engines/engine[0]/throttle") == "0.8"
    bridge.close()


def test_trigger_weapons_true_and_false(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    bridge.trigger_weapons(True)
    assert bridge.get("/controls/armament/trigger") == "true"
    bridge.trigger_weapons(False)
    assert bridge.get("/controls/armament/trigger") == "false"
    bridge.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_flightgear_bridge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fly_brain.flightgear_bridge'`.

- [ ] **Step 3: Implement `FlightGearBridge`**

Create `src/fly_brain/flightgear_bridge.py`:

```python
"""Bridge to a live FlightGear instance over its telnet property-tree
protocol (PLAN.md Section 4). Confirmed wire format, live telnet session
against f16-simplified, FlightGear 2020.3.19: every get/set reply is
"{path} = '{value}' ({type})\\r\\n" followed by the prompt "/> "."""

import math
import socket

_FT_TO_M = 0.3048
_KT_TO_MPS = 0.514444


class FlightGearBridge:
    def __init__(self, host="localhost", port=5501, timeout=5.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._buf = b""
        self._read_until_prompt()

    def _read_until_prompt(self):
        while not self._buf.endswith(b"/> "):
            chunk = self._sock.recv(4096)
            if not chunk:
                break
            self._buf += chunk
        response = self._buf
        self._buf = b""
        return response.decode(errors="replace")

    def _send(self, line):
        self._sock.sendall((line + "\r\n").encode())
        return self._read_until_prompt()

    def get(self, path):
        response = self._send(f"get {path}")
        return response.split("'", 2)[1]

    def set(self, path, value):
        self._send(f"set {path} {value}")

    def get_telemetry(self):
        return {
            "lat_deg": float(self.get("/position/latitude-deg")),
            "lon_deg": float(self.get("/position/longitude-deg")),
            "altitude_m": float(self.get("/position/altitude-ft")) * _FT_TO_M,
            "heading_rad": math.radians(float(self.get("/orientation/heading-deg"))),
            "pitch_rad": math.radians(float(self.get("/orientation/pitch-deg"))),
            "roll_rad": math.radians(float(self.get("/orientation/roll-deg"))),
            "airspeed_mps": float(self.get("/velocities/airspeed-kt")) * _KT_TO_MPS,
        }

    def send_controls(self, aileron, elevator, rudder, throttle):
        self.set("/controls/flight/aileron", aileron)
        self.set("/controls/flight/elevator", elevator)
        self.set("/controls/flight/rudder", rudder)
        self.set("/controls/engines/engine[0]/throttle", throttle)

    def trigger_weapons(self, fire):
        self.set("/controls/armament/trigger", "true" if fire else "false")

    def close(self):
        self._sock.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_flightgear_bridge.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run the full suite to confirm nothing broke**

Run: `.venv/bin/pytest -q`
Expected: 48 passed (41 from Task 1 + 7 new).

- [ ] **Step 6: Commit**

```bash
git add src/fly_brain/flightgear_bridge.py tests/test_flightgear_bridge.py
git commit -m "feat: add FlightGear telnet bridge, tested against a mock server"
```

---

### Task 3: Weapons-trigger decoder

**Files:**
- Modify: `src/fly_brain/motor.py` (add a new function; existing `RateTracker`, `decode_controls` stay untouched)
- Test: `tests/test_motor.py` (add new test functions; existing tests stay untouched)

**Interfaces:**
- Consumes: nothing new — same `rates: dict[str, float]` shape `RateTracker.update()` already produces (Task 2/existing code).
- Produces: `fly_brain.motor.decode_weapons_trigger(rates: dict, threshold: float = 0.5) -> bool` — `True` when `rates.get("loom", 0.0) >= threshold`, consumed by the eventual live mission loop (out of scope for this plan) to call `FlightGearBridge.trigger_weapons(...)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_motor.py`:

```python
from fly_brain.motor import decode_weapons_trigger


def test_weapons_trigger_fires_above_threshold():
    assert decode_weapons_trigger({"loom": 0.8}, threshold=0.5) is True


def test_weapons_trigger_does_not_fire_below_threshold():
    assert decode_weapons_trigger({"loom": 0.3}, threshold=0.5) is False


def test_weapons_trigger_fires_exactly_at_threshold():
    assert decode_weapons_trigger({"loom": 0.5}, threshold=0.5) is True


def test_weapons_trigger_defaults_to_no_fire_when_loom_missing():
    assert decode_weapons_trigger({}, threshold=0.5) is False


def test_weapons_trigger_uses_default_threshold():
    assert decode_weapons_trigger({"loom": 0.6}) is True
    assert decode_weapons_trigger({"loom": 0.4}) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_motor.py -v -k weapons_trigger`
Expected: FAIL — `ImportError: cannot import name 'decode_weapons_trigger'`.

- [ ] **Step 3: Implement `decode_weapons_trigger`**

Append to `src/fly_brain/motor.py`:

```python
def decode_weapons_trigger(rates, threshold=0.5):
    """Fires when the loom-detector's spike rate crosses threshold -- a
    fly's "something's closing fast" reflex repurposed as a fire-control
    trigger (PLAN.md Section 5)."""
    return rates.get("loom", 0.0) >= threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_motor.py -v -k weapons_trigger`
Expected: 5 passed.

- [ ] **Step 5: Run the full suite to confirm nothing broke**

Run: `.venv/bin/pytest -q`
Expected: 53 passed (48 from Task 2 + 5 new).

- [ ] **Step 6: Commit**

```bash
git add src/fly_brain/motor.py tests/test_motor.py
git commit -m "feat: add weapons-trigger decoder off the loom-detector rate"
```

---

### Task 4: Mission phase state machine

**Files:**
- Create: `src/fly_brain/mission.py`
- Test: `tests/test_mission.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure state-transition logic, deliberately independent of `FlightGearBridge` so it's testable without any live/mock server).
- Produces: `fly_brain.mission.MissionPhase` (an `enum.Enum` with members `TAKEOFF`, `CRUISE`, `LANDING`), `fly_brain.mission.next_phase(current_phase: MissionPhase, altitude_m: float, airspeed_mps: float, waypoints_remaining: int, min_takeoff_altitude_m: float = 150.0, min_takeoff_airspeed_mps: float = 60.0) -> MissionPhase` — pure function, the eventual live mission loop (out of scope for this plan) calls this each tick with the bridge's telemetry to decide which phase's logic (scripted vs. brain-driven) to run next.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mission.py`:

```python
from fly_brain.mission import MissionPhase, next_phase


def test_stays_in_takeoff_below_altitude_threshold():
    phase = next_phase(
        MissionPhase.TAKEOFF, altitude_m=50.0, airspeed_mps=70.0, waypoints_remaining=3
    )
    assert phase == MissionPhase.TAKEOFF


def test_stays_in_takeoff_below_airspeed_threshold():
    phase = next_phase(
        MissionPhase.TAKEOFF, altitude_m=200.0, airspeed_mps=40.0, waypoints_remaining=3
    )
    assert phase == MissionPhase.TAKEOFF


def test_advances_from_takeoff_to_cruise_once_airborne_and_fast_enough():
    phase = next_phase(
        MissionPhase.TAKEOFF, altitude_m=200.0, airspeed_mps=70.0, waypoints_remaining=3
    )
    assert phase == MissionPhase.CRUISE


def test_stays_in_cruise_while_waypoints_remain():
    phase = next_phase(
        MissionPhase.CRUISE, altitude_m=1000.0, airspeed_mps=150.0, waypoints_remaining=2
    )
    assert phase == MissionPhase.CRUISE


def test_advances_from_cruise_to_landing_when_no_waypoints_remain():
    phase = next_phase(
        MissionPhase.CRUISE, altitude_m=1000.0, airspeed_mps=150.0, waypoints_remaining=0
    )
    assert phase == MissionPhase.LANDING


def test_landing_is_terminal():
    phase = next_phase(
        MissionPhase.LANDING, altitude_m=0.0, airspeed_mps=0.0, waypoints_remaining=0
    )
    assert phase == MissionPhase.LANDING


def test_custom_thresholds_are_respected():
    phase = next_phase(
        MissionPhase.TAKEOFF,
        altitude_m=80.0,
        airspeed_mps=65.0,
        waypoints_remaining=1,
        min_takeoff_altitude_m=50.0,
        min_takeoff_airspeed_mps=60.0,
    )
    assert phase == MissionPhase.CRUISE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_mission.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fly_brain.mission'`.

- [ ] **Step 3: Implement `MissionPhase` and `next_phase`**

Create `src/fly_brain/mission.py`:

```python
"""Mission phase state machine (PLAN.md Section 6): scripted TAKEOFF and
LANDING, brain-driven CRUISE. Deliberately has no dependency on
FlightGearBridge or the brain modules -- it's pure transition logic over
telemetry values the caller already has."""

from enum import Enum, auto


class MissionPhase(Enum):
    TAKEOFF = auto()
    CRUISE = auto()
    LANDING = auto()


def next_phase(
    current_phase,
    altitude_m,
    airspeed_mps,
    waypoints_remaining,
    min_takeoff_altitude_m=150.0,
    min_takeoff_airspeed_mps=60.0,
):
    if current_phase == MissionPhase.TAKEOFF:
        if altitude_m >= min_takeoff_altitude_m and airspeed_mps >= min_takeoff_airspeed_mps:
            return MissionPhase.CRUISE
        return MissionPhase.TAKEOFF

    if current_phase == MissionPhase.CRUISE:
        if waypoints_remaining <= 0:
            return MissionPhase.LANDING
        return MissionPhase.CRUISE

    return MissionPhase.LANDING
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_mission.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run the full suite to confirm nothing broke**

Run: `.venv/bin/pytest -q`
Expected: 60 passed (53 from Task 3 + 7 new).

- [ ] **Step 6: Commit**

```bash
git add src/fly_brain/mission.py tests/test_mission.py
git commit -m "feat: add mission phase state machine (scripted takeoff/landing, brain cruise)"
```

---

## Out of Scope (follow-up work, not this plan)

- The live mission-loop script wiring `FlightGearBridge` + `LIFBrain` + `MissionPhase` + real connectome data together into a runnable `scripts/fly_f16.py` — this needs a live FlightGear process to test against manually (30-45s startup, per PLAN.md's confirmed findings) and real trial-and-error control-gain tuning, neither of which fits an automated TDD task loop. A follow-up plan (or direct iterative work against a running FlightGear instance) should cover it once these four building blocks are in place.
- The engine-start sequence (cutoff/starter/magnetos) and a real runway spawn position for TAKEOFF — PLAN.md's Section 3 findings flagged both as real gaps (cold-and-dark start, KSFO's default parking spot is over water) that the live mission loop needs to handle, not something to fake in a unit test.
- Widening `scripts/pull_connectome.py`'s neuron-type list to recover a working DNg13 steering signal — independent follow-up, per PLAN.md Section 1.
