"""End-to-end: real fly connectome flies FlightGear's F-16.

Spawns already airborne in cruise (FlightGear's built-in "cruise" aircraft
state -- engine running, throttle set, gear up -- sidesteps the ground-
handling/takeoff problem PLAN.md always said fly reflexes can't cover),
then hands control to the brain: bearing/distance to a scripted waypoint
drives the visual encoder, descending-neuron spike rates decode to control
surfaces and a weapons trigger off the loom detector.

Run FlightGear first, in a separate terminal:
    /usr/games/fgfs --fg-aircraft=$HOME/.fgfs/Aircraft --aircraft=f16-simplified \\
      --airport=KEDW --runway=22 --state=cruise --altitude=5000 --vc=300 \\
      --telnet=5501 --disable-real-weather-fetch --disable-ai-traffic \\
      --prop:/sim/terrasync/enabled=false --timeofday=noon

Then: .venv/bin/python scripts/fly_f16.py
"""

import math
import time

import torch

from fly_brain.brain import LIFBrain
from fly_brain.connectome import load_connectome, load_neuron_indices
from fly_brain.course import bearing_and_distance, latlon_to_local_xy
from fly_brain.encoding import encode_visual
from fly_brain.flightgear_bridge import FlightGearBridge
from fly_brain.mission import MissionPhase, next_phase
from fly_brain.motor import RateTracker, decode_controls, decode_weapons_trigger

CONNECTIVITY_PATH = "data/connectivity.parquet"
ANNOTATIONS_PATH = "data/annotations.parquet"

# Waypoints as (east_m, north_m) offsets from wherever the aircraft spawns.
# Computed at runtime relative to the actual spawn heading (see main()) so
# the first waypoint starts roughly ahead of the aircraft rather than at an
# arbitrary compass bearing that might be behind it -- an early run with a
# fixed offset put the first waypoint almost directly behind the spawn
# heading, and the aircraft (with no steering signal active yet at the
# time) flew straight into terrain trying to just cruise on. Second leg
# is a gentle turn, not another near-reversal.
WAYPOINT_RADIUS_M = 500.0
# encode_visual()'s closeness = 1/(1+distance) was designed/tested at the
# PyFlyt-era scale (checkpoints tens of meters away) -- at FlightGear's real
# distances (thousands of meters) closeness collapses to ~0 and NOTHING
# fires anywhere in the network, confirmed by an offline replay showing
# zero total spikes at raw distances. Scaling distance down before encoding
# is a sensor-calibration choice, not a change to the tested function.
DISTANCE_SCALE = 1000.0
MIN_TAKEOFF_ALTITUDE_M = 0.0  # already airborne (--state=cruise) -- skip TAKEOFF
MIN_TAKEOFF_AIRSPEED_MPS = 0.0
FIRE_THRESHOLD = 0.5
MAX_STEPS = 6000
STEP_HZ = 10  # brain/control tick rate; FlightGear itself runs its own physics loop

# No real pitch descending neuron exists in the pulled connectome data (see
# PLAN.md's "DNp_pitch left unmapped" note), so decode_controls's elevator
# output is always 0 -- nothing counteracts gradual sink/climb. A scripted
# altitude-hold trim added on top of (not replacing) the brain's elevator
# output is the honest fix: the brain still fully owns aileron/rudder/
# throttle/weapons, this only supplies what a real aircraft's trim system
# would (and what a working pitch DN would, if the connectome pull is
# widened for it later).
PITCH_TRIM_GAIN = 0.0003  # elevator units per meter of altitude error
PITCH_TRIM_LIMIT = 0.2


def heading_deg_to_math_rad(heading_deg):
    """FlightGear compass heading (0=North, clockwise) -> math convention
    (0=East, counter-clockwise) that bearing_and_distance expects. See the
    docstring note on FlightGearBridge.get_telemetry() for why this exists."""
    return math.radians(90.0 - heading_deg)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading connectome...")
    W, n_neurons = load_connectome(CONNECTIVITY_PATH, device=device)
    photoreceptor_idx, descending_idx = load_neuron_indices(ANNOTATIONS_PATH)
    print(f"  {n_neurons} neurons, {len(photoreceptor_idx)} photoreceptors, descending: {descending_idx}")

    brain = LIFBrain(W, n_neurons, device=device)
    rate_tracker = RateTracker(descending_idx)

    print("Connecting to FlightGear on localhost:5501 (make sure it's running)...")
    bridge = FlightGearBridge(host="localhost", port=5501)
    print("Connected.")

    telem = bridge.get_telemetry()
    origin_lat, origin_lon = telem["lat_deg"], telem["lon_deg"]
    target_altitude_m = telem["altitude_m"]
    spawn_heading_math_rad = heading_deg_to_math_rad(math.degrees(telem["heading_rad"]))
    print(f"Origin: lat={origin_lat:.6f} lon={origin_lon:.6f} alt={telem['altitude_m']:.1f}m")

    # Waypoints relative to actual spawn heading: mostly ahead, gentle turns.
    def ahead(distance_m, side_offset_m):
        fx = math.cos(spawn_heading_math_rad)
        fy = math.sin(spawn_heading_math_rad)
        sx, sy = -fy, fx  # left of heading
        return (fx * distance_m + sx * side_offset_m, fy * distance_m + sy * side_offset_m)

    waypoints_local = [ahead(4000.0, 1000.0), ahead(6000.0, -2000.0)]
    print(f"Waypoints (local m, relative to spawn): {waypoints_local}")

    phase = MissionPhase.CRUISE  # already airborne, no scripted TAKEOFF needed
    waypoint_index = 0
    fired_count = 0

    for step in range(MAX_STEPS):
        telem = bridge.get_telemetry()
        x_m, y_m = latlon_to_local_xy(telem["lat_deg"], telem["lon_deg"], origin_lat, origin_lon)
        heading_math_rad = heading_deg_to_math_rad(math.degrees(telem["heading_rad"]))

        waypoints_remaining = len(waypoints_local) - waypoint_index
        phase = next_phase(
            phase,
            altitude_m=telem["altitude_m"],
            airspeed_mps=telem["airspeed_mps"],
            waypoints_remaining=waypoints_remaining,
            min_takeoff_altitude_m=MIN_TAKEOFF_ALTITUDE_M,
            min_takeoff_airspeed_mps=MIN_TAKEOFF_AIRSPEED_MPS,
        )

        if phase == MissionPhase.LANDING:
            # Scripted: wings level, throttle back, hold heading. Full
            # autonomous flare/touchdown is out of scope -- ground handling
            # isn't something fly reflexes cover (PLAN.md's own caveat,
            # unchanged since v1).
            bridge.send_controls(aileron=0.0, elevator=0.0, rudder=0.0, throttle=0.3)
            bridge.trigger_weapons(False)
            print(f"step {step}: LANDING (scripted) alt={telem['altitude_m']:.0f}m speed={telem['airspeed_mps']:.0f}m/s")
            time.sleep(1.0 / STEP_HZ)
            continue

        target_x, target_y = waypoints_local[waypoint_index]
        bearing_err, dist = bearing_and_distance(
            position=(x_m, y_m, 0.0), heading=heading_math_rad, checkpoint=(target_x, target_y, 0.0)
        )

        ext_current = encode_visual(bearing_err, dist / DISTANCE_SCALE, photoreceptor_idx, n_neurons, device)
        S = brain.step(external_current=ext_current)
        rates = rate_tracker.update(S)
        action = decode_controls(rates)
        fire = decode_weapons_trigger(rates, threshold=FIRE_THRESHOLD)

        pitch_trim = max(-PITCH_TRIM_LIMIT, min(PITCH_TRIM_LIMIT,
                          PITCH_TRIM_GAIN * (target_altitude_m - telem["altitude_m"])))
        action[1] = max(-1.0, min(1.0, action[1] + pitch_trim))

        bridge.send_controls(*action)
        bridge.trigger_weapons(fire)
        if fire:
            fired_count += 1

        if dist < WAYPOINT_RADIUS_M:
            waypoint_index = min(waypoint_index + 1, len(waypoints_local) - 1)

        if step % 10 == 0:
            print(
                f"step {step}: phase={phase.name} wp={waypoint_index} bearing={math.degrees(bearing_err):+.1f}deg "
                f"dist={dist:.0f}m action={[round(a, 2) for a in action]} fire={fire} fired_total={fired_count}"
            )

        time.sleep(1.0 / STEP_HZ)

    bridge.close()
    print(f"Done. Fired weapons on {fired_count} of {MAX_STEPS} steps.")


if __name__ == "__main__":
    main()
