"""End-to-end: real fly connectome flies FlightGear's F-16.

Spawns already airborne in cruise (FlightGear's built-in "cruise" aircraft
state -- engine running, throttle set, gear up -- sidesteps the ground-
handling/takeoff problem PLAN.md always said fly reflexes can't cover),
then hands control to the brain: bearing/distance to a scripted waypoint
drives the visual encoder, descending-neuron spike rates decode to control
surfaces and a weapons trigger off the loom detector.

Run FlightGear first, in a separate terminal. Note --ai-scenario loads the
real 25-aircraft target scenario (Scenarios/moving_targets_around_KXTA.xml) --
--disable-ai-traffic would suppress it, so it's intentionally left off.
Spawn airport is KXTA itself (the scenario's real orbit center) -- KEDW,
used in an earlier version, is ~320km away, well outside render/visual
range, which is why the AI aircraft never appeared on screen even though
/ai/models/count correctly reported 25. --altitude=15000 matches the
scenario's real AI orbit altitude (Scenarios/moving_targets_around_KXTA.xml
sets <altitude-ft>15000</altitude-ft> on its aircraft) -- confirmed live
that spawning at 5000ft put terrain/cockpit in view but left every target
10,000ft directly overhead and invisible:
    /usr/games/fgfs --fg-aircraft=$HOME/.fgfs/Aircraft --aircraft=f16-simplified \\
      --airport=KXTA --runway=03L --state=cruise --altitude=15000 --vc=300 \\
      --telnet=5501 --disable-real-weather-fetch --ai-scenario=moving_targets_around_KXTA \\
      --prop:/sim/terrasync/enabled=true --timeofday=noon

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
MAX_STEPS = 400  # real per-step latency is network-round-trip bound (~12 telnet
# calls/step against live FlightGear), not CPU bound -- 6000 steps would run
# for hours; this is sized for a practical demo run instead.
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

# The LIF network's steering output has no roll damper of its own: a
# sustained one-direction aileron rate (the descending-neuron spike rate
# rarely flips sign quickly) banks the aircraft further and further with
# nothing opposing it, and an uncorrected bank bleeds altitude into an
# ever-tightening spiral dive -- confirmed live: FlightGear logged "hit
# Ocean and exploded" after ~150s of otherwise-normal brain-controlled
# cruise, not at handoff. A real aircraft's roll-rate damper/SAS supplies
# exactly this kind of correction; this is a scripted stand-in for it,
# layered on top of (not replacing) the brain's aileron output, the same
# pattern as PITCH_TRIM above.
ROLL_LIMIT_DEG = 20.0  # bank angle beyond which corrective aileron kicks in
ROLL_DAMP_GAIN = 0.05  # aileron units per degree of excess bank
ROLL_DAMP_LIMIT = 1.0
# Live test proved the soft damper above insufficient on its own: bank hit
# +103.4deg (well past inverted) because its correction was only ever added
# to the brain's own aileron command, which can still win the sum. Past this
# threshold, aileron/rudder authority is taken away from the brain entirely
# and driven to a hard wings-level recovery command instead -- the same
# "authority override" real fly-by-wire systems use for extreme-attitude
# recovery, not a subtle nudge. Elevator/throttle/weapons stay brain-owned.
ROLL_HARD_RECOVER_DEG = 60.0


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
    bridge.arm_weapons()
    ammo_start = int(float(bridge.get("/ai/submodels/submodel[2]/count")))
    print(f"Connected. Master-arm on, cannon ammo: {ammo_start} rounds.")

    telem = bridge.get_telemetry()
    origin_lat, origin_lon = telem["lat_deg"], telem["lon_deg"]
    target_altitude_m = telem["altitude_m"]
    print(f"Origin: lat={origin_lat:.6f} lon={origin_lon:.6f} alt={telem['altitude_m']:.1f}m")

    # The waypoint is the real AI target scenario's orbit center (see
    # Scenarios/moving_targets_around_KXTA.xml -- every target orbits
    # lat=37.258, lon=-115.797), not an arbitrary offset from spawn heading.
    # An earlier version picked waypoints relative to spawn heading alone,
    # which had no relationship to where the actual AI targets were --
    # confirmed live: the brain flew toward empty space while 25 real AI
    # aircraft orbited a different point entirely.
    TARGET_LAT_DEG = 37.258
    TARGET_LON_DEG = -115.797
    target_x, target_y = latlon_to_local_xy(TARGET_LAT_DEG, TARGET_LON_DEG, origin_lat, origin_lon)
    waypoints_local = [(target_x, target_y)]
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

        roll_deg = math.degrees(telem["roll_rad"])
        if abs(roll_deg) > ROLL_HARD_RECOVER_DEG:
            action[0] = -math.copysign(1.0, roll_deg)
            action[2] = 0.0
        else:
            excess_bank = max(0.0, abs(roll_deg) - ROLL_LIMIT_DEG)
            roll_damp = -math.copysign(1.0, roll_deg) * min(ROLL_DAMP_LIMIT, ROLL_DAMP_GAIN * excess_bank)
            action[0] = max(-1.0, min(1.0, action[0] + roll_damp))

        bridge.send_controls(*action)
        bridge.trigger_weapons(fire)
        if fire:
            fired_count += 1

        if dist < WAYPOINT_RADIUS_M:
            waypoint_index = min(waypoint_index + 1, len(waypoints_local) - 1)

        if step % 10 == 0:
            # Real ammo count from the sim, not just our own fire-flag
            # tally -- proves rounds actually left the aircraft rather than
            # trigger_weapons() being a no-op (see FlightGearBridge docstring).
            ammo_now = int(float(bridge.get("/ai/submodels/submodel[2]/count")))
            rounds_fired = ammo_start - ammo_now
            print(
                f"step {step}: phase={phase.name} wp={waypoint_index} bearing={math.degrees(bearing_err):+.1f}deg "
                f"dist={dist:.0f}m roll={roll_deg:+.1f}deg alt={telem['altitude_m']:.0f}m "
                f"action={[round(a, 2) for a in action]} fire={fire} rounds_fired={rounds_fired}"
            )

        time.sleep(1.0 / STEP_HZ)

    bridge.close()
    ammo_final = int(float(bridge.get("/ai/submodels/submodel[2]/count")))
    print(f"Done. Fired {ammo_start - ammo_final} real rounds of {ammo_start} over {MAX_STEPS} steps.")


if __name__ == "__main__":
    main()
