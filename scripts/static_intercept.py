"""Same bank-to-turn steering as simple_intercept.py, but toward a fixed
lat/lon point instead of a moving AI target -- isolates "can the control
law navigate to a point at all" from "can it out-turn a moving target,"
since spawning ~600m from an orbiting AI aircraft (simple_intercept.py's
setup) starts the geometry already close to worst-case for a wide-turn
pursuit law, and the target moving compounds that.

The waypoint is a fixed offset from wherever the aircraft spawns (read
once at startup), far enough out (see WAYPOINT_DIST_M) to give the turn
room to actually converge before arrival, the way a real intercept run
would set up a stand-off distance rather than merging immediately.

Run scripts/pick_target_and_fly.py first (only to deploy a valid protocol
file -- the AI target it locks onto is unused here, this script only reads
own-ship telemetry), then restart FlightGear, then run this script.
"""

import math
import time

from fly_brain.realtime_bridge import RealtimeBridge

WAYPOINT_DIST_M = 5000.0  # spawn a fixed point this far ahead, not right on top of it
WAYPOINT_RADIUS_M = 300.0  # consider "arrived" within this distance
STEP_HZ = 20
MAX_STEPS = 4800  # 4 minutes at 20Hz

# Cascaded roll control: outer loop picks a desired BANK ANGLE (capped);
# that feeds a middle loop that picks a desired ROLL RATE (also capped);
# an inner P controller drives aileron to hit that rate. This replaced a
# single-stage PD directly on bank-angle error, which was oscillating hard
# (aileron flipping -1.0/+1.0 every ~1s, confirmed live) -- P+D on angle
# alone has no notion of a sane rate to move at, so it kept overshooting.
# Bounding the intermediate rate command is what a real autopilot's roll
# channel does and is the actual fix, not just retuned gains on the same
# single-stage structure.
ROLL_MAX_DEG = 45.0  # never bank more than this
ROLL_RATE_MAX_DPS = 30.0  # never ask for a faster roll than this
K_BANK_TO_RATE = 0.7  # deg/s of desired roll rate per degree of bank error
K_RATE_TO_AILERON = 1.0 / ROLL_RATE_MAX_DPS  # full aileron at the rate cap

# Roll rate is finite-differenced from UDP telemetry samples that arrive at
# irregular real intervals (network/loop jitter, not a clean fixed dt) --
# feeding that raw derivative straight into the rate loop amplifies the
# jitter into exactly the -1.0/+1.0 flip-flopping seen live. A low-pass
# (EMA) filter on the rate estimate is the standard fix for a noisy
# derivative term; ROLL_RATE_FILTER is the smoothing factor (lower = more
# smoothing, more lag).
ROLL_RATE_FILTER_ALPHA = 0.3

# Altitude hold is reactive only -- no separate "turn pull" term fighting
# it. An explicit nose-up-proportional-to-bank term (tried first) reliably
# overpowered this loop even while already 3500ft too high, confirmed
# live, because it applied a bias unconditionally regardless of the actual
# altitude error. Letting bank-induced sink show up as alt_err_ft and
# relying on this loop's own gain to react to it is simpler and doesn't
# fight itself.
ALT_HOLD_GAIN = 1.0 / 2000.0  # elevator units per ft of altitude error


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def offset_point(lat, lon, heading_deg, dist_m):
    """A point dist_m ahead of (lat, lon) along heading_deg."""
    R = 6371000.0
    hr = math.radians(heading_deg)
    dr = dist_m / R
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(hr))
    lon2 = lon1 + math.atan2(
        math.sin(hr) * math.sin(dr) * math.cos(lat1), math.cos(dr) - math.sin(lat1) * math.sin(lat2)
    )
    return math.degrees(lat2), math.degrees(lon2)


def main():
    print("Connecting to FlightGear's real-time UDP link (5502/5503)...")
    rt = RealtimeBridge()
    telem = rt.get_telemetry()

    # 90deg off current heading, not straight ahead -- straight ahead
    # wouldn't exercise the turn at all, and the turn is the part that
    # actually failed against the moving target.
    waypoint_heading = (telem["heading_deg"] + 90.0) % 360.0
    target_lat, target_lon = offset_point(
        telem["lat_deg"], telem["lon_deg"], waypoint_heading, WAYPOINT_DIST_M
    )
    target_alt_ft = telem["altitude_m"] / 0.3048
    print(f"Waypoint: {target_lat:.5f}, {target_lon:.5f}, {WAYPOINT_DIST_M:.0f}m ahead")

    prev_roll_deg = telem["roll_deg"]
    prev_time = time.monotonic()
    filtered_roll_rate_dps = 0.0

    for step in range(MAX_STEPS):
        telem = rt.get_telemetry()
        own_lat, own_lon, own_alt_ft = telem["lat_deg"], telem["lon_deg"], telem["altitude_m"] / 0.3048
        heading_deg = telem["heading_deg"]

        dist = haversine_m(own_lat, own_lon, target_lat, target_lon)
        want_bearing = bearing_deg(own_lat, own_lon, target_lat, target_lon)
        bearing_err = angle_diff_deg(want_bearing, heading_deg)

        rudder = max(-0.3, min(0.3, bearing_err / 90.0))
        alt_err_ft = target_alt_ft - own_alt_ft
        elevator = max(-0.6, min(0.6, -alt_err_ft * ALT_HOLD_GAIN))

        now = time.monotonic()
        roll_deg = telem["roll_deg"]
        dt = now - prev_time
        raw_roll_rate_dps = (roll_deg - prev_roll_deg) / dt if dt > 0 else 0.0
        prev_roll_deg, prev_time = roll_deg, now
        filtered_roll_rate_dps += ROLL_RATE_FILTER_ALPHA * (raw_roll_rate_dps - filtered_roll_rate_dps)

        # Cascade: bank error -> desired rate (capped) -> aileron. See
        # ROLL_MAX_DEG/ROLL_RATE_MAX_DPS comment above.
        desired_roll_deg = max(-ROLL_MAX_DEG, min(ROLL_MAX_DEG, bearing_err * 1.5))
        roll_error_deg = desired_roll_deg - roll_deg
        desired_rate_dps = max(
            -ROLL_RATE_MAX_DPS, min(ROLL_RATE_MAX_DPS, K_BANK_TO_RATE * roll_error_deg)
        )
        rate_error_dps = desired_rate_dps - filtered_roll_rate_dps
        aileron = max(-1.0, min(1.0, K_RATE_TO_AILERON * rate_error_dps))

        if abs(roll_deg) > 90.0:
            elevator = 0.0

        rt.send_controls(aileron=aileron, elevator=elevator, rudder=rudder, throttle=1.0, fire=False)

        if step % 20 == 0:
            print(
                f"step {step}: dist={dist:.0f}m bearing_err={bearing_err:+.1f}deg "
                f"roll={roll_deg:+.1f}deg roll_rate={filtered_roll_rate_dps:+.1f}dps alt_err={alt_err_ft:+.0f}ft "
                f"action=[ail={aileron:.2f} elev={elevator:.2f} rud={rudder:.2f}]"
            )

        if dist < WAYPOINT_RADIUS_M:
            print(f"Arrived at waypoint, dist={dist:.0f}m, after {step} steps.")
            break

        time.sleep(1.0 / STEP_HZ)

    rt.close()


if __name__ == "__main__":
    main()
