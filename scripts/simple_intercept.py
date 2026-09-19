"""Simple scripted intercept: no fly-brain, no LIF network. Just proportional
steering toward the nearest real AI target and a fire command when roughly
aimed and close -- a control-path sanity check to isolate "can the aircraft
and weapons actually produce a visible effect" from "does the brain fly well."

Fully real-time: own-ship telemetry, the locked target's position, flight
controls, and the weapons trigger all go over FlightGear's generic UDP
protocol (RealtimeBridge) -- no telnet anywhere in this script.

Run scripts/pick_target_and_fly.py first (needs FlightGear already running,
uses telnet once to find the nearest AI target and deploy a UDP protocol
locked to it), then restart FlightGear, then run this script.
"""

import math
import time

from fly_brain.realtime_bridge import RealtimeBridge

FIRE_RANGE_M = 1500.0
FIRE_BEARING_DEG = 15.0
STEP_HZ = 20  # matches the generic protocol's --generic=...,20,... rate
MAX_STEPS = 4800  # 4 minutes at 20Hz

# Bank-to-turn steering: an OUTER loop picks a desired bank angle from
# bearing error, CAPPED at ROLL_MAX_DEG; an INNER PD loop drives aileron to
# reach and hold that bank (proportional on bank error, derivative on roll
# rate). This replaced a naive `aileron = bearing_err / 30` law that had no
# ceiling on commanded bank -- confirmed live it doesn't crash (the rate
# term keeps it from tumbling), but once the jet overshoots the target and
# bearing error grows past ~90deg, aileron pins near +-1.0 and it just
# barrel-rolls in place forever, distance climbing from 440m to 5689m over
# a 30s run, never actually turning back toward the target. A real aircraft
# doesn't bank harder as heading error grows past 90deg, it banks a bounded
# amount and lets the turn eat the heading error over time -- that's what
# the ROLL_MAX_DEG cap restores.
ROLL_MAX_DEG = 45.0  # never commands more bank than this, however large the bearing error
ROLL_KP = 0.035  # aileron units per degree of (desired - actual) bank error
ROLL_KD = 0.03  # aileron units per deg/s of roll rate (damping)


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


def main():
    print("Connecting to FlightGear's real-time UDP link (5502/5503)...")
    rt = RealtimeBridge()
    telem = rt.get_telemetry()
    dist = haversine_m(
        telem["lat_deg"], telem["lon_deg"], telem["target_lat_deg"], telem["target_lon_deg"]
    )
    print(f"Locked target: initial dist={dist:.0f}m")

    prev_roll_deg = telem["roll_deg"]
    prev_time = time.monotonic()

    for step in range(MAX_STEPS):
        telem = rt.get_telemetry()
        own_lat, own_lon, own_alt_ft = telem["lat_deg"], telem["lon_deg"], telem["altitude_m"] / 0.3048
        heading_deg = telem["heading_deg"]
        t_lat, t_lon, t_alt_ft = telem["target_lat_deg"], telem["target_lon_deg"], telem["target_alt_ft"]

        dist = haversine_m(own_lat, own_lon, t_lat, t_lon)
        want_bearing = bearing_deg(own_lat, own_lon, t_lat, t_lon)
        bearing_err = angle_diff_deg(want_bearing, heading_deg)

        rudder = max(-0.3, min(0.3, bearing_err / 90.0))
        alt_err_ft = t_alt_ft - own_alt_ft
        elevator = max(-0.5, min(0.5, -alt_err_ft / 5000.0))

        now = time.monotonic()
        roll_deg = telem["roll_deg"]
        dt = now - prev_time
        roll_rate_dps = (roll_deg - prev_roll_deg) / dt if dt > 0 else 0.0
        prev_roll_deg, prev_time = roll_deg, now

        # Outer loop: desired bank angle from bearing error, capped. Inner
        # loop: PD on (desired - actual) bank plus roll-rate damping. See
        # ROLL_MAX_DEG comment above for why the cap is the actual fix.
        desired_roll_deg = max(-ROLL_MAX_DEG, min(ROLL_MAX_DEG, bearing_err * 1.5))
        roll_error_deg = desired_roll_deg - roll_deg
        aileron = max(-1.0, min(1.0, ROLL_KP * roll_error_deg - ROLL_KD * roll_rate_dps))

        # Past 90deg bank, "elevator" is no longer pitch relative to the
        # horizon -- pulling it adds to the roll/dive instead of arresting
        # it (this is exactly what turned a recoverable bank into a 160deg
        # inverted dive in an earlier run). Cut pitch authority there.
        if abs(roll_deg) > 90.0:
            elevator = 0.0

        fire = dist < FIRE_RANGE_M and abs(bearing_err) < FIRE_BEARING_DEG

        rt.send_controls(aileron=aileron, elevator=elevator, rudder=rudder, throttle=1.0, fire=fire)

        if step % 20 == 0:
            print(
                f"step {step}: dist={dist:.0f}m bearing_err={bearing_err:+.1f}deg "
                f"roll={roll_deg:+.1f}deg roll_rate={roll_rate_dps:+.1f}dps alt_err={alt_err_ft:+.0f}ft "
                f"action=[ail={aileron:.2f} elev={elevator:.2f} rud={rudder:.2f}] fire={fire}"
            )

        time.sleep(1.0 / STEP_HZ)

    rt.close()


if __name__ == "__main__":
    main()
