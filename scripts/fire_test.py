"""Weapons-visibility check only: no steering, no target tracking. Holds
wings-level straight flight and fires continuously so the gun effect can be
observed in isolation from the flight-control bug in simple_intercept.py.

Run FlightGear first (see scripts/fly_f16.py's docstring for the launch
command), then: .venv/bin/python scripts/fire_test.py
"""

import time

from fly_brain.flightgear_bridge import FlightGearBridge

STEP_HZ = 5
RUN_SECONDS = 30


def main():
    print("Connecting to FlightGear on localhost:5501...")
    bridge = FlightGearBridge(host="localhost", port=5501)
    bridge.arm_weapons()
    ammo_start = int(float(bridge.get("/ai/submodels/submodel[2]/count")))
    print(f"Connected. Cannon ammo: {ammo_start} rounds. Firing continuously for {RUN_SECONDS}s, wings level.")

    steps = RUN_SECONDS * STEP_HZ
    for step in range(steps):
        bridge.send_controls(aileron=0.0, elevator=0.0, rudder=0.0, throttle=0.7)
        bridge.trigger_weapons(True)
        if step % STEP_HZ == 0:
            ammo_now = int(float(bridge.get("/ai/submodels/submodel[2]/count")))
            print(f"t={step // STEP_HZ}s: ammo={ammo_now} rounds_fired={ammo_start - ammo_now}")
        time.sleep(1.0 / STEP_HZ)

    bridge.trigger_weapons(False)
    ammo_final = int(float(bridge.get("/ai/submodels/submodel[2]/count")))
    print(f"Done. Fired {ammo_start - ammo_final} rounds.")
    bridge.close()


if __name__ == "__main__":
    main()
