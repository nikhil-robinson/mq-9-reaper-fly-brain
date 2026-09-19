"""Fire a real AIM-9L missile (not the cannon): no steering, wings level,
isolates "does a missile actually launch and render" from flight control.

Root cause chain found live, via telnet (see FlightGearBridge for how `run
nasal script=...` executes real Nasal against the running sim):
  1. The default f16-simplified loadout carries CATM-9L on the left wingtip
     pylon -- a captive TRAINING round, intentionally inert, never
     selectable as a fireable weapon by the aircraft's own fire-control
     Nasal code (confirmed live: cycling weapon types always landed back on
     the cannon, no matter how many times cycled).
  2. Fix: telnet `set /payload/weight/selected` to "1 x AIM-9L" (a real
     option already listed under that pylon's /payload/weight/opt[3]/name)
     swaps in a live round -- confirmed live via the pylon's own
     getWeapons() showing a real loaded AIM-9L object afterward.
  3. Firing missiles (unlike the cannon) is NOT a raw JSBSim FCS property
     like guntrigger -- it only happens through the aircraft's Nasal
     fire-control state machine, reached by calling
     pylons.fcs.selectWeapon("AIM-9L") over telnet's `run nasal script=...`
     (confirmed working: FlightGear's telnet server supports executing
     arbitrary Nasal via `run nasal script=<code>`, undocumented in --help
     but present in this FlightGear 2020.3.19 build), then the normal
     controls/armament/trigger property.
  4. Confirmed live: /ai/models/model-added and model-removed both showed
     '/ai/models/aim-9l', and /ai/models/munition[0] recorded a real
     impact record with name='AIM-9L' -- the missile is genuinely modeled
     and simulated, it's just not counted in /ai/models/count (that only
     tracks scripted AI aircraft, not player-launched submodels).

Run FlightGear first (see scripts/fly_f16.py's docstring for the launch
command), then: .venv/bin/python scripts/fire_missile.py
"""

import time

from fly_brain.flightgear_bridge import FlightGearBridge

PYLON_PROPERTY = "/payload/weight/selected"
LIVE_ROUND_OPTION = "1 x AIM-9L"
WEAPON_TYPE = "AIM-9L"


def run_nasal(bridge, script):
    """Execute Nasal directly against the live sim via telnet's `run nasal
    script=<code>` -- not in --help's short command list, but supported by
    this FlightGear build. The script must contain no literal spaces (the
    telnet line parser splits on whitespace); Nasal doesn't need them."""
    bridge._send(f"run nasal script={script}")


def main():
    print("Connecting to FlightGear on localhost:5501...")
    bridge = FlightGearBridge(host="localhost", port=5501)

    print(f"Swapping in a live round: {PYLON_PROPERTY} = {LIVE_ROUND_OPTION}")
    bridge.set(PYLON_PROPERTY, LIVE_ROUND_OPTION)
    time.sleep(1.0)  # let the pylon listener rebuild the weapon object

    print(f"Selecting {WEAPON_TYPE} via the real fire-control state machine...")
    run_nasal(bridge, f'pylons.fcs.selectWeapon("{WEAPON_TYPE}")')
    time.sleep(0.5)

    bridge.arm_weapons()
    bridge.send_controls(aileron=0.0, elevator=0.0, rudder=0.0, throttle=0.7)

    print("Firing (holding trigger for 1s)...")
    bridge.set("/controls/armament/trigger", "true")
    time.sleep(1.0)
    bridge.set("/controls/armament/trigger", "false")

    time.sleep(2.0)  # let the missile clear the rail and register impact
    added = bridge.get("/ai/models/model-added")
    removed = bridge.get("/ai/models/model-removed")
    impact_name = bridge.get("/ai/models/munition[0]/name")
    print(f"model-added={added!r} model-removed={removed!r} impact name={impact_name!r}")

    bridge.close()


if __name__ == "__main__":
    main()
