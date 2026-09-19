"""Verify the F-15's AFCS heading/altitude hold actually converges, the
same objective-evidence approach used for the c172p KAP140 test in session
2 (visual confirmation alone was previously disputed by the user -- this
reads real telemetry over telnet instead of eyeballing the cockpit).

Launch FlightGear first, in a separate terminal:
    flatpak run --command=fgfs org.flightgear.FlightGear \\
      --fg-aircraft=$HOME/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2024/Aircraft \\
      --aircraft=f15c --airport=KXTA --runway=03L --state=cruise \\
      --altitude=15000 --vc=350 --telnet=5501 --disable-real-weather-fetch \\
      --prop:/sim/terrasync/enabled=true --timeofday=noon

Then: .venv/bin/python scripts/test_f15_autopilot.py
"""

import time

from fly_brain.flightgear_bridge import FlightGearBridge


def main():
    fg = FlightGearBridge()

    start_heading = float(fg.get("/orientation/heading-deg"))
    start_alt = float(fg.get("/position/altitude-ft"))
    target_heading = (start_heading + 60.0) % 360.0
    target_alt = start_alt + 2000.0
    print(f"start heading={start_heading:.1f} alt={start_alt:.0f}ft")
    print(f"target heading={target_heading:.1f} alt={target_alt:.0f}ft")

    # heading-bug-deg via `set` gets typed as a string, which silently
    # breaks the autopilot on this codebase's earlier c172p test -- use
    # run_nasal instead, confirmed correct there.
    fg.run_nasal(f"setprop('/autopilot/settings/heading-bug-deg', {target_heading});")
    fg.run_nasal(f"setprop('/autopilot/settings/target-altitude-ft', {target_alt});")

    # Engage attitude hold (wing-leveler + pitch-hold), then cycle heading
    # mode twice: wing-leveler -> dg-heading-hold -> true-heading-hold.
    fg.set("/sim/model/f15/controls/AFCS/att-hold", "true")
    time.sleep(0.2)
    fg.run_nasal("aircraft.afcs_heading_switch(0);")
    time.sleep(0.2)
    fg.run_nasal("aircraft.afcs_heading_switch(0);")
    fg.set("/sim/model/f15/controls/AFCS/alt-hold", "true")

    print(f"hdg lock = {fg.get('/autopilot/locks/heading')!r}")
    print(f"alt lock = {fg.get('/autopilot/locks/altitude')!r}")

    print(f"{'t':>5} {'heading':>8} {'alt':>8} {'aileron':>8} {'elevator':>8}")
    for i in range(30):
        t = i * 2
        heading = float(fg.get("/orientation/heading-deg"))
        alt = float(fg.get("/position/altitude-ft"))
        aileron = float(fg.get("/controls/flight/aileron"))
        elevator = float(fg.get("/controls/flight/elevator"))
        print(f"{t:5d} {heading:8.1f} {alt:8.0f} {aileron:8.2f} {elevator:8.2f}")
        time.sleep(2.0)

    final_heading = float(fg.get("/orientation/heading-deg"))
    final_alt = float(fg.get("/position/altitude-ft"))
    heading_err = abs((final_heading - target_heading + 180) % 360 - 180)
    alt_err = abs(final_alt - target_alt)
    print(f"\nfinal heading error = {heading_err:.1f} deg, altitude error = {alt_err:.0f} ft")
    print("CONVERGED" if heading_err < 15 and alt_err < 500 else "DID NOT CONVERGE")

    fg.close()


if __name__ == "__main__":
    main()
