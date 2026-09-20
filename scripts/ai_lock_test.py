"""Does a REAL AI-managed aircraft give the AIM-9 seeker its lock?

This is the open problem from session 6: the seeker sat at status 0 (SEARCH)
forever because the target was a hand-built /ai/models/aircraft[0] property
node. awg_9.Target.new() caches ~14 child nodes and its accessors throw on a
missing one; the radar/* subtree in particular is something only FlightGear's
AI manager populates.

Getting OUR OWN scenario loaded turned out to be impossible in this flatpak
(see ai_probe.py for the full bisect: only FG_ROOT/AI is scanned, it is
read-only, --data and FG_HOME/AI both fail, and injecting a catalog entry
crashed FlightGear). But aircraft_demo -- a stock scenario in FG_ROOT/AI --
loads fine and spawns a genuine AI-managed aircraft. It sits at KSFO rather
than KXTA, and for THIS question the location is irrelevant: we only need to
know whether a real AI node satisfies the seeker. So start at KSFO and use
the 737 as the target. FG_AIRPORT/FG_RUNWAY in with_sim.sh exist for this.
"""
import sys, time
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from f15_intercept import Telnet, startup
import f15_props as P

t = Telnet()
time.sleep(2)

def aircraft_count():
    n = 0
    while t.get(f"/ai/models/aircraft[{n}]/callsign") is not None:
        n += 1
    return n

print("--- load the stock scenario (real AI manager owns the result)", flush=True)
t.set("/sim/probe/ls", "-")
t.nasal('setprop("/sim/probe/ls",""~fgcommand("load-scenario",'
        'props.Node.new({"name":"aircraft_demo"})));')
time.sleep(5)
print("  rc =", t.get("/sim/probe/ls"), "aircraft =", aircraft_count(), flush=True)
if not aircraft_count():
    print("NO AI AIRCRAFT -- cannot test the seeker"); t.close(); sys.exit(1)

node = "/ai/models/aircraft[0]"
print("--- what the AI manager populates that a hand-built node never had", flush=True)
for p in ("callsign", "position/latitude-deg", "position/longitude-deg",
          "position/altitude-ft", "velocities/true-airspeed-kt",
          "radar/in-range", "radar/range-nm", "radar/bearing-deg",
          "radar/elevation-deg", "radar/h-offset", "radar/v-offset"):
    print(f"  {p} = {t.get(f'{node}/{p}')!r}", flush=True)

print("--- engines + arm", flush=True)
startup(t)
P.arm_aim9(t)

print("--- hand the seeker the REAL node", flush=True)
print("  contact typeof:", P.set_contact(t, node), flush=True)
print("  contact range/bearing:", P.contact_info(t), flush=True)

status, fired = P.try_lock_and_fire(t, node)
print(f"  seeker status={status!r} (1 == MISSILE_LOCK)  fired={fired}", flush=True)
t.close()
sys.exit(0 if fired or status == "1" else 2)
