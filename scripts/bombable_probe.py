"""Why are the Bombable F-15s flying straight instead of attacking?

Established: addon loads, scenario spawns RED1-3, and bombable/ state IS
attached (damage=0 tracked). But headings drift only 0.005deg/3s -- they are
not manoeuvring. This dumps the AI's real bombable subtree and the addon's
own attack bookkeeping instead of guessing property names.
"""
import sys
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import f15_props as fp
from f15_intercept import Telnet, startup
from fly_brain.geo import haversine_m

MENU = "/bombable/menusettings/"

t = Telnet()
time.sleep(3)
startup(t)

for k, v in (("ai-aircraft-attack-enabled", 1), ("max-ai-attack-player", 0),
             ("bombable-enabled", 1), ("ai-aircraft-skill-combo", 3),
             ("ai-weapon-power-combo", 2)):
    t.set(MENU + k, v)
time.sleep(2)

print("=== menusettings", flush=True)
print(t._send("ls " + MENU), flush=True)
print("=== AI[0] bombable subtree", flush=True)
print(t._send("ls /ai/models/aircraft[0]/bombable"), flush=True)
print("=== AI[0] controls subtree", flush=True)
print(t._send("ls /ai/models/aircraft[0]/controls"), flush=True)
print("=== AI[0] bombable/attributes", flush=True)
print(t._send("ls /ai/models/aircraft[0]/bombable/attributes"), flush=True)

# Is the addon even tracking us as a target?
t.set("/sim/probe/bt", "-")
fp._nasal(t, 'setprop("/sim/probe/bt",""~size(keys(bombable)));')
time.sleep(0.4)
print("bombable symbols:", t.get("/sim/probe/bt"), flush=True)

own = (float(t.get("/position/latitude-deg")),
       float(t.get("/position/longitude-deg")))
for i in range(3):
    n = "/ai/models/aircraft[%d]" % i
    la, lo = t.get(n + "/position/latitude-deg"), t.get(n + "/position/longitude-deg")
    if la is None:
        break
    d = haversine_m(own[0], own[1], float(la), float(lo)) / 1852.0
    print("  %s range=%.2fnm" % (t.get(n + "/callsign"), d), flush=True)

print("=== watching 60s for manoeuvring", flush=True)
for k in range(20):
    row = []
    for i in range(3):
        n = "/ai/models/aircraft[%d]" % i
        cs = t.get(n + "/callsign")
        if cs is None:
            break
        row.append("%s hdg=%.1f roll=%s" % (
            cs, float(t.get(n + "/orientation/true-heading-deg") or 0),
            t.get(n + "/orientation/roll-deg")))
    print("  " + " | ".join(row), flush=True)
    time.sleep(3)
print("PROBE COMPLETE", flush=True)
t.close()
