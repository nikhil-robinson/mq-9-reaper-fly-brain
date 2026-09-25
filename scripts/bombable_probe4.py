"""Does attack-initialized ever flip to 1 on its own, and after how long?

Times from telnet-ready (proxy for scenario load, since AI spawns at scenario
load time which precedes telnet readiness) rather than from our own script's
setup, since attack_init's 55+rand() second delay starts at AI model LOAD.
"""
import sys
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from f15_intercept import Telnet, startup

MENU = "/bombable/menusettings/"
t0 = time.monotonic()
t = Telnet()
time.sleep(2)
startup(t)
t.set("/bombable/player-faction", "A")
for k, v in (("bombable-enabled", 1), ("friendly-fire-allowed", 1),
             ("ai-aircraft-attack-enabled", 1), ("ai-aircraft-weapons-enabled", 1),
             ("max-ai-attack-player", 0), ("ai-aircraft-skill-combo", 3)):
    t.set(MENU + k, v)

node = "/ai/models/aircraft[1]"  # RED2
for _ in range(30):
    elapsed = time.monotonic() - t0
    inited = t.get(node + "/bombable/initializers/attack-initialized")
    hdg = t.get(node + "/orientation/true-heading-deg")
    roll = t.get(node + "/orientation/roll-deg")
    print("t+%.0fs inited=%s hdg=%s roll=%s" % (elapsed, inited, hdg, roll), flush=True)
    time.sleep(5)
print("PROBE4 COMPLETE", flush=True)
t.close()
