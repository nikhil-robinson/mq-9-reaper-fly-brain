"""RED2/RED3 ARE in bombable.attributes -- so why don't they maneuver?

Force attack_loop directly and watch heading/roll for real movement, rather
than relying on the game's own scheduler (which may not be ticking these
nodes for a reason still unknown).
"""
import sys
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import f15_props as fp
from f15_intercept import Telnet, startup

MENU = "/bombable/menusettings/"
t = Telnet()
time.sleep(3)
startup(t)
t.set("/bombable/player-faction", "A")
for k, v in (("bombable-enabled", 1), ("friendly-fire-allowed", 1),
             ("ai-aircraft-attack-enabled", 1), ("ai-aircraft-weapons-enabled", 1),
             ("max-ai-attack-player", 0), ("ai-aircraft-skill-combo", 3)):
    t.set(MENU + k, v)
time.sleep(3)

node = "/ai/models/aircraft[1]"  # RED2, confirmed in attributes

# Read the attack-related attributes actually stored for this node.
t.set("/sim/probe/a1", "-")
fp._nasal(t, 'setprop("/sim/probe/a1",contains(bombable.attributes["%s"],"attacks")?"yes":"no");' % node)
time.sleep(0.4)
print("has_attacks_attr=%s" % t.get("/sim/probe/a1"), flush=True)

t.set("/sim/probe/a2", "-")
fp._nasal(t, 'setprop("/sim/probe/a2",""~bombable.attributes["%s"].faction);' % node)
time.sleep(0.4)
print("faction=%s" % t.get("/sim/probe/a2"), flush=True)

print("=== before force: baseline heading over 12s", flush=True)
for _ in range(4):
    print("  hdg=%s roll=%s" % (t.get(node + "/orientation/true-heading-deg"),
                                t.get(node + "/orientation/roll-deg")), flush=True)
    time.sleep(3)

print("=== calling attack_loop directly", flush=True)
t.set("/sim/probe/al", "-")
fp._nasal(t, 'bombable.attack_loop(0,"%s",0.5);setprop("/sim/probe/al","called");' % node)
time.sleep(1)
print("attack_loop_call=%s" % t.get("/sim/probe/al"), flush=True)

for _ in range(8):
    print("  hdg=%s roll=%s targeting=%s" % (
        t.get(node + "/orientation/true-heading-deg"),
        t.get(node + "/orientation/roll-deg"),
        t.get(node + "/bombable/targeting-player")), flush=True)
    time.sleep(3)
print("PROBE3 COMPLETE", flush=True)
t.close()
