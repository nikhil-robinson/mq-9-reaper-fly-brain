"""What IS in bombable.attributes, since attr_keys=1 but not our AI node?"""
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

t.set("/sim/probe/x", "-")
fp._nasal(t, 'setprop("/sim/probe/x",""~keys(bombable.attributes)[0]);')
time.sleep(0.5)
print("first_key=%s" % t.get("/sim/probe/x"), flush=True)

# Check whether the player node itself got attribute-initialized (i.e. was
# object_init ever called on anything?)
t.set("/sim/probe/x2", "-")
fp._nasal(t, 'setprop("/sim/probe/x2",contains(bombable.attributes,"/")?"yes":"no");')
time.sleep(0.5)
print("root_present=%s" % t.get("/sim/probe/x2"), flush=True)

# List all AI aircraft node paths that actually exist
i = 0
while True:
    cs = t.get("/ai/models/aircraft[%d]/callsign" % i)
    if cs is None:
        break
    print("aircraft[%d] callsign=%s" % (i, cs), flush=True)
    i += 1

# Try each aircraft index against attributes
for j in range(i):
    node = "/ai/models/aircraft[%d]" % j
    t.set("/sim/probe/x3", "-")
    fp._nasal(t, 'setprop("/sim/probe/x3",contains(bombable.attributes,"%s")?"yes":"no");' % node)
    time.sleep(0.4)
    print("%s in attributes=%s" % (node, t.get("/sim/probe/x3")), flush=True)

print("PROBE2 COMPLETE", flush=True)
t.close()
