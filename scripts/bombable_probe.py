"""Is Bombable's attack_loop actually running for our AI F-15s?

Faction is NOT the blocker: nearestEnemyAI() defaults playerIsEnemy=1 ("attack
unless proven friendly") and skips the friendly test entirely when
friendly-fire-allowed is set. So if the AI still cruises, attack_loop never
started. That loop writes bombable/targeting-player and
bombable/ai-attack-target, so their presence is the decisive test.
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
time.sleep(2)

n = "/ai/models/aircraft[0]"
print("=== bombable subtree", flush=True)
print(t._send("ls " + n + "/bombable"), flush=True)

for label, script in (
    ("in_attributes", 'setprop("/sim/probe/x",contains(bombable.attributes,'
                      '"/ai/models/aircraft[0]")?"yes":"no");'),
    ("attr_keys", 'setprop("/sim/probe/x",""~size(keys(bombable.attributes)));'),
    ("has_attack_fn", 'setprop("/sim/probe/x",contains(bombable,"attack_loop")'
                      '?"yes":"no");'),
    ("faction", 'setprop("/sim/probe/x",contains(bombable.attributes,'
                '"/ai/models/aircraft[0]")and contains('
                'bombable.attributes["/ai/models/aircraft[0]"],"faction")?'
                'bombable.attributes["/ai/models/aircraft[0]"].faction:"none");'),
):
    t.set("/sim/probe/x", "-")
    fp._nasal(t, script)
    time.sleep(0.5)
    print("  %s=%s" % (label, t.get("/sim/probe/x")), flush=True)

print("=== forcing attack_init", flush=True)
t.set("/sim/probe/y", "-")
fp._nasal(t, 'bombable.attack_init("/ai/models/aircraft[0]");'
             'setprop("/sim/probe/y","called");')
time.sleep(3)
print("  attack_init=%s" % t.get("/sim/probe/y"), flush=True)
print(t._send("ls " + n + "/bombable"), flush=True)

for _ in range(10):
    print("  hdg=%s roll=%s targeting_player=%s attack_target=%s" % (
        t.get(n + "/orientation/true-heading-deg"),
        t.get(n + "/orientation/roll-deg"),
        t.get(n + "/bombable/targeting-player"),
        t.get(n + "/bombable/ai-attack-target")), flush=True)
    time.sleep(3)
print("PROBE COMPLETE", flush=True)
t.close()
