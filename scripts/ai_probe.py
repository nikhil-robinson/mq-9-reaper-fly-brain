"""Get a REAL AI target aircraft in, without a writable FG_ROOT.

Findings that led here, all measured rather than assumed:
  - load-scenario never hits the filesystem. It resolves a name against the
    catalog at /sim/ai/scenarios, which FGAIManager builds ONCE at startup by
    scanning FG_ROOT/AI only -- every one of the 30 entries had
    path=/app/share/flightgear/AI/*.xml. Byte-identical known-good content
    gave rc=1 from FG_ROOT/AI but rc=0 from FG_HOME/AI and from a --data dir,
    so the file's location was never the problem: the catalog lookup was.
    FG_ROOT is read-only in this flatpak.
  - a catalog entry's real children are `id` (the filename stem, e.g.
    'aircraft_demo'), `path`, `name` (the human title) and `description` --
    dumped from scenario[4], not guessed. An injected entry carrying only
    {name, path} gave rc=0: it had no `id` to match on.

Two ways to exploit the fact that each entry carries its own path:
  A. append our own entry with id + name + path.
  B. if A fails, repoint an EXISTING entry's path at our XML and load its
     already-resolvable id. That is immune to any name-vs-id lookup question,
     because we only change what an already-working name resolves TO.

Uses telnet `set`, NOT Nasal: `run nasal script=` is space-delimited and
silently truncates at the first space, so a multi-line Nasal block would be
discarded with no error.
"""
import sys
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from f15_intercept import Telnet

SCENARIO = "static_targets_scenario"
PATH = "/home/nikhil/.fgfs/fgdata_extra/AI/" + SCENARIO + ".xml"

t = Telnet()
time.sleep(2)


def aircraft_count():
    n = 0
    while t.get("/ai/models/aircraft[%d]/callsign" % n) is not None:
        n += 1
    return n


def load(name):
    t.set("/sim/probe/ls", "-")
    t.nasal('setprop("/sim/probe/ls",""~fgcommand("load-scenario",'
            'props.Node.new({"name":"%s"})));' % name)
    time.sleep(4)
    return t.get("/sim/probe/ls")


def report(tag):
    n = aircraft_count()
    print("  [%s] aircraft=%d" % (tag, n), flush=True)
    for k in range(n):
        node = "/ai/models/aircraft[%d]" % k
        vals = {}
        for p in ("callsign", "position/latitude-deg", "position/longitude-deg",
                  "position/altitude-ft", "velocities/true-airspeed-kt",
                  # the radar subtree is exactly what the hand-built node lacked
                  "radar/in-range", "radar/range-nm", "radar/bearing-deg"):
            vals[p] = t.get(node + "/" + p)
        print("    aircraft[%d]: %s" % (k, vals), flush=True)
    return n


print("aircraft before: %d" % aircraft_count(), flush=True)

# --- A: append our own catalog entry, this time WITH an id
i = 0
while t.get("/sim/ai/scenarios/scenario[%d]/path" % i) is not None:
    i += 1
print("A: injecting scenario[%d] id=%s path=%s" % (i, SCENARIO, PATH), flush=True)
for key, val in (("id", SCENARIO), ("name", SCENARIO), ("path", PATH)):
    t.set("/sim/ai/scenarios/scenario[%d]/%s" % (i, key), val)
print("A: rc=%r" % load(SCENARIO), flush=True)
if report("A"):
    t.close()
    sys.exit(0)

# --- B: repoint an existing, already-resolvable entry at our file
print("B: repointing scenario[4] (aircraft_demo) path -> %s" % PATH, flush=True)
t.set("/sim/ai/scenarios/scenario[4]/path", PATH)
print("B: readback path = %r" % t.get("/sim/ai/scenarios/scenario[4]/path"), flush=True)
print("B: rc=%r" % load("aircraft_demo"), flush=True)
# TGT01/02/03 callsigns prove OUR file loaded. 'United 720' means we got the
# stock 737 instead and the repoint did not take effect.
if report("B"):
    t.close()
    sys.exit(0)

print("NEITHER A NOR B PRODUCED AN AI AIRCRAFT", flush=True)
t.close()
sys.exit(1)
