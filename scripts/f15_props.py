"""FlightGear-side setup for the obstacle run: place obstacle models, put a
target in the sky, arm the jet, and fire.

All of this goes over telnet, NOT the native binary protocol. That split is
deliberate: telnet is far too slow for the flight control loop (an earlier
session's telnet-driven hot loop rolled inverted and hit terrain within ~15
steps), but these are one-off discrete actions where latency is irrelevant.

TELNET NASAL HAS ONE HARD RULE -- NO SPACES.
FlightGear's telnet `run nasal script=<...>` is space-delimited: everything
after the first space is silently discarded, the script never runs, and you
get no error at all. Confirmed live and precisely:

    setprop("/sim/probe/a", "spaced");   -> property unchanged, silent no-op
    setprop("/sim/probe/b","nospace");   -> works

This is why several earlier attempts "did nothing" with no diagnostic. Every
script here is therefore written space-free, which also rules out any keyword
needing a trailing space (`var x=`, `if (...)`), so this module avoids them
entirely. _nasal() asserts the rule rather than trusting us to remember it.
"""

import time

# Each gate is TWO of these tower objects placed independently apart (see
# f15_course.py for the full history of why: the real Red Bull gate mesh
# bakes both pylons into one fixed-offset object, which broke down under
# both non-uniform scaling -- rendered as a horizontal streak -- and under
# uniform scaling -- no independent control of pylon spacing vs height).
#
# tower_big.xml wraps Models/Power/coolingtower.ac (a real FG_ROOT model,
# 2 named parts sharing one origin) with a uniform 4x scale and an explicit
# `range` animation. The range animation is required: confirmed live the
# tower only rendered once the aircraft was already close to it, a known
# OpenSceneGraph issue where a runtime scale transform's visibility-culling
# bounding sphere is computed from the object's ORIGINAL unscaled size. The
# range animation overrides visibility purely by distance to the viewer,
# forcing the tower to render out to 50km regardless of that stale bound.
#
# The wrapper lives in .fgfs/Models/f15course/, NOT inside FG_ROOT (which is
# read-only) and NOT inside FG_HOME's TerraSync tree (which actively prunes
# files it does not recognize -- confirmed live an earlier wrapper placed
# there was deleted from disk before FlightGear could load it).
_TOWER_DIR = "/home/nikhil/.fgfs/Models/f15course"
OBSTACLE_MODELS = {
    "tower": f"{_TOWER_DIR}/tower_big.xml",
}


def query_elevation_m(t, lat, lon):
    """Real ground elevation at a point, in METRES, via geo.elevation().

    Confirmed live at KXTA: aircraft position and a point 2km north both
    returned real terrain (~1343m / ~4406ft, matching the field elevation),
    not nil -- so scenery is loaded there and this is safe to call for gate
    placement on the actual course. Returns None if that tile genuinely has
    not streamed in (geo.elevation returns nil in that case); callers must
    handle that rather than silently placing a gate at 0m/underground.
    """
    t.set("/sim/probe/elev", "-")
    _nasal(t, f'setprop("/sim/probe/elev",geo.elevation({lat:.6f},{lon:.6f})'
              f'==nil?"nil":""~geo.elevation({lat:.6f},{lon:.6f}));')
    time.sleep(0.3)
    v = t.get("/sim/probe/elev")
    if v in (None, "nil", "-"):
        return None
    return float(v)


MISSILE_LOCK = 1  # fox2.nas status constant
MASTER_ARM = "/sim/model/f15/controls/armament/master-arm-switch"
WEAPON_SELECTOR = "/sim/model/f15/controls/armament/weapon-selector"
SELECTOR_SRM = 1  # 1 = short range missile (AIM-9), per weapons.nas


def _nasal(t, script):
    """Run a space-free Nasal snippet. See the module docstring."""
    assert " " not in script, f"telnet nasal must be space-free: {script!r}"
    t.nasal(script)


# FlightGear culls runtime-placed static models by these global LOD ranges
# BEFORE any per-model range animation gets a say, so tower_big.xml's own
# 50km range animation was not enough on its own -- the towers still only
# appeared once the jet was nearly on top of them.
#
# The names matter: since FlightGear 2018.3 these are a base distance plus
# two DELTAS (defaults.xml: detailed 1500, rough-delta 7500, bare-delta
# 21000, i.e. a static model vanishes entirely at 30km). There is no
# /sim/rendering/static-lod/rough or .../bare -- writing those just creates
# dead properties that nothing reads, which is exactly what an earlier
# version of this did. Totals below put the vanish point at 60km.
STATIC_LOD_M = {
    "/sim/rendering/static-lod/detailed": 30000,
    "/sim/rendering/static-lod/rough-delta": 15000,
    "/sim/rendering/static-lod/bare-delta": 15000,
}


def widen_static_lod(t):
    """Stop FlightGear culling the towers until the jet is close to them.

    Verifies by reading the properties back: these are typed <double> nodes
    and a silently-ignored write here is invisible in the sim until someone
    notices the gates are missing.
    """
    for prop, m in STATIC_LOD_M.items():
        t.set(prop, m)
    got = {p: t.get(p) for p in STATIC_LOD_M}
    total = sum(float(v) for v in got.values() if v is not None)
    print(f"  Static model LOD: {got} -> models drawn out to {total/1000:.0f}km")
    return total


def place_obstacles(t, obstacles, hold=None):
    """Place a visible 3D model at each obstacle position.

    geo.put_model's elevation argument is in METRES. Passing feet (or calling
    the raw add-model fgcommand with an "elevation-ft" key) took FlightGear
    down outright mid-placement -- telnet went to a broken pipe.

    `hold`, if given, is called between placements to keep flying the
    aircraft. This is a telnet-only loop -- if the caller is also flying the
    aircraft over the SEPARATE native-protocol connection, leaving that
    connection silent for the whole loop lets FlightGear keep applying
    stale, last-commanded control surfaces the entire time (confirmed live
    this let a jet drift from 496kt to 636kt and its heading flip -177deg
    before control resumed). It takes a callable rather than the jet so the
    caller decides what "keep flying" means -- holding ALTITUDE, not just
    pitch, which is the distinction that put an earlier version into the
    ground during this loop.
    """
    widen_static_lod(t)
    for o in obstacles:
        if hold is not None:
            hold()
        path = OBSTACLE_MODELS.get(o.kind, OBSTACLE_MODELS["tower"])
        _nasal(t, f'geo.put_model("{path}",{o.lat:.8f},{o.lon:.8f},'
                  f'{o.base_ft * 0.3048:.2f},0);')
        time.sleep(0.05)
    time.sleep(0.5)
    # Verify by RESULT rather than by predicting whether the path resolves:
    # add-model happily creates a node for a file that does not exist, so the
    # only honest check is whether the tree actually came back populated.
    first = t.get("/models/model[0]/path")
    print(f"  Placed {len(obstacles)} obstacle models (model[0]={first!r})")
    return bool(first)


def find_ai_target(t, callsign="BANDIT1", tries=10):
    """Locate the REAL AI aircraft FlightGear spawned from the scenario.

    A hand-built /ai/models node (the old spawn_target) never produced a
    seeker lock -- awg_9.Target.new(node) caches ~14 child nodes on
    construction and a synthetic node was always missing one, so its
    accessors threw and the contact never validated. A scenario-spawned
    aircraft is a REAL AI model: FlightGear itself maintains every property
    Target.new() expects, so this just has to find which index it landed at
    (normally aircraft[0], but don't assume it if something else is already
    in /ai/models).
    """
    for _ in range(tries):
        i = 0
        while True:
            node = f"/ai/models/aircraft[{i}]"
            cs = t.get(f"{node}/callsign")
            if cs is None:
                break
            if cs == callsign:
                print(f"  Found {callsign} at {node}")
                return node
            i += 1
        time.sleep(1.0)
    return None


# Stations that actually accept an AIM-9, read off the <payload> block in
# f15-common.xml rather than assumed. The jet spawns with every station set
# to "none": confirmed live, armament.AIM.active was empty and
# get_sel_missile_count() returned 0, so there was simply nothing to fire and
# the seeker could never report a lock no matter what was commanded.
AIM9_STATIONS = (0, 2, 8, 10)


def load_aim9(t):
    """Mount AIM-9s. ext_stores.nas listens on payload/weight[N]/selected."""
    for n in AIM9_STATIONS:
        t.set(f"/payload/weight[{n}]/selected", "AIM-9")
        time.sleep(0.25)
    time.sleep(1.0)
    loaded = [t.get(f"/payload/weight[{n}]/selected") for n in AIM9_STATIONS]
    print(f"  Stations {list(AIM9_STATIONS)} -> {loaded}")


def arm_aim9(t):
    """Master arm on, AIM-9 selected. Paths verified in weapons.nas."""
    load_aim9(t)
    t.set(MASTER_ARM, 1)
    time.sleep(0.4)
    t.set(WEAPON_SELECTOR, SELECTOR_SRM)
    time.sleep(0.8)
    print(f"  Master arm={t.get(MASTER_ARM)} selector={t.get(WEAPON_SELECTOR)}")


def missile_status(t):
    """Read the selected missile's seeker status out of Nasal into a property.

    The missile object lives only in Nasal (weapons.Current_srm) -- there is
    no property mirroring its status, so it has to be poked across.
    """
    t.set("/sim/probe/msl", "-")
    _nasal(t, 'setprop("/sim/probe/msl",aircraft.Current_srm==nil?"none"'
              ':""~aircraft.Current_srm.status);')
    time.sleep(0.3)
    return t.get("/sim/probe/msl")


def set_contact(t, node="/ai/models/aircraft[0]"):
    """Hand the seeker a target object built from the real AI node.

    fox2.nas reads the global `armament.contact`, and awg_9.nas builds those
    Target objects from /ai/models nodes (awg_9.Target.new(node)). Relying on
    the radar sweep to populate it did not work for a hand-made AI node --
    confirmed live, armament.contact stayed nil and the seeker sat at status
    0 (SEARCH) forever no matter how long it ran. Constructing the contact
    directly is the same object the radar would have produced, just without
    waiting for a scan -- and now that the node is a REAL scenario-spawned
    aircraft rather than a synthetic one, Target.new()'s cached accessors
    have everything they expect.
    """
    path = node.lstrip("/")  # getNode() wants a relative-to-globals path, no leading /
    t.set("/sim/probe/ct", "-")
    _nasal(t, f'armament.contact=awg_9.Target.new('
              f'props.globals.getNode("{path}"));'
              f'setprop("/sim/probe/ct",typeof(armament.contact));')
    time.sleep(0.5)
    return t.get("/sim/probe/ct")


def contact_info(t):
    """Range/bearing the seeker itself sees, for diagnosing a failed lock."""
    t.set("/sim/probe/ci", "-")
    _nasal(t, 'setprop("/sim/probe/ci",armament.contact==nil?"nil":'
              '(""~armament.contact.get_range()~"nm/"'
              '~armament.contact.get_bearing()~"deg"));')
    time.sleep(0.4)
    return t.get("/sim/probe/ci")


def try_lock_and_fire(t, node="/ai/models/aircraft[0]"):
    """Start the seeker, and release if it reports a lock.

    Returns (status_before, fired). Preconditions that silently block a
    launch, per weapons.nas/fox2.nas: master arm off, empty station, no lock,
    target outside the seeker FOV or the dynamic launch zone, or a class-A
    target slower than 15kt.
    """
    set_contact(t, node)
    _nasal(t, 'if(aircraft.Current_srm!=nil){aircraft.Current_srm.status=0;'
              'aircraft.Current_srm.search();}')
    time.sleep(1.0)
    st = missile_status(t)
    fired = False
    if st == str(MISSILE_LOCK):
        _nasal(t, 'aircraft.Current_srm.release();')
        time.sleep(0.5)
        fired = True
    return st, fired
