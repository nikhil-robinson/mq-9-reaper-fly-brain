"""Phase 0: answer the four unknowns the dogfight design rests on.

No combat code gets written until these are measured. Session 6 burned real
time building on an unverified assumption (FG_HOME/AI being searched, which it
is not), so each of these is a measurement, not a guess:

  1. do two instances see each other as MP contacts?   <- gates everything
  2. which sim/multiplay/generic/int[N] carries the flare counter?
  3. where does a fox2 impact report land -- shooter, victim, or both?
  4. what framerate do we actually get with two instances?

Prints findings as "FINDING: key=value" lines so they can be grepped.
"""
import sys
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import f15_props as fp
from f15_intercept import Telnet, startup

BLUE_TELNET, RED_TELNET = 5501, 5601
BLUE_CS, RED_CS = "BLUE1", "RED1"


def finding(key, value):
    print("FINDING: %s=%s" % (key, value), flush=True)


def mp_contacts(t):
    """Callsigns this instance sees as multiplayer contacts.

    MP contacts live under /ai/models/multiplayer[N], not aircraft[N].
    Telnet.get() returns None for a property that does not exist, which is
    what makes this index walk terminate at all.
    """
    out = []
    i = 0
    while True:
        cs = t.get("/ai/models/multiplayer[%d]/callsign" % i)
        if cs is None:
            return out
        out.append(cs)
        i += 1


blue = Telnet(port=BLUE_TELNET)
red = Telnet(port=RED_TELNET)
time.sleep(2)

# Engines on both, then hand each jet back to its own autopilot. startup()
# ends by clearing the FCS locks so our stick reaches the FCS -- with no
# control loop running that leaves nobody piloting, and the jet glides into
# terrain during the telnet work. Confirmed live in session 7.
for name, t in ((BLUE_CS, blue), (RED_CS, red)):
    print("=== %s startup" % name, flush=True)
    startup(t)
    t.set("/autopilot/settings/target-altitude-ft", t.get("/position/altitude-ft"))
    t.set("/autopilot/settings/true-heading-deg",
          round(float(t.get("/orientation/heading-deg")), 1))
    t.set("/autopilot/locks/altitude", "altitude-hold")
    t.set("/autopilot/locks/heading", "true-heading-hold")

# --- 1. MP contact visibility. Gates the whole design.
seen_blue, seen_red = [], []
for _ in range(15):
    seen_blue, seen_red = mp_contacts(blue), mp_contacts(red)
    if seen_blue and seen_red:
        break
    time.sleep(2)
finding("blue_sees", seen_blue)
finding("red_sees", seen_red)
if not (seen_blue and seen_red):
    print("FATAL: the two instances do not see each other over MP. STOP -- "
          "the design's fallback (kinematic enemy) is a different plan.",
          flush=True)
    blue.close()
    red.close()
    sys.exit(1)

# --- 4. framerate, against fox2.nas:39's ~25fps guidance threshold
for name, t in (("blue", blue), ("red", red)):
    finding("%s_fps" % name, t.get("/sim/frame-rate"))

# --- 2. which generic int carries flares
for name, t in (("blue", blue), ("red", red)):
    finding("%s_generic_ints" % name,
            [t.get("/sim/multiplay/generic/int[%d]" % i) for i in range(20)])
blue.set("/sim/probe/fl", "-")
fp._nasal(blue, 'setprop("/sim/probe/fl",""~armament.flareNode);')
time.sleep(0.4)
finding("blue_flareNode_symbol", blue.get("/sim/probe/fl"))
# fox2 reads the TARGET's node, so find what a Target object resolves it to.
blue.set("/sim/probe/fl2", "-")
fp._nasal(blue, 'setprop("/sim/probe/fl2",armament.contact==nil?"nil-contact"'
                ':(armament.contact.getFlareNode()==nil?"nil-node"'
                ':armament.contact.getFlareNode().getPath()));')
time.sleep(0.4)
finding("blue_target_flare_path", blue.get("/sim/probe/fl2"))

# --- 3. impact report location: fire one AIM-9 at the MP contact
fp.arm_aim9(blue)
blue.set(fp.RADAR_SERVICEABLE, 1)
blue.set(fp.RADAR_STANDBY, 0)
blue.set(fp.RADAR_DISPLAY, "true")
blue.set(fp.RADAR_RANGE, 40)
# Separation FIRST: a designation failure is meaningless if the two jets are
# 40km apart and opening, which is exactly what killed the head-on start.
from fly_brain.geo import bearing_deg, haversine_m, angle_diff_deg
bl = (float(blue.get("/position/latitude-deg")),
      float(blue.get("/position/longitude-deg")))
rl = (float(red.get("/position/latitude-deg")),
      float(red.get("/position/longitude-deg")))
sep_nm = haversine_m(bl[0], bl[1], rl[0], rl[1]) / 1852.0
brg = bearing_deg(bl[0], bl[1], rl[0], rl[1])
err = angle_diff_deg(brg, float(blue.get("/orientation/heading-deg")))
finding("separation_nm", round(sep_nm, 2))
finding("blue_bearing_err_to_red", round(err, 1))

fp.designate(blue, RED_CS)
time.sleep(2)
finding("blue_contact_info", fp.contact_info(blue))


def radar_internals(t):
    """awg_9's own state, so a failed designation says WHICH stage failed
    rather than only that active_u was nil."""
    out = []
    for key, script in (
        ("tgts", 'setprop("/sim/probe/ri",""~size(awg_9.tgts_list));'),
        ("active_cs", 'setprop("/sim/probe/ri",""~awg_9.active_u_callsign);'),
        ("first_cs", 'setprop("/sim/probe/ri",size(awg_9.tgts_list)==0?"empty"'
                     ':(""~awg_9.tgts_list[0].Callsign.getValue()));'),
        ("first_type", 'setprop("/sim/probe/ri",size(awg_9.tgts_list)==0?"empty"'
                       ':(""~awg_9.tgts_list[0].type));'),
        ("airbone", 'setprop("/sim/probe/ri",size(awg_9.tgts_list)==0?"empty"'
                    ':(""~awg_9.tgts_list[0].airbone));'),
        ("disp", 'setprop("/sim/probe/ri",size(awg_9.tgts_list)==0?"empty"'
                 ':(""~awg_9.tgts_list[0].get_display()));'),
        ("dev", 'setprop("/sim/probe/ri",size(awg_9.tgts_list)==0?"empty"'
                ':(""~awg_9.tgts_list[0].deviation));'),
    ):
        t.set("/sim/probe/ri", "-")
        fp._nasal(t, script)
        time.sleep(0.4)
        out.append("%s=%s" % (key, t.get("/sim/probe/ri")))
    return " ".join(out)


finding("blue_radar_internals", radar_internals(blue))

# Re-read the target flare path now that a contact genuinely exists.
blue.set("/sim/probe/fl2", "-")
fp._nasal(blue, 'setprop("/sim/probe/fl2",armament.contact==nil?"nil-contact"'
                ':(armament.contact.getFlareNode()==nil?"nil-node"'
                ':armament.contact.getFlareNode().getPath()));')
time.sleep(0.4)
finding("blue_target_flare_path_locked", blue.get("/sim/probe/fl2"))

blue.nasal('if(aircraft.Current_srm!=nil){aircraft.Current_srm.status=0;'
           'aircraft.Current_srm.search();}')
time.sleep(2)
status = fp.missile_status(blue)
finding("blue_seeker_status", status)
if status == str(fp.MISSILE_LOCK):
    blue.nasal('aircraft.Current_srm.release();')
    finding("blue_fired", "yes")
    for _ in range(20):
        for who, t in (("blue", blue), ("red", red)):
            listing = t._send("ls /ai/models")
            hits = [l.strip() for l in listing.splitlines() if "impact" in l]
            if any("'" in h and "''" not in h for h in hits):
                finding("%s_impact_report" % who, hits)
        time.sleep(1)
else:
    finding("blue_fired", "no -- never locked")

blue.close()
red.close()
print("PROBE COMPLETE", flush=True)
