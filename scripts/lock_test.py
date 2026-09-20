"""Does the AIM-9 seeker lock onto a real, MOVING AI target?

Where this stands, all measured:
  - a real scenario-spawned AI aircraft is now obtainable at runtime via
    f15_props.register_scenario() (the catalog entry needs an `id` child),
    and those DO carry a live radar/* subtree -- which is what a hand-built
    /ai/models node never had and why session 6's seeker never locked.
  - awg_9.Target.new() now returns a hash instead of throwing. That blocker
    is gone.
  - setting armament.contact directly is futile: awg_9.nas's rdr_loop() nils
    it every frame unless the radar is on and displayed.
  - with the radar on and active_u_callsign set, active_u was STILL nil, on a
    verified-level aircraft (alt held 15169->15309ft) with the target at
    1.54nm. So designation itself is failing, and it is not a side effect of
    the jet diverging.
  - airbone is NOT the cause: it needs type "aircraft" AND a `radar` child
    (awg_9.nas:1215 + :1156), and our targets have that subtree.

So this run stops guessing and reads the radar's own internals -- tgts_list
size, active_u_callsign, display-rdr, and the first target's
callsign/airbone/display -- alongside the seeker status. Without those, the
only visible symptom is active_u=nil, which cannot distinguish "scan never
ran" from "scan ran but filtered our target out" from "callsign mismatch".

The target is deliberately boring (straight, constant speed, constant
altitude, no maneuver) and MOVING: fox2.nas refuses to launch at a class-A
target slower than 15kt, so the static scenario could never fire even if the
lock worked.
"""
import sys
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import f15_props as fp
from f15_intercept import Telnet, startup, bearing_deg, angle_diff_deg

SCENARIO = "moving_target_scenario"
PATH = "/home/nikhil/.fgfs/fgdata_extra/AI/" + SCENARIO + ".xml"
CALLSIGN = "BANDIT1"

t = Telnet()
time.sleep(2)

# Engines first: master arm and the weapon selector need the jet powered.
startup(t)

# HAND THE JET BACK TO ITS OWN AUTOPILOT. startup() ends by calling
# connect_stick(), which clears /autopilot/locks/{altitude,heading} so OUR
# stick reaches the FCS -- but this is a setup-only test with no flight loop
# running, so that leaves nobody piloting at all and the jet glides into
# terrain during the ~40s of telnet weapons work (exactly the failure
# connect_stick's own docstring warns about; confirmed live here before this
# was added). Re-engaging the locks costs us nothing because we never send a
# stick input, and a crashing jet also invalidates the radar result: at a wild
# attitude the target falls outside the seeker FOV, so a designation failure
# would be a symptom rather than the bug.
#
# It must be a real HEADING hold, not wing-leveler. wing-leveler keeps the
# wings level but commands no heading, and confirmed live the jet still
# drifted from its 45deg spawn heading to 163deg over ~60s of setup -- by the
# time the radar was read the target was 147deg off the nose at 10.4nm and
# opening, outside awg_9's +-60deg azimuth field, so az_scan forced
# display=0. Every earlier "display=0" result was this, not an RCS, terrain,
# airbone or overflight problem. true-heading-hold rather than
# dg-heading-hold: the DG path is where the F-15's known-unfixed
# "HEADING TRUE SYNC" bug lives (session 3).
alt_now = t.get("/position/altitude-ft")
t.set("/autopilot/settings/target-altitude-ft", alt_now)
t.set("/autopilot/locks/altitude", "altitude-hold")
t.set("/autopilot/locks/heading", "true-heading-hold")
time.sleep(2)
print("  autopilot re-engaged at %sft: alt-lock=%r hdg-lock=%r" % (
    alt_now, t.get("/autopilot/locks/altitude"),
    t.get("/autopilot/locks/heading")))

if not fp.register_scenario(t, SCENARIO, PATH):
    print("FAILED: scenario did not load")
    sys.exit(1)

node = fp.find_ai_target(t, callsign=CALLSIGN)
if node is None:
    print("FAILED: no %s after load" % CALLSIGN)
    sys.exit(1)
print("  target: speed=%s kt alt=%s ft radar/in-range=%s range-nm=%s" % (
    t.get(node + "/velocities/true-airspeed-kt"),
    t.get(node + "/position/altitude-ft"),
    t.get(node + "/radar/in-range"),
    t.get(node + "/radar/range-nm")))

fp.arm_aim9(t)

# POINT AT THE TARGET. The AIM-9 seeker FOV is 27.5deg and awg_9's scan field
# is +-60deg, so the target has to actually be ahead of us -- holding whatever
# heading we happened to drift to is not good enough.
own_lat = float(t.get("/position/latitude-deg"))
own_lon = float(t.get("/position/longitude-deg"))
tgt_lat = float(t.get(node + "/position/latitude-deg"))
tgt_lon = float(t.get(node + "/position/longitude-deg"))
want_hdg = bearing_deg(own_lat, own_lon, tgt_lat, tgt_lon)
t.set("/autopilot/settings/true-heading-deg", round(want_hdg, 1))
print("  turning to bearing %.1f (own %.5f,%.5f -> tgt %.5f,%.5f)" % (
    want_hdg, own_lat, own_lon, tgt_lat, tgt_lon))
for _ in range(20):
    hdg = float(t.get("/orientation/heading-deg"))
    err = abs(angle_diff_deg(want_hdg, hdg))
    print("    hdg=%.1f err=%.1f rng=%s" % (
        hdg, err, t.get(node + "/radar/range-nm")), flush=True)
    if err < 10.0:
        break
    time.sleep(2)

fp.designate(t, CALLSIGN)


def radar_internals():
    """awg_9's own state, so a failure says WHICH stage failed."""
    out = []
    for probe, script in (
        ("tgts", 'setprop("/sim/probe/ri",""~size(awg_9.tgts_list));'),
        ("active_cs", 'setprop("/sim/probe/ri",""~awg_9.active_u_callsign);'),
        ("display_rdr", 'setprop("/sim/probe/ri",""~getprop("%s"));' % fp.RADAR_DISPLAY),
        ("range2", 'setprop("/sim/probe/ri",""~awg_9.range_radar2);'),
        # The awg_9.scan_* counters are USELESS read live: awg_9.nas:601-603
        # zeroes them at the end of every visibility pass, after mirroring
        # them into these properties at :597-599. Read the properties.
        ("p_visible", 'setprop("/sim/probe/ri",""~getprop('
                      '"/instrumentation/radar/scan_tgt_visible"));'),
        ("p_hid_rcs", 'setprop("/sim/probe/ri",""~getprop('
                      '"/instrumentation/radar/scan_tgt_hidden_rcs"));'),
        ("vis0", 'setprop("/sim/probe/ri",size(awg_9.tgts_list)==0?"empty"'
                 ':(""~awg_9.tgts_list[0].get_visible()));'),
        # THE final display gate, awg_9.nas:500 --
        #   if (radar_mode < 2 and deviation > l_az_fld and deviation < r_az_fld)
        #       set_display(get_visible())  else  set_display(0)
        # get_visible() is already 1, so deviation vs the azimuth field is the
        # only thing that can still force display=0. A target we have already
        # overflown sits at ~180deg and is excluded.
        ("dev", 'setprop("/sim/probe/ri",size(awg_9.tgts_list)==0?"empty"'
                ':(""~awg_9.tgts_list[0].deviation));'),
        ("az_fld", 'setprop("/sim/probe/ri",""~awg_9.l_az_fld~"->"~awg_9.r_az_fld);'),
        ("rdr_mode", 'setprop("/sim/probe/ri",""~getprop('
                     '"/instrumentation/radar/radar-mode"));'),
        ("not_acting", 'setprop("/sim/probe/ri",size(awg_9.tgts_list)==0?"empty"'
                       ':(""~awg_9.tgts_list[0].not_acting));'),
        ("first", 'setprop("/sim/probe/ri",size(awg_9.tgts_list)==0?"empty"'
                  ':(""~awg_9.tgts_list[0].Callsign.getValue()~"/airbone="'
                  '~awg_9.tgts_list[0].airbone~"/disp="'
                  '~awg_9.tgts_list[0].get_display()));'),
    ):
        t.set("/sim/probe/ri", "-")
        fp._nasal(t, script)
        time.sleep(0.4)
        out.append("%s=%s" % (probe, t.get("/sim/probe/ri")))
    return " ".join(out)


print("  radar internals: %s" % radar_internals())

# Start the seeker. armament.contact is NOT set here -- the radar owns it.
t.nasal('if(aircraft.Current_srm!=nil){aircraft.Current_srm.status=0;'
        'aircraft.Current_srm.search();}')
time.sleep(1.0)

for i in range(40):
    st = fp.missile_status(t)
    # Keep pointing at it. The target moves, so the bearing that was right
    # before designation drifts out of the seeker's 27.5deg FOV otherwise.
    own_lat = float(t.get("/position/latitude-deg"))
    own_lon = float(t.get("/position/longitude-deg"))
    tgt_lat = float(t.get(node + "/position/latitude-deg"))
    tgt_lon = float(t.get(node + "/position/longitude-deg"))
    t.set("/autopilot/settings/true-heading-deg",
          round(bearing_deg(own_lat, own_lon, tgt_lat, tgt_lon), 1))
    print("  t+%ds status=%r contact=%s hdg=%s tgt_rng_nm=%s" % (
        i, st, fp.contact_info(t), t.get("/orientation/heading-deg"),
        t.get(node + "/radar/range-nm")), flush=True)
    if st == str(fp.MISSILE_LOCK):
        print("LOCK ACQUIRED", flush=True)

        # FIRE. Verify by RESULT, not by the call returning: count live AIM-9s
        # before and after, and look for a missile entity in the AI tree. A
        # release() that silently no-ops (master arm, empty station, outside
        # the dynamic launch zone, target under 15kt) would otherwise look
        # identical to a successful launch.
        def missile_count():
            """Airborne AIM-9 models. These live in /models/model[N], NOT in
            /ai/models: FlightGear's own log shows the release going through
            the MODEL MANAGER ("modelmgr: Adding model
            Aircraft/F-15/Models/Stores/aim-9/aim-9-0.xml"). Counting
            /ai/models/missile[] instead reported 0 through a launch that had
            demonstrably happened ("Launch AIM-9 at BANDIT1" in the log)."""
            n = 0
            found = 0
            while True:
                path = t.get("/models/model[%d]/path" % n)
                if path is None:
                    break
                if "aim-9" in path.lower():
                    found += 1
                n += 1
            return found

        def srm_left():
            """Rounds still on the rails. armament.AIM.active is a VECTOR --
            concatenating it straight into a property yielded nothing at all
            (both reads came back as the placeholder), so take its size."""
            t.set("/sim/probe/nm", "-")
            fp._nasal(t, 'setprop("/sim/probe/nm",""~size(armament.AIM.active));')
            time.sleep(0.4)
            return t.get("/sim/probe/nm")

        before_m, before_s = missile_count(), srm_left()
        print("  firing: srm_status=%s missiles_in_tree=%d rounds_left=%s" % (
            st, before_m, before_s), flush=True)

        t.nasal('aircraft.Current_srm.release();')
        time.sleep(2)

        after_m, after_s = missile_count(), srm_left()
        print("  FIRED: missiles_in_tree %d -> %d  rounds_left %s -> %s" % (
            before_m, after_m, before_s, after_s), flush=True)

        # Watch for the kill: the AI target's own `valid` flag goes false when
        # FlightGear removes it, which is the honest end-to-end confirmation.
        for k in range(25):
            print("    +%ds tgt_rng=%s tgt_valid=%s missiles=%d" % (
                k, t.get(node + "/radar/range-nm"),
                t.get(node + "/valid"), missile_count()), flush=True)
            if t.get(node + "/valid") in (None, "false"):
                print("TARGET DESTROYED", flush=True)
                break
            time.sleep(1)
        break
    time.sleep(1)
else:
    print("NO LOCK. %s" % radar_internals())

t.close()
