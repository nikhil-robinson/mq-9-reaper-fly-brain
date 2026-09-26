"""F-15 dogfight against Bombable AI, in a SINGLE FlightGear instance.

Supersedes the two-instance multiplayer arena (scripts/dogfight.py). That
approach worked -- Phase 0 confirmed both instances saw each other as MP
contacts at 31-57fps -- but it cost two FlightGear processes on a 7.3GB box
already into swap, and FlightGear still has no damage model, so a "kill" had
to be inferred from separation rather than actually happening.

Bombable supplies, in one instance, the three things that were hard:
  - AI aircraft that manoeuvre onto the player and shoot
  - a real damage model
  - explosions and destruction

Our jet is still flown by CombatPilot through the calibrated decode layer
(g-demand + roll-rate demand), so the connectome remains a later policy swap
rather than a rewrite.

Run via:  AI_SCENARIO=bombable_dogfight scripts/with_sim.sh scripts/dogfight_bombable.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import f15_props as fp
from f15_intercept import Telnet, connect_stick, g_to_elevator, startup
from fly_brain.combat import (CAS_ROLL_RATE_FULL_DPS, CORNER_SPEED_KT,
                              CombatPilot)
from fly_brain.geo import angle_diff_deg, bearing_deg, haversine_m
from fly_brain.native_protocol import NativeBridge

STEP_HZ = 20
SPEED_UP = int(os.environ.get("SPEED_UP", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "3000"))
M_TO_NM = 1.0 / 1852.0
MENU = "/bombable/menusettings/"
# FCS_DIAG=1: read back the real JSBSim FCS/CAS chain alongside our own
# g_cmd, on its own telnet connection at 5Hz, entirely separate from the
# control loop. This does NOT change any control-law code -- it exists to
# answer, with real telemetry instead of another guess, whether g_cmd is
# even reaching the airframe as a matching real load factor. See the
# decision tree: g_cmd vs stick-commanded-g (is our G_TABLE wrong?),
# stick-commanded-g vs Nz (is the CAS achieving the demand?), ptc-saturated
# / pitch-input-sum pinned (is nose-down authority clipped?).
FCS_DIAG = os.environ.get("FCS_DIAG", "0") == "1"
FCS_DIAG_PROPS = [
    "/fdm/jsbsim/fcs/elevator-cmd-norm",
    "/fdm/jsbsim/fcs/pitch-trim-cmd-norm",
    "/fdm/jsbsim/fcs/stick-pitch",
    "/fdm/jsbsim/fcs/stick-pitch-scheduled",
    "/fdm/jsbsim/fcs/stick-commanded-g",
    "/fdm/jsbsim/accelerations/Nz",
    "/fdm/jsbsim/cas/ptc-deviation-g",
    "/fdm/jsbsim/cas/ptc-saturated",
    "/fdm/jsbsim/cas/ptc-feedback-g",
    "/fdm/jsbsim/fcs/pitch-input-sum",
    "/fdm/jsbsim/fcs/pitch-input-deg",
    "/fdm/jsbsim/fcs/elevator-pos-deg",
    "/fdm/jsbsim/aero/alpha-deg",
    "/fdm/jsbsim/flight-path/gamma-deg",
    # Roll chain: bank collapses mid-turn (-36 -> -8deg) while the controller
    # commands full opposite roll, and no amount of reasoning from the outside
    # has explained it. Read the same chain the pitch axis exposed.
    "/controls/flight/aileron",
    "/fdm/jsbsim/fcs/aileron-cmd-norm",
    "/fdm/jsbsim/fcs/stick-roll",
    "/fdm/jsbsim/cas/roll-output",
    "/fdm/jsbsim/velocities/p-rad_sec",
    "/orientation/roll-deg",
]
# TELNET MUST NOT TOUCH THE HOT LOOP. native_protocol.py's docstring says it
# outright ("an earlier session's telnet-driven hot loop rolled inverted and
# hit terrain within ~15 steps") and this script proved it again: reading three
# enemies costs ~21 telnet round trips, which dragged the control loop from
# 20Hz down to roughly 1Hz. At 1Hz the jet is effectively unflown between
# commands and spirals into the ground regardless of how good the control law
# is -- three crashes were blamed on the control law before the rate was
# measured. Enemy state is therefore polled by a BACKGROUND THREAD and the
# control loop only ever reads the latest snapshot.
ENEMY_POLL_SEC = 0.5
# How many consecutive empty /ai/models indices to tolerate before concluding
# the list has ended, and a hard stop -- see read_enemies.
AI_SCAN_TOLERANCE = 3
AI_SCAN_MAX = 12
# 0.35, not 0.1: missile_status itself costs ~0.3s (it has to let the Nasal
# run), so polling every 0.1s asked for the connection more often than the
# probe could complete and monopolised it. The 1.5s lock latch below is what
# actually covers the gaps between reads.
LOCK_POLL_SEC = 0.35  # seeker status only -- cheap, and shots are fleeting
# How long a seen lock is trusted for. Covers the poll interval and the
# seeker's own flicker -- see poll_lock; fire_srm is the real gate.
LOCK_LATCH_SEC = 1.5
# Give up on a committed target only if it runs this far away -- well beyond
# the AIM-9's 9nm seeker limit, so we never drop one we could still shoot.
ABANDON_RANGE_NM = 12.0


def enable_bombable(t):
    """Turn the fight on. These are the in-game Bombable menu items, which are
    plain properties, so they can be set over telnet like anything else.

    FACTIONS are what actually make the AI hostile. Bombable's README is
    explicit: "Old AI models, without added faction will not attack you, until
    friendly fire is allowed." Confirmed live -- with no faction on either
    side the three F-15s simply cruised, headings changing 0.01deg over 9
    seconds, because Bombable had no reason to consider us an enemy. The AI
    models are tagged faction "B" in F-15-bombableinclude.xml; the player is
    "A". friendly-fire-allowed is set as well so a factionless model still
    engages rather than silently ignoring us.
    """
    t.set("/bombable/player-faction", "A")
    for key, val in (("bombable-enabled", 1),
                     ("friendly-fire-allowed", 1),
                     ("ai-aircraft-attack-enabled", 1),
                     ("ai-aircraft-weapons-enabled", 1),
                     ("max-ai-attack-player", 0),   # 0 = all of them
                     ("ai-aircraft-skill-combo", 3),
                     ("ai-weapon-power-combo", 2)):
        t.set(MENU + key, val)
    time.sleep(1.0)
    print("  bombable: attack=%s weapons=%s max_attackers=%s skill=%s "
          "friendly_fire=%s player_faction=%s" % (
              t.get(MENU + "ai-aircraft-attack-enabled"),
              t.get(MENU + "ai-aircraft-weapons-enabled"),
              t.get(MENU + "max-ai-attack-player"),
              t.get(MENU + "ai-aircraft-skill-level"),
              t.get(MENU + "friendly-fire-allowed"),
              t.get("/bombable/player-faction")), flush=True)


def read_enemies(t):
    """Live enemy state straight from the AI tree.

    Also reads Bombable's own per-AI weapons-firing flag -- this is the real
    "they are shooting at me" signal (set true while the AI's gun_loop is
    actively firing), which lets EVADE trigger from a real threat instead of
    only reacting after our own damage has already gone up.
    """
    # DO NOT BAIL ON THE FIRST MISSING INDEX.
    #
    # A single None used to end the scan, so one transient read returned an
    # EMPTY list -- and the caller's `if got:` then silently skipped the whole
    # cycle including the radar designation. With three threads sharing one
    # telnet connection that happened often enough that designate() was never
    # called in an entire run (0 designations, 0 locks, 0 shots, and no errors
    # because nothing raised). Verified on an idle connection that the scan
    # itself is fine: it returns RED1, RED2, RED3.
    #
    # Scan a few indices past a gap before giving up, so a dropped read costs
    # one aircraft for one cycle instead of the entire list.
    out = []
    i = 0
    misses = 0
    while misses < AI_SCAN_TOLERANCE and i < AI_SCAN_MAX:
        cs = t.get("/ai/models/aircraft[%d]/callsign" % i)
        if cs is None:
            misses += 1
            i += 1
            continue
        n = "/ai/models/aircraft[%d]" % i
        lat = t.get(n + "/position/latitude-deg")
        lon = t.get(n + "/position/longitude-deg")
        if lat is not None and lon is not None:
            damage = t.get(n + "/bombable/attributes/damage")
            out.append(dict(
                callsign=cs, node=n,
                lat=float(lat), lon=float(lon),
                alt_ft=float(t.get(n + "/position/altitude-ft") or 0),
                heading=float(t.get(n + "/orientation/true-heading-deg") or 0),
                speed_kt=float(t.get(n + "/velocities/true-airspeed-kt") or 0),
                damage=float(damage) if damage not in (None, "") else 0.0,
                firing=(t.get(n + "/bombable/weapons-firing") in ("1", "true", 1, True)
                       or t.get(n + "/bombable/fire-particles/ai-weapon-firing")
                          in ("1", "true", 1, True)),
                alive=(damage is None or float(damage) < 100.0),
            ))
        i += 1
    return out


def main():
    t = Telnet()
    time.sleep(2)
    # Keep the autopilot flying the jet through ALL of the setup below
    # (Bombable menus, arming, target search, the accelerate-to-corner wait).
    # Releasing the stick here left it unpiloted for minutes and it flew into
    # terrain -- see startup()'s docstring.
    startup(t, hand_over_stick=False)

    # HOLD THE JET WHILE WE SET UP. startup() ends by calling connect_stick(),
    # which clears the autopilot locks so our stick reaches the FCS -- and with
    # no control loop running yet that leaves NOBODY flying. All the telnet
    # setup below (bombable menu, arming, radar, reading three enemies -- about
    # 20 round trips) takes tens of seconds, and confirmed live the jet dived
    # from 18000ft to 10700ft during it and hit terrain ("Aircraft hit
    # ShrubCover"), with the pilot's first command a futile 5.38g recovery.
    # So: re-engage FlightGear's own hold, do the setup, and only take the
    # stick back immediately before the loop.
    hold_alt = t.get("/position/altitude-ft")
    t.set("/autopilot/settings/target-altitude-ft", hold_alt)
    t.set("/autopilot/settings/true-heading-deg",
          round(float(t.get("/orientation/heading-deg")), 1))
    t.set("/autopilot/locks/altitude", "altitude-hold")
    t.set("/autopilot/locks/heading", "true-heading-hold")
    # HOLD SPEED TOO. Without the autothrottle the jet bleeds energy through
    # the whole setup and the climb: confirmed live it handed over at 273kt,
    # where only (273/180)^2 = 2.3g exists, so it could not turn onto a target
    # 172deg behind without departing. A slow F-15 cannot fight.
    t.set("/autopilot/settings/target-speed-kt", CORNER_SPEED_KT)
    t.set("/autopilot/locks/speed", "speed-with-throttle")
    time.sleep(1)

    enable_bombable(t)
    fp.arm_aim9(t)
    t.set(fp.RADAR_SERVICEABLE, 1)
    t.set(fp.RADAR_STANDBY, 0)
    t.set(fp.RADAR_DISPLAY, "true")
    t.set(fp.RADAR_RANGE, 40)

    enemies = read_enemies(t)
    print("  enemies: %s" % [e["callsign"] for e in enemies], flush=True)
    if not enemies:
        print("FATAL: no Bombable AI spawned", flush=True)
        return

    # Do NOT climb to the enemy block. Climbing costs energy, and energy is
    # what a dogfight is fought with -- an earlier version climbed to 18000ft
    # and handed over at 273kt, unable to turn. The enemies are spawned near
    # our own altitude instead, and we accelerate to corner speed before the
    # fight starts.
    print("  accelerating to %.0fkt before taking the stick" % CORNER_SPEED_KT,
          flush=True)
    for _ in range(25):
        spd = float(t.get("/velocities/airspeed-kt") or 0)
        if spd >= CORNER_SPEED_KT - 40:
            break
        time.sleep(2)
    print("  at %.0fft / %.0fkt, taking the stick" % (
        float(t.get("/position/altitude-ft")),
        float(t.get("/velocities/airspeed-kt") or 0)), flush=True)

    connect_stick(t)     # release the autopilot -- the loop starts NOW

    nb = NativeBridge()
    telem = nb.get_telemetry()
    pilot = CombatPilot(aggression=1.0, callsign="BLUE1",
                        target_alt_ft=telem["altitude_m"] / 0.3048)
    print("  defending %.0fft" % pilot.target_alt_ft, flush=True)

    # Background enemy poller: keeps telnet entirely out of the control loop.
    # Also does radar designation/lock and reads our own damage here, for the
    # same reason -- these are telnet round trips and must never touch the
    # 20Hz hot loop.
    state = {"enemies": enemies, "stop": False, "locked": False,
            "locked_on": None, "own_damage": 0.0, "g_cmd": 0.0,
            "want_target": None, "last_lock_t": -1e9}

    # ONE-CONNECTION-PER-THREAD.
    #
    # Every background thread opens its OWN Telnet. Sharing one connection was
    # the root of a whole family of bugs that looked like aircraft or weapon
    # problems and were neither.
    #
    # FlightGear's telnet is a request/response stream. A single timeout leaves
    # one unread reply in the buffer, and from that moment every read returns
    # the PREVIOUS command's answer -- the stream is permanently off by one,
    # silently, with no exception. Locking made the connection thread-SAFE
    # (one command at a time) but cannot resynchronise a desynced stream, and
    # holding the lock across a probe's sleep starved the other threads
    # instead.
    #
    # Proved by comparison, mid-fight, on the very same running sim: on a
    # FRESH connection missile_status() returned '1' (MISSILE_LOCK) and
    # designated_callsign() returned 'RED1' -- everything healthy -- while the
    # shared connection was reading the seeker as "-" and the contact as None.
    # The seeker had been locking all along; we were reading a stale stream and
    # concluding the weapon was broken.
    #
    # Separate sockets remove the interleaving entirely, which is both simpler
    # and stronger than any locking discipline. FlightGear accepts multiple
    # telnet clients on the same port.

    if FCS_DIAG:
        diag_t = Telnet()

        def poll_fcs_diag():
            t = Telnet()   # own connection -- see ONE-CONNECTION-PER-THREAD
            while not state["stop"]:
                try:
                    vals = [diag_t.get(p) for p in FCS_DIAG_PROPS]
                    print("FCS_DIAG g_cmd=%.3f %s" % (
                        state["g_cmd"],
                        " ".join("%s=%s" % (p.rsplit("/", 1)[-1], v)
                                 for p, v in zip(FCS_DIAG_PROPS, vals))),
                          flush=True)
                except Exception as exc:
                    print("  fcs diag poll error: %s" % exc, flush=True)
                time.sleep(0.2)

        diag_thread = threading.Thread(target=poll_fcs_diag, daemon=True)
        diag_thread.start()

    def poll_enemies():
        # OWN CONNECTION. See ONE-CONNECTION-PER-THREAD below. Guarded so the
        # thread cannot die silently if FlightGear refuses the socket.
        try:
            t = Telnet()
        except Exception as exc:
            print("  enemy poller: NO TELNET (%s) -- thread dead" % exc,
                  flush=True)
            return
        last_designated = None
        while not state["stop"]:
            try:
                got = read_enemies(t)
                if got:
                    state["enemies"] = got
                # DESIGNATION IS NOT HOSTAGE TO THIS CYCLE'S READ.
                #
                # All of the radar work below used to sit inside
                # `if got:`, so ONE empty read skipped it entirely --
                # and with three threads sharing one telnet that was
                # frequent enough that designate() was never called in
                # a whole run (0 designations, 0 locks, 0 shots, no
                # errors, because nothing raised). Work from the last
                # known list: a dropped read should cost fresh
                # positions for one cycle, never the designation the
                # seeker depends on.
                if state["enemies"]:
                    # DESIGNATE WHAT THE PILOT IS ACTUALLY CHASING.
                    #
                    # This used to compute its OWN "nearest" here, from its
                    # own lagging telnet position reads, entirely independent
                    # of the target the control loop had picked at 20Hz from
                    # the native FDM. The two disagreed constantly, and the
                    # radar was then designating a different aircraft than the
                    # one we were pursuing. Confirmed live by probing the sim
                    # mid-fight: every seeker precondition passed -- status=0
                    # searching, class A, target_air=1, contact valid, type
                    # AIR, speed 495kt, fire range 0.2-9nm, FOV 27.5deg -- but
                    # armament.contact.get_range() read 6.77nm while the jet
                    # was 0.75nm behind its actual target at 7deg off the
                    # nose. The seeker was faithfully searching the wrong
                    # aeroplane, so FOV_check could never pass and no AIM-9
                    # ever locked.
                    #
                    # It also broke the lock even when it did work, because
                    # the main loop gates has_lock on
                    # `state["locked_on"] == target["callsign"]` -- so a lock
                    # on the poller's choice was rejected anyway.
                    want = state.get("want_target")
                    if want is None:
                        want = state["enemies"][0]["callsign"]
                    # VERIFY THE DESIGNATION, DO NOT ASSUME IT STUCK.
                    #
                    # This used to designate only when the wanted target
                    # CHANGED, so a designation dropped later was never
                    # repaired. Probed live mid-fight: the seeker was
                    # searching with four rounds available and the jet was
                    # pointing 8deg off the bandit -- inside the firing gate
                    # at last -- while awg_9.active_u_callsign and
                    # armament.contact both read nil. A perfect shot with
                    # nothing designated to shoot at, and the run finished
                    # with 0 locks.
                    if want != last_designated or fp.designated_callsign(t) != want:
                        fp.designate(t, want)
                        last_designated = want
                        state["locked"] = False
                    # state["locked"] is owned by poll_lock (fast cadence).
                    state["locked_on"] = last_designated
                    node = next((e["node"] for e in state["enemies"]
                                 if e["callsign"] == last_designated), None)
                    if node is not None:
                        state["locked_on_node"] = node
                dmg = t.get("/bombable/attributes/damage")
                if dmg not in (None, ""):
                    state["own_damage"] = float(dmg)
            except Exception as exc:          # a dropped telnet must not kill the flight
                print("  enemy poll error: %s" % exc, flush=True)
            time.sleep(ENEMY_POLL_SEC)

    poller = threading.Thread(target=poll_enemies, daemon=True)
    poller.start()

    def poll_lock():
        # A THREAD MUST NOT DIE ON ITS CONNECTION. This constructor used to sit
        # outside any try, so if FlightGear refused the socket the daemon
        # thread vanished silently -- no log line, no exception anywhere. For
        # the fire worker that meant every shot request was dropped: a run
        # logged 50 locks, 12 moments with lock+range+boresight all satisfied,
        # and FIRED=0 REFUSED=0, because nothing was consuming the flag.
        try:
            t = Telnet()
        except Exception as exc:
            print("  %s: NO TELNET (%s) -- thread dead" % ("lock poller", exc),
                  flush=True)
            return   # own connection -- see ONE-CONNECTION-PER-THREAD
        """Watch the seeker on its OWN fast cadence.

        The enemy poller reads every bandit's position first -- a dozen-plus
        telnet round trips -- and only then checks the seeker, so one cycle
        can take seconds. A lock that comes and goes inside that window is
        never seen, and a shot that was available is never taken: run 8 got
        exactly ONE launch in ~900 steps despite the seeker reaching
        MISSILE_LOCK (probed live at status=1) while the nose was on target.
        Missile status is three cheap round trips, so poll it by itself and
        let the expensive enemy sweep run at its own pace. Safe to share the
        connection now that Telnet holds a lock -- see its docstring.
        """
        while not state["stop"]:
            try:
                st = fp.missile_status(t)
                if st == str(fp.MISSILE_LOCK):
                    state["last_lock_t"] = time.monotonic()
                elif st == str(fp.MISSILE_SEARCH):
                    # Revive a stalled search loop -- see fp.kick_seeker. A
                    # dead timer chain looks exactly like "searching" and never
                    # locks, however good the geometry is.
                    fp.kick_seeker(t)
                # LATCH THE LOCK BRIEFLY.
                #
                # The seeker's real lock flickers, and this flag is read over
                # telnet with a 0.3s sleep, so short genuine locks are missed
                # outright. Measured over run 27's 400 logged samples: lock
                # 17%, range gate 46%, boresight 18% -- 53 samples had BOTH
                # range and boresight, but only 3 also had the lock flag, so
                # 50 good firing positions were thrown away on a stale read.
                # While locked the jet reached 1.21nm and 7deg, well inside
                # both gates.
                #
                # Latching is safe only because fp.fire_srm tests the status
                # and releases in ONE Nasal statement: if the seeker is not
                # actually locked at that instant fox2 refuses, the round
                # stays on the rail, and the caller un-counts it. The Nasal
                # check is the real gate; this flag only decides when to try.
                state["locked"] = (
                    time.monotonic() - state.get("last_lock_t", -1e9)
                    <= LOCK_LATCH_SEC)
            except Exception as exc:
                print("  lock poll error: %s" % exc, flush=True)
            time.sleep(LOCK_POLL_SEC)

    lock_poller = threading.Thread(target=poll_lock, daemon=True)
    lock_poller.start()

    target = enemies[0]
    committed_target = None
    min_range_nm = 1e9
    kills = 0
    fire_requested = {"missile": False, "gun_on": False}
    known_alive = {e["callsign"] for e in enemies}
    loop_t0 = time.monotonic()

    # Firing is a telnet action, so it also happens off the hot loop. The main
    # loop only ever SETS a flag; this thread notices and acts on it, which
    # keeps every fp.* call (each several telnet round trips) off the 20Hz path.
    def fire_worker():
        # A THREAD MUST NOT DIE ON ITS CONNECTION. This constructor used to sit
        # outside any try, so if FlightGear refused the socket the daemon
        # thread vanished silently -- no log line, no exception anywhere. For
        # the fire worker that meant every shot request was dropped: a run
        # logged 50 locks, 12 moments with lock+range+boresight all satisfied,
        # and FIRED=0 REFUSED=0, because nothing was consuming the flag.
        try:
            t = Telnet()
        except Exception as exc:
            print("  %s: NO TELNET (%s) -- thread dead" % ("fire worker", exc),
                  flush=True)
            return   # own connection -- see ONE-CONNECTION-PER-THREAD
        gun_currently_on = False
        while not state["stop"]:
            if fire_requested["missile"]:
                fire_requested["missile"] = False
                # locked_on_node is NOT required to shoot -- fp.fire_srm
                # releases through aircraft.Current_srm and never looks at the
                # AI node. Requiring it here silently dropped the shot (the
                # request was consumed, then discarded, with nothing logged)
                # whenever the poller had not managed to set it.
                if not state["locked"]:
                    print("  fire skipped -- no lock at trigger time", flush=True)
                elif True:
                    # The lock is already established by the poller's own
                    # designate()/missile_status() cycle above; this only
                    # needs to release the round, not re-run the whole
                    # search/lock sequence.
                    # Atomic check-and-release, and believe only the
                    # read-back: status 2 (MISSILE_FLYING) means the round
                    # actually left the rail. The old unconditional print
                    # claimed launches that never happened -- see fp.fire_srm.
                    st = fp.fire_srm(t)
                    if st == str(fp.MISSILE_FLYING):
                        print("  FIRED missile at %s" % state.get("locked_on"),
                              flush=True)
                    else:
                        # Rail still loaded: give the budget back so a
                        # refused attempt costs nothing but the cooldown.
                        pilot.missiles_fired = max(0, pilot.missiles_fired - 1)
                        continue   # do not re-arm; nothing was launched
                    state["locked"] = False   # this round is gone
                    # RE-ARM THE NEXT ROUND, or we fire exactly once and then
                    # sit on three live AIM-9s for the rest of the fight.
                    # release() puts the round at MISSILE_FLYING(2), and
                    # weapons.nas:226 only hands Current_srm to the next
                    # missile when the selector is re-run against a refreshed
                    # count (`Current_srm == nil or Current_srm.status == 2`).
                    # Nothing does that on its own -- the cockpit does it when
                    # the pilot re-selects -- so do it here, and start the new
                    # seeker searching the same way arm_aim9 does.
                    time.sleep(0.5)
                    fp.rearm_next_aim9(t)
            want_gun = fire_requested["gun_on"]
            if want_gun != gun_currently_on:
                fp.fire_gun(t, want_gun)
                gun_currently_on = want_gun
            time.sleep(0.1)

    firer = threading.Thread(target=fire_worker, daemon=True)
    firer.start()

    for step in range(MAX_STEPS):
        telem = nb.get_telemetry()
        own_lat, own_lon = telem["lat_deg"], telem["lon_deg"]

        # CRASH DETECTION. Without this, a run that has already hit the
        # ground just keeps commanding a dead, stationary wreck for the rest
        # of MAX_STEPS -- confirmed live, a real crash at step ~1920 produced
        # over 2000 more logged lines of agl=2ft/spd=0kt/roll=+169deg before
        # the script finally stopped on its own step budget. Stop the instant
        # it is unambiguous: essentially zero AGL and airspeed together.
        if telem["agl_m"] / 0.3048 < 20.0 and telem["airspeed_kt"] < 20.0:
            print("WE CRASHED at step %d (agl=%.0fft spd=%.0fkt)"
                  % (step, telem["agl_m"] / 0.3048, telem["airspeed_kt"]),
                  flush=True)
            break

        enemies = state["enemies"]
        if not enemies:
            print("ALL ENEMIES DESTROYED at step %d -- VICTORY" % step, flush=True)
            break

        # Detect a kill: Bombable removes the node once destroyed, or its own
        # damage attribute reaches 100. Either way this jet is no longer a
        # target and disappears from `enemies` on the next poll.
        now_alive = {e["callsign"] for e in enemies if e.get("alive", True)}
        newly_dead = known_alive - now_alive
        for cs in newly_dead:
            kills += 1
            print("  %s DESTROYED (kill #%d)" % (cs, kills), flush=True)
        known_alive = now_alive

        if state["own_damage"] >= 100.0:
            print("WE WERE SHOT DOWN at step %d (damage=%.0f)"
                  % (step, state["own_damage"]), flush=True)
            break

        # COMMIT TO ONE TARGET UNTIL IT IS DEAD. Kill them one by one.
        #
        # This used to re-pick the nearest enemy every step at 20Hz, and with
        # three bandits in a turning fight the "nearest" flips constantly --
        # logged live switching RED2 (2.58nm) -> RED1 (3.79nm) -> RED2 within
        # a couple of seconds. That thrashed the RADAR, which is the part
        # that actually matters: fp.designate() costs ~2.4s per call (it sets
        # the callsign, waits for the radar to sweep it up, then verifies),
        # so a target that changes faster than that can NEVER be designated.
        #
        # Confirmed live by probing mid-fight: the jet was tracking RED1 with
        # the nose 3deg off at 2.58nm -- a textbook shot -- while
        # awg_9.active_u_callsign and armament.contact were both still RED2,
        # 5.1nm away. Every other seeker precondition passed. The seeker was
        # simply pointed at the wrong aeroplane because the designation could
        # not keep up with the target selection.
        #
        # So pick once, and hold until the target dies or goes truly beyond
        # reach. Switching only on death is also what "take them down one by
        # one" means tactically: chasing whichever happens to be closest this
        # instant is how you merge with all three and kill none.
        alive_callsigns = {e["callsign"] for e in enemies}
        if committed_target not in alive_callsigns:
            committed_target = None          # dead, or gone from the list
        if committed_target is not None:
            cur = next(e for e in enemies if e["callsign"] == committed_target)
            cur_nm = haversine_m(own_lat, own_lon, cur["lat"], cur["lon"]) * M_TO_NM
            if cur_nm > ABANDON_RANGE_NM:
                print("  abandoning %s at %.1fnm -- out of reach"
                      % (committed_target, cur_nm), flush=True)
                committed_target = None
        if committed_target is None:
            new_tgt = min(enemies, key=lambda e: haversine_m(
                own_lat, own_lon, e["lat"], e["lon"]))
            committed_target = new_tgt["callsign"]
            print("  committing to %s" % committed_target, flush=True)
        target = next(e for e in enemies if e["callsign"] == committed_target)

        # Publish the chosen target so the poller designates THIS one on the
        # radar. Single source of truth -- see poll_enemies.
        state["want_target"] = target["callsign"]
        rng_m = haversine_m(own_lat, own_lon, target["lat"], target["lon"])
        rng_nm = rng_m * M_TO_NM
        min_range_nm = min(min_range_nm, rng_nm)

        # Threat: any enemy actively firing at us, within its gun's real
        # range (Bombable's F-15 AI gun maxDamageDistance_m is 1500m). This is
        # what triggers EVADE -- a real "they are shooting at me" signal, not
        # only reacting after damage has already landed.
        attacker = next((e for e in enemies if e.get("firing")
                         and haversine_m(own_lat, own_lon, e["lat"], e["lon"])
                            < 1600), None)
        if attacker is not None:
            threat_brg = angle_diff_deg(
                bearing_deg(own_lat, own_lon, attacker["lat"], attacker["lon"]),
                telem["heading_deg"])
            threat = dict(missile_inbound=True, missile_bearing_deg=threat_brg)
        else:
            threat = dict(missile_inbound=False, missile_bearing_deg=None)

        has_lock = state["locked"] and state.get("locked_on") == target["callsign"]

        cmd = pilot.step(dict(
            lat=own_lat, lon=own_lon,
            alt_ft=telem["altitude_m"] / 0.3048,
            # AGL and pitch drive the dynamic ground-recovery floor. An MSL
            # deck is meaningless here: terrain sits at ~5400ft.
            agl_ft=telem["agl_m"] / 0.3048,
            pitch_deg=telem["pitch_deg"],
            heading_deg=telem["heading_deg"], roll_deg=telem["roll_deg"],
            airspeed_kt=telem["airspeed_kt"],
            climb_rate_fps=telem["climb_rate_fps"],
            tgt_lat=target["lat"], tgt_lon=target["lon"],
            tgt_heading_deg=target["heading"], tgt_speed_kt=target["speed_kt"],
            tgt_alt_ft=target["alt_ft"],
            tgt_range_nm=rng_nm,
            has_lock=has_lock,
        ), threat)

        state["g_cmd"] = cmd.g_cmd
        if cmd.fire_missile:
            fire_requested["missile"] = True
        fire_requested["gun_on"] = cmd.fire_gun

        nb.send_controls(
            aileron=max(-1.0, min(1.0, cmd.roll_rate_dps / CAS_ROLL_RATE_FULL_DPS)),
            elevator=g_to_elevator(cmd.g_cmd),
            rudder=0.0,
            throttle=cmd.throttle,
            speedup=SPEED_UP,
        )

        if step % 20 == 0:
            hz = (step + 1) / max(1e-6, time.monotonic() - loop_t0)
            brg = bearing_deg(own_lat, own_lon, target["lat"], target["lon"])
            print("  step %d %s tgt=%s rng=%.2fnm brg_err=%+.0f alt=%.0f "
                  "agl=%.0f spd=%.0f roll=%+.0f g=%.2f lock=%s %.1fHz "
                  "| enemies=%d dmg=%.0f ourdmg=%.0f kills=%d" % (
                      step, cmd.state, target["callsign"], rng_nm,
                      angle_diff_deg(brg, telem["heading_deg"]),
                      telem["altitude_m"] / 0.3048, telem["agl_m"] / 0.3048,
                      telem["airspeed_kt"],
                      telem["roll_deg"], cmd.g_cmd, has_lock, hz, len(enemies),
                      target.get("damage", 0.0), state["own_damage"],
                      kills), flush=True)
            # Enemy heading vs bearing-to-us answers "is it chasing me?"
            back = bearing_deg(target["lat"], target["lon"], own_lat, own_lon)
            print("      %s hdg=%.0f spd=%.0f alt=%.0f bearing_to_us=%.0f "
                  "aspect=%+.0f firing=%s (us: spd=%.0f alt=%.0f)" % (
                target["callsign"], target["heading"], target["speed_kt"],
                target["alt_ft"],
                back, angle_diff_deg(back, target["heading"]),
                target.get("firing", False), telem["airspeed_kt"],
                telem["altitude_m"] / 0.3048), flush=True)

        time.sleep(1.0 / (STEP_HZ * SPEED_UP))
    else:
        step = MAX_STEPS - 1

    rate = (step + 1) / max(1e-6, time.monotonic() - loop_t0)
    state["stop"] = True
    print("RESULT closest_approach=%.2fnm enemies_left=%d kills=%d "
          "our_damage=%.0f loop_rate=%.1fHz"
          % (min_range_nm, len(state["enemies"]), kills, state["own_damage"],
             rate), flush=True)
    if rate < 10.0:
        print("WARNING loop ran at %.1fHz -- the jet is barely being flown"
              % rate, flush=True)
    nb.close()
    t.close()


if __name__ == "__main__":
    main()
