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
    out = []
    i = 0
    while True:
        cs = t.get("/ai/models/aircraft[%d]/callsign" % i)
        if cs is None:
            return out
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


def main():
    t = Telnet()
    time.sleep(2)
    startup(t)

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
            "locked_on": None, "own_damage": 0.0}

    def poll_enemies():
        last_designated = None
        while not state["stop"]:
            try:
                got = read_enemies(t)
                if got:
                    state["enemies"] = got
                    nearest = min(got, key=lambda e: haversine_m(
                        float(t.get("/position/latitude-deg") or 0),
                        float(t.get("/position/longitude-deg") or 0),
                        e["lat"], e["lon"]))
                    if nearest["callsign"] != last_designated:
                        fp.designate(t, nearest["callsign"])
                        last_designated = nearest["callsign"]
                        state["locked"] = False
                    st = fp.missile_status(t)
                    state["locked"] = (st == str(fp.MISSILE_LOCK))
                    state["locked_on"] = last_designated
                    state["locked_on_node"] = nearest["node"]
                dmg = t.get("/bombable/attributes/damage")
                if dmg not in (None, ""):
                    state["own_damage"] = float(dmg)
            except Exception as exc:          # a dropped telnet must not kill the flight
                print("  enemy poll error: %s" % exc, flush=True)
            time.sleep(ENEMY_POLL_SEC)

    poller = threading.Thread(target=poll_enemies, daemon=True)
    poller.start()

    target = enemies[0]
    min_range_nm = 1e9
    kills = 0
    fire_requested = {"missile": False, "gun_on": False}
    known_alive = {e["callsign"] for e in enemies}
    loop_t0 = time.monotonic()

    # Firing is a telnet action, so it also happens off the hot loop. The main
    # loop only ever SETS a flag; this thread notices and acts on it, which
    # keeps every fp.* call (each several telnet round trips) off the 20Hz path.
    def fire_worker():
        gun_currently_on = False
        while not state["stop"]:
            if fire_requested["missile"]:
                fire_requested["missile"] = False
                if state["locked"] and state.get("locked_on_node"):
                    # The lock is already established by the poller's own
                    # designate()/missile_status() cycle above; this only
                    # needs to release the round, not re-run the whole
                    # search/lock sequence.
                    t.nasal('aircraft.Current_srm.release();')
                    print("  FIRED missile at %s" % state.get("locked_on"),
                          flush=True)
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

        # Chase the nearest one. Threat prioritisation across multiple
        # attackers is deliberately this simple for now.
        target = min(enemies, key=lambda e: haversine_m(
            own_lat, own_lon, e["lat"], e["lon"]))

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
            tgt_range_nm=rng_nm,
            has_lock=has_lock,
        ), threat)

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
            print("      %s hdg=%.0f spd=%.0f bearing_to_us=%.0f aspect=%+.0f "
                  "firing=%s (us: spd=%.0f)" % (
                target["callsign"], target["heading"], target["speed_kt"],
                back, angle_diff_deg(back, target["heading"]),
                target.get("firing", False), telem["airspeed_kt"]), flush=True)

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
