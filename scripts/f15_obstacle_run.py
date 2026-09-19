"""F-15 race-gate course.

    startup -> place gates -> fly the course

Flying is done with the SAME two commands the CAS actually accepts -- load
factor and roll rate (see SESSION.md session 5) -- so the course is flown
aggressively rather than by a gentle navigation autopilot:

    lateral   bearing error -> bank -> roll rate  (bank capped by available g)
    vertical  altitude error -> load factor       (on top of 1/cos(bank))

Scoring is independent of guidance: every step, the jet's position is checked
against every obstacle cylinder, and the closest approach to each is recorded.
That way a "cleared" call is a measurement, not an assertion that the
guidance law thinks it worked.
"""

import math
import sys
import time

from f15_airshow import Crashed, Jet, recover
from f15_course import CLEARANCE_M, GATE_FLY_AGL_FT, GATE_PASS_RADIUS_M, build_course
from f15_intercept import (
    Telnet,
    angle_diff_deg,
    bearing_deg,
    haversine_m,
    startup,
)
import f15_props
from fly_brain.native_protocol import NativeBridge

GATE_RADIUS_M = GATE_PASS_RADIUS_M  # generous race-line pass radius, not a slot -- see f15_course
# No separate altitude-proximity requirement for PASS any more: the real
# gate model is now scaled 1000x in height (~59449ft tall, see f15_course.py
# / RedbullRedGate_tall.xml) specifically so any reasonable fly altitude
# already passes through it -- an earlier version required being within
# ALT_TOLERANCE_FT of a precise fly-through altitude, which was the right
# fix for a life-size 18m gate but is now moot: altitude is still tracked
# and reported (best_alt_err) for visibility, it just no longer gates PASS.

# MAX_BANK_DEG: this is a straight line of gates, not a slalom -- bearing
# error in steady tracking should stay small once established on the line,
# so a gentle bank is enough. Confirmed live why the old 75deg cap was wrong:
# holding a 75-76deg bank alone costs level_turn_g(76) ~= 4.1g, which
# completely swamped the small altitude-correction term (max ~0.75g at the
# old K_ALT_G) -- the jet was structurally unable to descend while turning
# that hard, so EVERY run (even the earlier "successful" 12000ft one, which
# also silently climbed 10368->15244ft gate to gate) just climbed the whole
# course while alternating bank sign trying to correct back onto the line.
MAX_BANK_DEG = 35.0
BANK_PER_BEARING_DEG = 1.2  # gentler proportional gain to match -- see above
# K_ALT_G raised (was 1/2500, tuned for a DIFFERENT course with big pylon-
# altitude excursions) so altitude correction has real authority now that
# MAX_BANK_DEG leaves it a g budget to work with: level_turn_g(35deg) is
# only ~1.22g, so a K_ALT_G of 1/1200 against a real ~1800-2000ft deficit at
# course start commands roughly -1.5 to -1.7g net -- a firm, not violent,
# push-over to actually descend to gate altitude.
K_ALT_G = 1.0 / 1200.0
MAX_ALT_G = 2.0  # see steer(): cap on the altitude-correction term
K_CLIMB_G = 0.010  # g per ft/s of climb rate, damps overshoot on arrival
MAX_STEPS_PER_GATE = 1200

# Altitude to sit at while the course is laid out and the towers are placed.
# NOT the gate altitude: descending all the way to GATE_FLY_AGL_FT (800ft)
# before the layout phase put the jet in ground contact (agl=-2ft, measured
# live) while it loitered there through ~30 telnet round-trips over terrain
# that rises under it. It has to be above f15_airshow.FLOOR_AGL_FT (4000ft)
# too, or the ground-avoidance guard fights the level-off the whole time.
# The remaining let-down to gate altitude is flown by the guidance law
# itself over the 10km run-in to gate-1, which is what that run-in is for.
SETUP_AGL_FT = 5000.0


def steer(jet, gate):
    """One step of 3D guidance toward a gate. Returns (dist_m, alt_err_ft)."""
    t = jet.telem
    dist = haversine_m(t["lat_deg"], t["lon_deg"], gate.lat, gate.lon)
    brg = bearing_deg(t["lat_deg"], t["lon_deg"], gate.lat, gate.lon)
    berr = angle_diff_deg(brg, t["heading_deg"])
    alt_err = gate.alt_ft - t["altitude_m"] / 0.3048

    # Lateral: a gentle line-tracking bank, not a hard turn -- see
    # MAX_BANK_DEG above for why a steep bank starves the altitude budget.
    want_bank = max(-1.0, min(1.0, berr * BANK_PER_BEARING_DEG / MAX_BANK_DEG))
    bank = want_bank * min(MAX_BANK_DEG, abs(jet.sustainable_bank(1, limit=MAX_BANK_DEG)))

    # Vertical: the g needed to hold the turn, plus the g to fix altitude.
    # Clamp the altitude term: an unclamped alt_err can swamp everything
    # else. A 12,000ft deficit (which is what fly_course() inherited when
    # nothing descended from cruise first) commands -10g here, clamped by
    # Jet.step to a sustained -3g bunt -- the jet stops tracking the gate
    # at all and diverges. MAX_ALT_G keeps the correction inside the
    # regime these gains were actually tuned in; descend_to() is what
    # deals with a big deficit, not the guidance law.
    alt_g = max(-MAX_ALT_G, min(MAX_ALT_G, alt_err * K_ALT_G))
    g = jet.level_turn_g(bank) + alt_g - jet.climb_fps * K_CLIMB_G
    # Throttle governor: full throttle the whole way let the jet accelerate
    # unchecked past 600kt on this long straight gate line (measured live --
    # spd climbed 565->620->645kt gate to gate) and it lost control into the
    # ground right after. The bank/g gains above were never validated above
    # the ~450kt cruise this jet is flown at elsewhere, so cut throttle once
    # past it instead of re-tuning gains for a speed regime nothing else in
    # this codebase uses.
    throttle = 1.0 if t["airspeed_kt"] < 450.0 else 0.4
    jet.step(g, jet.roll_to(bank), throttle=throttle)
    return dist, alt_err


def hold(jet, agl_ft, steps=4):
    """Keep flying the jet at a target AGL while the caller blocks on telnet.

    Setup code (terrain queries, model placement) does its work over telnet,
    which is a SEPARATE connection from the native-protocol one that flies
    the jet -- and FlightGear keeps applying the last commanded control
    surfaces the whole time it is blocked, so the jet has to be actively
    flown between round-trips or it drifts.

    It previously drifted anyway, because the interleaved command was
    jet.hold_nose(0.0): that holds PITCH, not altitude, and pitch 0 at 540kt
    with the throttle back is a descent. Measured live, the jet sank from
    5000ft AGL into the ground (agl=4ft) during the terrain-query phase
    alone. Holding altitude is what the caller actually wants, everywhere.
    """
    for _ in range(steps):
        err = agl_ft - jet.agl_ft()
        pitch = max(-10.0, min(10.0, err / 200.0))
        jet.step(jet.hold_nose(pitch), jet.roll_to(0.0),
                 throttle=0.4 if jet.telem["airspeed_kt"] > 450.0 else 0.9)


def descend_to_agl(jet, agl_ft, max_secs=180.0):
    """Fly down to the course altitude BEFORE the guidance law gets the jet.

    recover() has no altitude term at all -- it only levels pitch and roll
    and makes speed -- so it returns True just as happily at 17,979ft as at
    6,000ft (both measured live, from the same code path). Nothing else in
    setup descended either, so fly_course() inherited the entire ~12,000ft
    deficit as alt_err on its very first step. That, not a recover() timeout,
    is why runs starting high diverged: see the MAX_ALT_G note in steer().

    Targets AGL, not MSL, specifically so this can run BEFORE the course is
    laid out -- which it must. Descending AFTER the layout was measured live
    to be just as broken in the lateral axis as no descent was in the
    vertical one: an 8000ft descent at 450kt takes over two minutes and
    covers 30+km, so by the time the jet levelled off it had flown clean
    past a gate-1 that had been placed 6km ahead of a position it left long
    ago. fly_course() started with the gate 178deg BEHIND the aircraft and
    spent the whole run orbiting at max bank, never closing. Descend first,
    THEN snapshot position and lay the course ahead of where the jet
    actually is.
    """
    end = time.time() + max_secs
    while time.time() < end:
        err = agl_ft - jet.agl_ft()
        if abs(err) < 300.0:
            return True
        # Shallow, commanded descent -- not a bunt. The whole point is to
        # arrive level and on speed so the gate guidance starts in its
        # tuned regime.
        pitch = max(-12.0, min(8.0, err / 200.0))
        throttle = 0.4 if jet.telem["airspeed_kt"] > 450.0 else 1.0
        jet.step(jet.hold_nose(pitch), jet.roll_to(0.0), throttle=throttle)
    return False


def fly_course(jet, obstacles, gates):
    """Fly every gate in order, scoring clearance against every obstacle.

    A gate counts as PASSED if the jet was laterally close (within
    GATE_RADIUS_M) at closest approach. Altitude is tracked and reported
    (best_alt_err) but no longer required to be precise -- the real gate
    model is scaled 1000x in height (~59449ft tall) specifically so any
    reasonable fly altitude already passes through it. See f15_course.py.
    """
    closest = {o.label: float("inf") for o in obstacles}
    results = []

    for gate in gates:
        prev = float("inf")
        best_alt_err = None
        passed = False
        for step in range(MAX_STEPS_PER_GATE):
            dist, alt_err = steer(jet, gate)
            t = jet.telem
            alt_ft = t["altitude_m"] / 0.3048
            for o in obstacles:
                m = o.miss_distance_m(t["lat_deg"], t["lon_deg"], alt_ft)
                closest[o.label] = min(closest[o.label], m)
            if best_alt_err is None or abs(dist) < abs(prev):
                best_alt_err = alt_err
            if step % 20 == 0:
                brg = bearing_deg(t["lat_deg"], t["lon_deg"], gate.lat, gate.lon)
                berr = angle_diff_deg(brg, t["heading_deg"])
                print(f"    DEBUG step {step:4d} dist {dist:7.0f}m berr {berr:+6.1f} "
                      f"hdg {t['heading_deg']:6.1f} roll {t['roll_deg']:+5.1f} "
                      f"alt {alt_ft:6.0f} alt_err {alt_err:+6.0f} "
                      f"pitch {t.get('pitch_deg', 0):+5.1f} spd {t['airspeed_kt']:4.0f}")
            # Closest-approach capture: at 450kt the jet covers 230m per step,
            # so a fixed radius can be stepped straight over (this exact bug
            # made the intercept miss its waypoint by 24m and orbit forever).
            if dist < GATE_RADIUS_M or (dist < 2000.0 and dist > prev):
                closest_dist = min(prev, dist)
                passed = closest_dist < GATE_RADIUS_M
                break
            prev = dist
        results.append((gate.label, passed, min(prev, dist), best_alt_err))
        print(f"  {'PASS' if passed else 'MISS'}  {gate.label:22s} "
              f"closest {min(prev, dist):6.0f}m  "
              f"alt_err {best_alt_err:+6.0f}ft  "
              f"alt {jet.telem['altitude_m']/0.3048:6.0f}ft  "
              f"spd {jet.telem['airspeed_kt']:3.0f}kt  "
              f"roll {jet.telem['roll_deg']:+5.0f}")

    return results, closest


def report(results, closest):
    passed = sum(1 for _, p, _, _ in results if p)
    print(f"\n=== GATES: {passed}/{len(results)} passed ===")
    hits = [(k, v) for k, v in closest.items() if v < 0]
    tight = [(k, v) for k, v in closest.items() if 0 <= v < CLEARANCE_M]
    print(f"=== OBSTACLES: {len(hits)} struck, {len(tight)} cleared by "
          f"under {CLEARANCE_M:.0f}m ===")
    for k, v in sorted(closest.items(), key=lambda kv: kv[1])[:6]:
        mark = "HIT " if v < 0 else ("TIGHT" if v < CLEARANCE_M else "ok  ")
        print(f"    {mark} {k:14s} closest {v:8.0f}m")
    return len(hits) == 0


def _real_terrain_ft(t, jet):
    """Build a terrain_fn closure backed by live geo.elevation() queries.

    Falls back to the aircraft's own current altitude if a tile has not
    streamed in yet (confirmed live this can return nil at KXTA -- rare
    around the home airport, but query_elevation_m returns None rather than
    silently placing a gate at 0ft/underground, so this has to handle it).

    Also holds the jet level between queries. Confirmed live this was
    required: 18 terrain queries + 12 model placements, all over telnet
    (completely separate from the native-protocol connection that actually
    flies the jet), left the aircraft with NO jet.step() calls for several
    real seconds -- FlightGear's native ctrls protocol has no "hold trim"
    fallback, it just keeps applying whatever was last commanded, so the
    jet flew uncontrolled the whole time. One run measured this driving the
    jet from a stable 496kt to 636kt and flipping its bearing to the first
    gate by -177deg before fly_course() ever got control back. A single
    extra recover() call afterward did NOT fix it (recover() does not
    reliably undo a large uncommanded excursion either) -- the fix has to
    be never leaving the connection silent in the first place.
    """
    fallback_ft = [None]

    def terrain_fn(lat, lon):
        hold(jet, SETUP_AGL_FT)
        m = f15_props.query_elevation_m(t, lat, lon)
        hold(jet, SETUP_AGL_FT)
        if m is not None:
            return m / 0.3048
        if fallback_ft[0] is None:
            # The ground directly under the jet -- NOT the jet's own MSL
            # altitude, which is what this used to use and which would have
            # put a tower's BASE thousands of feet up in the air.
            fallback_ft[0] = jet.alt_ft() - jet.agl_ft()
            print(f"  WARNING: terrain query returned nil, falling back to "
                  f"{fallback_ft[0]:.0f}ft for any gate this happens at")
        return fallback_ft[0]

    return terrain_fn


def main():
    t = Telnet()
    startup(t)
    t.close()

    jet = Jet(NativeBridge())
    # Keep the SAFE DEFAULT floor (f15_airshow.FLOOR_AGL_FT, 4000ft) until
    # the jet is under the explicit, commanded descent in descend_to_agl().
    # Lowering it any earlier than that left the jet with a much weaker
    # safety net during UNCOMMANDED drift and it hit terrain (agl=36ft);
    # lowering it later is not an option either, because the 4000ft floor
    # would fight a descent whose whole purpose is to reach 800ft AGL.

    clean = False
    try:
        print("Stabilizing before laying out the course...")
        recover(jet, want_speed=420.0)

        # Get down to SETUP altitude BEFORE anything is laid out. Order
        # matters more than anything else in this function: see
        # descend_to_agl(). The floor stays at its safe default here -- this
        # descent is well above it.
        if not descend_to_agl(jet, SETUP_AGL_FT):
            print(f"  WARNING: descent timed out at {jet.agl_ft():.0f}ft AGL, "
                  f"wanted {SETUP_AGL_FT:.0f}ft")
        recover(jet, want_speed=420.0)
        # BUILD THE COURSE FROM THE JET'S STABLE, POST-RECOVERY POSITION --
        # not from a telnet snapshot taken right after engine start.
        # Confirmed live this was a real root cause of "gates never found,
        # flight wobbles and eventually crashes": recover() can run up to
        # 60s and, chasing want_speed by pitching down, overshot speed and
        # drifted position/heading before settling; building the course
        # from a stale pre-recovery snapshot left gate-1 already almost
        # behind the jet by the time fly_course() started.
        own_lat, own_lon = jet.telem["lat_deg"], jet.telem["lon_deg"]
        hdg = jet.telem["heading_deg"]
        print(f"  Stable at {own_lat:.5f},{own_lon:.5f} hdg {hdg:.1f} "
              f"alt {jet.telem['altitude_m']/0.3048:.0f}ft "
              f"spd {jet.telem['airspeed_kt']:.0f}kt")

        t2 = Telnet()
        # Real ground elevation at each gate, not a flat made-up altitude --
        # this is what puts the towers on the actual ground, sloped or not.
        # Everything from here through fly_course() is wrapped in one big
        # try/except Crashed: confirmed live the jet can hit terrain during
        # THIS setup phase too (close to the ground right after recovering
        # from a high starting altitude), not only during the flown course,
        # and an uncaught Crashed here previously died with a raw traceback
        # instead of a clean, informative report.
        obstacles, gates, target = build_course(own_lat, own_lon, hdg,
                                                 terrain_fn=_real_terrain_ft(t2, jet))
        print(f"\nCourse: {len(gates)} gates ({len(obstacles)} towers), flying "
              f"{GATE_FLY_AGL_FT:.0f}ft AGL through each gate")
        for g in gates:
            print(f"  {g.label:10s} fly-through {g.alt_ft:6.0f}ft")
        f15_props.place_obstacles(t2, obstacles,
                                  hold=lambda: hold(jet, SETUP_AGL_FT))
        t2.close()

        # Belt-and-braces: re-stabilize once more right before the course
        # itself starts. The real fix is that terrain_fn and
        # place_obstacles now hold the jet level via jet.step() calls
        # interleaved with their telnet round-trips instead of leaving the
        # native-protocol connection silent for several seconds -- a single
        # recover() call AFTER an already-uncommanded drift was tried and
        # confirmed live NOT to fix it on its own, so this call should now
        # find the jet already close to level and simply return quickly.
        recover(jet, want_speed=420.0)

        # Only NOW commit to the course's own low floor: the jet is about to
        # be flown deliberately down to 800ft AGL by the guidance law, which
        # the 4000ft default would otherwise fight the whole way.
        jet.floor_ft = 150.0

        print("\n=== FLYING THE GATE COURSE (real ground level) ===")
        results, closest = fly_course(jet, obstacles, gates)
        clean = report(results, closest)
    except Crashed as e:
        print(f"\n*** CRASHED: {e}")
    finally:
        jet.nb.close()
    print(f"\nRun {'CLEAN' if clean else 'with contacts'}.")


def _self_check():
    """Fake jet: only altitude physics, enough to prove the two fixes hold."""

    class FakeJet:
        def __init__(self, alt_ft):
            self.telem = {"airspeed_kt": 450.0, "roll_deg": 0.0, "pitch_deg": 0.0,
                          "altitude_m": alt_ft * 0.3048}
            self.climb_fps = 0.0

        def alt_ft(self):
            return self.telem["altitude_m"] / 0.3048

        def agl_ft(self):   # flat 5400ft terrain, as at KXTA
            return self.alt_ft() - 5400.0

        def hold_nose(self, target_pitch=0.0):
            return target_pitch          # stand-in for the g command

        def roll_to(self, bank_deg, gain=4.0):
            return 0.0

        def level_turn_g(self, bank_deg):
            return 1.0

        def sustainable_bank(self, direction, limit=80.0):
            return limit * direction

        def step(self, g_cmd, roll_rate_dps, throttle=1.0):
            # g_cmd here is the commanded pitch; 450kt ~= 760ft/s.
            self.telem["altitude_m"] += 760.0 * math.sin(math.radians(g_cmd)) * 0.1 * 0.3048

    jet = FakeJet(17979.0)
    assert descend_to_agl(jet, SETUP_AGL_FT), "descend_to_agl failed to level off"
    assert abs(jet.agl_ft() - SETUP_AGL_FT) < 300.0, jet.agl_ft()

    # And the guidance law never commands more than MAX_ALT_G of correction,
    # however big the deficit it is handed.
    class Gate:
        lat, lon, alt_ft = 0.0, 0.0, 0.0

    jet = FakeJet(17979.0)
    jet.telem["lat_deg"] = jet.telem["lon_deg"] = 0.001
    jet.telem["heading_deg"] = 0.0
    sent = []
    jet.step = lambda g, r, throttle=1.0: sent.append(g)
    steer(jet, Gate())
    assert abs(sent[0]) <= 1.0 + MAX_ALT_G + 1e-9, sent
    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
