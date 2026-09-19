"""F-15 airshow: a 23-state choreography, one small controller per maneuver.

Architecture (matches the state machine it implements):

    Show          ordered list of states, runs one at a time
    Maneuver      name, controller, time cap, entry gate
    controller    fn(jet, m) -> True when its completion condition is met
    Jet           commands the aircraft in ITS OWN units, and integrates
                  attitude so maneuvers can say "through 180deg of heading"
                  or "through 360deg of pitch" rather than guessing at time

Everything commands the two things the F-15's CAS actually accepts (see
SESSION.md session 5, Systems/f-15-cas.xml):

    PITCH is a G-COMMAND axis     elevator -> commanded load factor Nz
    ROLL  is a RATE-COMMAND axis  aileron  -> roll rate, +-213.1 deg/s

so a maneuver reads the way a pilot briefs it -- "80 degrees of bank, pull
until the nose comes through 180" -- instead of being a PID tune on a
surface deflection.

Two physical limits are enforced everywhere, because violating either is
what turned every earlier attempt into a departure rather than a maneuver:

    available g   = (V / Vstall)^2      stall speed scales as sqrt(n)
    level turn g  = 1 / cos(bank)       lift must still carry the weight

Honest notes on three of the requested states -- see COBRA, TUMBLE and
TAILSLIDE below. The F-15 has no thrust vectoring and its CAS actively
limits AoA, so those are flown as the closest thing the airframe will
actually do, not as the Su-27 maneuvers they're named after.

Run:
    ./scripts/with_sim.sh scripts/f15_airshow.py --list
    ./scripts/with_sim.sh scripts/f15_airshow.py roll-left uturn-right
    FG_START=ground ./scripts/with_sim.sh scripts/f15_airshow.py --show
"""

import math
import sys
import time

from f15_intercept import (
    CAS_ROLL_RATE_FULL_DPS,
    STALL_SPEED_KT,
    Telnet,
    angle_diff_deg,
    g_to_elevator,
    startup,
)
from fly_brain.native_protocol import NativeBridge

STEP_HZ = 20
G_LIMIT_POS = 9.0  # F-15 symmetric structural limit
G_LIMIT_NEG = -3.0  # the CAS g-demand table bottoms out at -3.99
ROLL_RATE_LIMIT_DPS = CAS_ROLL_RATE_FULL_DPS  # use all of it -- that is the point
PITCH_RECOVER_G_PER_DEG = 0.05  # g of push/pull per degree of pitch error
ENTRY_SPEED_KT = 400.0  # maneuvers are entered from here, not from whatever
# speed the last one left behind. Confirmed live why this matters: a break
# turn entered at 276kt (bled off by a preceding 7g pull-up) made only 10deg
# of heading in 30s. Only (V/Vs)^2 of g exists, so at 276kt that is 2.35g --
# and an 80deg bank needs 5.76g. The jet was not turning, it was mushing.
M_PER_FT = 0.3048

# Hard deck. Real display flying has one, and this routine needs it more than
# most: Split-S, tailslide, tumble and rolling-descent all dump thousands of
# feet by design, over terrain that sits at ~5400ft here. Without it the jet
# flew into the ground mid-show ("Aircraft hit ShrubCover. and exploded." in
# FlightGear's own log) and -- because nothing was watching -- the script kept
# dutifully flying a destroyed aircraft until its wall-clock timeout.
FLOOR_AGL_FT = 4000.0  # below this, recovery overrides the maneuver
CRASH_AGL_FT = 60.0


class Crashed(Exception):
    """The aircraft is on the ground and not in a landing maneuver."""



class Jet:
    """Commands the aircraft in load factor and roll rate, and integrates
    attitude change so maneuvers can use real completion conditions."""

    def __init__(self, bridge):
        self.nb = bridge
        self.telem = bridge.get_telemetry()
        # Accumulated attitude change. These are integrated per step rather
        # than compared against a start value, because a wrapped comparison
        # (abs(angle_diff(now, start))) peaks at 180 and then DECREASES --
        # confirmed live, a break turn sailed straight through its own
        # "turned >= 178" window between two slow steps and kept going to
        # 223deg. Integrating cannot skip a window and handles >360 too.
        self.turned_deg = 0.0
        self.rolled_deg = 0.0
        self.pitched_deg = 0.0
        # Euler pitch spans only -90..+90 and never wraps, so integrating it
        # CANNOT measure a loop: confirmed live, a loop's "pitched >= 350"
        # condition was unreachable, the jet just zoomed to 43000ft and
        # tumbled at 134deg AoA. Rotation through a loop is the integral of
        # BODY PITCH RATE, which is what this accumulates.
        self.looped_deg = 0.0
        self.climb_fps = 0.0  # filtered, for damping altitude controllers
        self._last_t = time.monotonic()
        self.gear_down = False
        self.floor_ft = FLOOR_AGL_FT
        self.allow_ground = False  # landing sets this
        self.floor_breached = False

    def zero_integrators(self):
        self.turned_deg = self.rolled_deg = self.pitched_deg = 0.0
        self.looped_deg = 0.0

    def step(self, g_cmd, roll_rate_dps, throttle=1.0, brakes=0.0,
             flaps=0.0, speedbrake=0.0, rudder=0.0):
        prev = self.telem
        t = self.nb.get_telemetry()
        self.telem = t

        self.turned_deg += angle_diff_deg(t["heading_deg"], prev["heading_deg"])
        self.rolled_deg += angle_diff_deg(t["roll_deg"], prev["roll_deg"])
        self.pitched_deg += angle_diff_deg(t["pitch_deg"], prev["pitch_deg"])
        now = time.monotonic()
        dt = now - self._last_t
        self.looped_deg += t["pitch_rate_dps"] * dt
        if dt > 0:
            raw = (t["altitude_m"] - prev["altitude_m"]) / M_PER_FT / dt
            self.climb_fps += 0.3 * (raw - self.climb_fps)  # low-pass
        self._last_t = now

        if t["agl_m"] / M_PER_FT < CRASH_AGL_FT and not self.allow_ground:
            raise Crashed(
                f"ground contact: agl={t['agl_m']/M_PER_FT:.0f}ft "
                f"spd={t['airspeed_kt']:.0f}kt pitch={t['pitch_deg']:+.0f}"
            )

        # Ground collision avoidance overrides whatever the maneuver wanted.
        # Roll upright first, THEN pull: pulling while inverted or steeply
        # banked points the lift vector further into the ground, which is the
        # classic way a recovery becomes the impact.
        if t["agl_m"] / M_PER_FT < self.recovery_floor_ft() and not self.allow_ground:
            self.floor_breached = True
            if abs(t["roll_deg"]) > 30.0:
                # Roll upright at the full 213 deg/s, unloaded. Every tenth
                # of a second spent here is altitude, so do not trickle it
                # through a proportional gain.
                g_cmd = 1.0
                roll_rate_dps = -math.copysign(ROLL_RATE_LIMIT_DPS, t["roll_deg"])
            else:
                g_cmd = min(0.8 * self.g_available(), 5.0)
                roll_rate_dps = self.roll_to(0.0)
            throttle = 1.0

        g = max(G_LIMIT_NEG, min(min(G_LIMIT_POS, self.g_available()), g_cmd))
        rate = max(-ROLL_RATE_LIMIT_DPS, min(ROLL_RATE_LIMIT_DPS, roll_rate_dps))

        self.nb.send_controls(
            aileron=max(-1.0, min(1.0, rate / CAS_ROLL_RATE_FULL_DPS)),
            elevator=g_to_elevator(g),
            rudder=max(-1.0, min(1.0, rudder)),
            throttle=throttle,
            gear_down=self.gear_down,
            brakes=brakes,
            flaps=flaps,
            speedbrake=speedbrake,
        )
        time.sleep(1.0 / STEP_HZ)
        return t

    def recovery_floor_ft(self):
        """Altitude actually needed to pull out of the CURRENT dive.

        A fixed deck is not enough: confirmed live, the jet passed 4000ft AGL
        at 673kt in a 42deg dive and hit the ground anyway, because a 5g
        pull-out at that speed needs a radius of v^2/((n-1)g) ~= 8000ft. The
        deck has to scale with energy and dive angle, not be a constant.
        """
        t = self.telem
        v = t["airspeed_kt"] * 0.514444  # m/s
        n = max(1.5, min(5.0, 0.8 * self.g_available()))
        gamma = math.radians(max(0.0, -t["pitch_deg"]))

        # Phase 1: roll upright. You cannot pull your way out while inverted
        # -- pulling points the nose further DOWN -- so the recovery has to
        # roll to wings-level first, descending the whole time at 1g. This
        # term was missing and is why a guard that engaged at a geometrically
        # "sufficient" altitude still hit the ground: the pull started ~0.9s
        # and several hundred feet later than the maths assumed.
        roll_s = abs(t["roll_deg"]) / ROLL_RATE_LIMIT_DPS
        lost_rolling_m = v * math.sin(gamma) * roll_s

        # Phase 2: the pull-out arc itself.
        radius_m = v * v / ((n - 1.0) * 9.81)
        lost_pulling_m = radius_m * (1.0 - math.cos(gamma))

        need_ft = (lost_rolling_m + lost_pulling_m) / M_PER_FT
        return max(self.floor_ft, need_ft * 1.5 + 500.0)

    # --- physics helpers -------------------------------------------------
    def g_available(self):
        """Max attainable load factor: stall speed scales as sqrt(n)."""
        return (self.telem["airspeed_kt"] / STALL_SPEED_KT) ** 2

    @staticmethod
    def level_turn_g(bank_deg):
        """n = 1/cos(phi): lift's vertical component must still carry weight."""
        return 1.0 / max(0.15, math.cos(math.radians(bank_deg)))

    def sustainable_bank(self, direction, limit=80.0):
        """Steepest bank the current energy state can actually hold."""
        have = 0.85 * self.g_available()
        if have <= 1.05:
            return 20.0 * direction
        return min(limit, math.degrees(math.acos(1.0 / have))) * direction

    def roll_to(self, bank_deg, gain=4.0):
        """Roll rate command that snaps to a bank angle and holds it."""
        err = angle_diff_deg(bank_deg, self.telem["roll_deg"])
        return max(-ROLL_RATE_LIMIT_DPS, min(ROLL_RATE_LIMIT_DPS, gain * err))

    def hold_nose(self, target_pitch=0.0):
        """g that drives pitch toward a target, on top of bank compensation."""
        t = self.telem
        base = self.level_turn_g(t["roll_deg"]) if abs(t["roll_deg"]) < 85 else 1.0
        return base + PITCH_RECOVER_G_PER_DEG * (target_pitch - t["pitch_deg"])

    def alt_ft(self):
        return self.telem["altitude_m"] / M_PER_FT

    def agl_ft(self):
        return self.telem["agl_m"] / M_PER_FT


# --- controllers ---------------------------------------------------------
# Each takes (jet, m) where m is the Maneuver (for elapsed/state) and
# returns True once its completion condition is met.

def takeoff(jet, m):
    """Roll, rotate at Vr, positive climb, gear up, clean to 1500ft AGL."""
    jet.allow_ground = True  # on the runway by definition
    jet.gear_down = True
    t = jet.telem
    v = t["airspeed_kt"]
    if v < 160.0 and jet.agl_ft() < 20:
        jet.step(1.0, jet.roll_to(0.0), throttle=1.0)  # accelerate, nose down
        return False
    if jet.agl_ft() < 50:
        jet.step(1.35, jet.roll_to(0.0), throttle=1.0)  # rotate
        return False
    jet.gear_down = False  # positive rate -- gear up
    jet.step(jet.hold_nose(12.0), jet.roll_to(0.0), throttle=1.0)
    if jet.agl_ft() > 1500.0:
        jet.allow_ground = False  # hard deck applies from here on
        return True
    return False


def afterburner_accel(jet, m):
    """Full burner, level, until the jet is genuinely fast.

    Systems/f-15-engines.xml: augmentation-stage = int(1+(throttle-1)*5),
    so the burner only lights at throttle exactly 1.0 -- 0.95 is dry thrust.
    """
    jet.step(jet.hold_nose(0.0), jet.roll_to(0.0), throttle=1.0)
    return jet.telem["airspeed_kt"] > 550.0


def _pass(bank, secs, throttle=1.0, target_pitch=0.0):
    def fn(jet, m):
        jet.step(jet.hold_nose(target_pitch), jet.roll_to(bank), throttle=throttle)
        return m.elapsed() >= secs
    return fn


def high_speed_pass(jet, m):
    jet.step(jet.hold_nose(0.0), jet.roll_to(0.0), throttle=1.0)
    return m.elapsed() >= 8.0


def _break_turn(direction, degrees=180.0):
    """Max-performance level turn. At 80deg bank a level turn needs 5.8g,
    giving ~14 deg/s at 450kt -- the nose comes through 180 in ~13s."""
    def fn(jet, m):
        bank = jet.sustainable_bank(direction)
        # Only pull once the lift vector points into the turn: pulling before
        # that just climbs, and pulling nose-high goes vertical (seen live).
        rolled_in = abs(angle_diff_deg(bank, jet.telem["roll_deg"])) < 15.0
        g = jet.level_turn_g(bank) if rolled_in else 1.0
        g += PITCH_RECOVER_G_PER_DEG * (0.0 - jet.telem["pitch_deg"])
        jet.step(g, jet.roll_to(bank), throttle=1.0)
        return abs(jet.turned_deg) >= degrees
    return fn


def vertical_climb(jet, m):
    """Pull to ~70deg nose-high and hold it until energy runs out."""
    if jet.telem["pitch_deg"] < 65.0 and abs(jet.looped_deg) < 120:
        jet.step(5.0, jet.roll_to(0.0), throttle=1.0)
        return False
    jet.step(jet.hold_nose(70.0), jet.roll_to(0.0), throttle=1.0)
    return jet.telem["airspeed_kt"] < 180.0


def _aileron_roll(direction, turns=1.0):
    """Full-rate roll. aileron=+-1.0 is 213.1 deg/s: ~1.7s per revolution.
    Held near 1g -- pulling while inverted points you at the ground."""
    def fn(jet, m):
        jet.step(1.0, ROLL_RATE_LIMIT_DPS * direction, throttle=1.0)
        return abs(jet.rolled_deg) >= 355.0 * turns
    return fn


def split_s(jet, m):
    """Roll inverted, then pull through the back half of a loop.
    Exits 180deg of heading reversed and several thousand feet lower."""
    if abs(abs(jet.telem["roll_deg"]) - 180.0) > 20.0 and abs(jet.looped_deg) < 30:
        jet.step(1.0, ROLL_RATE_LIMIT_DPS, throttle=0.7)  # roll to inverted
        return False
    jet.step(5.0, 0.0, throttle=0.7)  # pull through the back half of a loop
    return abs(jet.looped_deg) >= 170.0


def knife_edge_pass(jet, m):
    """90deg bank held with rudder carrying the fuselage.

    A real knife-edge is sustained on fuselage lift from rudder input. The
    F-15 cannot hold it indefinitely, so this is a pass, not level flight:
    it will descend, and that is correct behaviour rather than a bug.
    """
    jet.step(1.0, jet.roll_to(88.0), throttle=1.0, rudder=0.4)
    return m.elapsed() >= 6.0


def high_alpha_pass(jet, m):
    """Slow, nose-high, high angle of attack -- the F-15's real 'show' pass.
    Throttled back so it decelerates, holding altitude with attitude."""
    jet.step(jet.hold_nose(15.0), jet.roll_to(0.0), throttle=0.85)
    return jet.telem["airspeed_kt"] < 200.0 or m.elapsed() >= 20.0


def cobra(jet, m):
    """NOT a Pugachev's Cobra -- the F-15 cannot do one.

    A true cobra needs relaxed static stability plus thrust vectoring to
    swing past 90deg AoA and recover. The F-15 has neither, and its CAS
    actively limits AoA. This flies the closest honest equivalent: a rapid
    max-g pitch-up to a very high nose attitude, held briefly, then pushed
    straight back to level. Named as requested; described accurately here.
    """
    if abs(jet.looped_deg) < 60.0 and m.elapsed() < 4.0:
        jet.step(G_LIMIT_POS, jet.roll_to(0.0), throttle=0.6)
        return False
    jet.step(jet.hold_nose(0.0), jet.roll_to(0.0), throttle=1.0)
    return abs(jet.telem["pitch_deg"]) < 8.0


def acceleration(jet, m):
    jet.step(jet.hold_nose(0.0), jet.roll_to(0.0), throttle=1.0)
    return jet.telem["airspeed_kt"] > 600.0 or m.elapsed() >= 25.0


def vertical_tailslide(jet, m):
    """Vertical until the airspeed decays to nothing, then let it fall.

    Whether this looks like a real tailslide depends on how far JSBSim's
    aero tables extend past the stall -- outside the fitted envelope the
    result is whatever the extrapolation does, not validated aerodynamics.
    """
    if jet.telem["pitch_deg"] < 80.0 and jet.telem["airspeed_kt"] > 150.0:
        jet.step(4.0, jet.roll_to(0.0), throttle=1.0)
        return False
    jet.step(1.0, 0.0, throttle=0.3)  # hang, then slide
    return jet.telem["pitch_deg"] < 0.0


def tumble(jet, m):
    """Deliberate departure from controlled flight, then recover.

    A real tumble is post-stall gyration -- by definition NOT a commanded
    maneuver. This drives the jet out of its envelope (low speed, full
    pitch and roll command), lets it depart, then flies a standard
    recovery: controls neutral, unload to ~1g, let the nose fall and the
    speed build back.
    """
    if m.elapsed() < 6.0 and jet.telem["airspeed_kt"] > 140.0:
        jet.step(G_LIMIT_POS, ROLL_RATE_LIMIT_DPS, throttle=0.4)
        return False
    jet.step(1.0, 0.0, throttle=1.0)  # neutral + unload = the real recovery
    return jet.telem["airspeed_kt"] > 300.0 and abs(jet.telem["roll_deg"]) < 30.0


def inverted_pass(jet, m):
    """Roll inverted and hold it. Inverted level flight needs NEGATIVE g:
    upside down, lift points at the ground, so the jet must push to -1g."""
    if abs(abs(jet.telem["roll_deg"]) - 180.0) > 25.0 and m.elapsed() < 4.0:
        jet.step(1.0, ROLL_RATE_LIMIT_DPS, throttle=1.0)
        return False
    jet.step(-1.0, jet.roll_to(180.0), throttle=1.0)
    return m.elapsed() >= 10.0


def loop(jet, m):
    """Continuous pull through 360deg of pitch."""
    jet.step(5.0, jet.roll_to(0.0, gain=1.5), throttle=1.0)
    return abs(jet.looped_deg) >= 350.0


def barrel_roll(jet, m):
    """A roll and a loop at once: sustained pull while rolling right through,
    so the jet corkscrews around a straight line instead of spinning on it."""
    jet.step(2.5, ROLL_RATE_LIMIT_DPS * 0.35, throttle=1.0)
    return abs(jet.rolled_deg) >= 355.0


def crossing_pass(jet, m):
    jet.step(jet.hold_nose(0.0), jet.roll_to(0.0), throttle=1.0)
    return m.elapsed() >= 6.0


def max_power_climb(jet, m):
    jet.step(jet.hold_nose(35.0), jet.roll_to(0.0), throttle=1.0)
    return jet.alt_ft() > 25000.0 or jet.telem["airspeed_kt"] < 250.0


def rolling_descent(jet, m):
    """Descend while rolling continuously -- a descending corkscrew."""
    jet.step(0.4, ROLL_RATE_LIMIT_DPS * 0.5, throttle=0.4, speedbrake=0.5)
    return jet.alt_ft() < 12000.0 or m.elapsed() >= 25.0


def final_burner_pass(jet, m):
    jet.step(jet.hold_nose(0.0), jet.roll_to(0.0), throttle=1.0)
    return m.elapsed() >= 8.0


def break_to_downwind(jet, m):
    """The pitch-out: hard turn plus speedbrake to wash off energy."""
    bank = jet.sustainable_bank(1, limit=70.0)
    jet.step(jet.level_turn_g(bank), jet.roll_to(bank), throttle=0.3, speedbrake=1.0)
    return abs(jet.turned_deg) >= 180.0 or jet.telem["airspeed_kt"] < 250.0


def landing(jet, m):
    """Gear down, descend on speed, flare, hold it off, brake."""
    jet.gear_down = True
    jet.allow_ground = True  # the whole point is to reach the ground
    agl = jet.agl_ft()
    if agl > 50.0:
        # Descend at ~1 deg nose-low per 1000ft to go, decelerating.
        jet.step(jet.hold_nose(-3.0), jet.roll_to(0.0),
                 throttle=0.25, flaps=1.0, speedbrake=0.3)
        return False
    if agl > 5.0:
        jet.step(jet.hold_nose(6.0), jet.roll_to(0.0), throttle=0.0, flaps=1.0)
        return False
    jet.step(1.0, jet.roll_to(0.0), throttle=0.0, flaps=1.0,
             speedbrake=1.0, brakes=1.0)
    return jet.telem["airspeed_kt"] < 40.0


# --- the show ------------------------------------------------------------

class Maneuver:
    def __init__(self, name, desc, fn, cap, needs_entry=True,
                 entry_speed=ENTRY_SPEED_KT):
        self.name, self.desc, self.fn, self.cap = name, desc, fn, cap
        self.needs_entry = needs_entry  # recover to level/on-speed first
        # Energy maneuvers need a faster entry than a pass does. A loop must
        # climb its own diameter -- radius = v^2/((n-1)g), so ~9000ft at 5g
        # and 450kt -- and pay for it out of kinetic energy. Entered at 400kt
        # and 22000ft it simply cannot: confirmed live, the jet zoomed
        # 22000->32855ft, decayed to 77kt and fell out before 30deg of arc.
        self.entry_speed = entry_speed
        self.start = 0.0

    def elapsed(self):
        return time.time() - self.start


def M(*a, **k):
    return Maneuver(*a, **k)


SHOW = [
    M("takeoff", "Roll, rotate, gear up, climb out", takeoff, 90, needs_entry=False),
    M("afterburner-accel", "Full burner acceleration", afterburner_accel, 60),
    M("high-speed-pass", "High speed level pass", high_speed_pass, 20),
    M("break-turn", "Max-performance 180 break turn", _break_turn(1), 45),
    M("vertical-climb", "Near-vertical zoom climb", vertical_climb, 40, entry_speed=550.0),
    M("aileron-roll", "360 aileron roll, left", _aileron_roll(-1), 15),
    M("split-s", "Split-S: inverted, pull through", split_s, 40, entry_speed=450.0),
    M("belly-roll", "360 belly roll, right", _aileron_roll(1), 15),
    M("knife-edge-pass", "90deg knife-edge pass", knife_edge_pass, 20),
    M("high-alpha-pass", "Slow high-AoA pass", high_alpha_pass, 35),
    M("cobra", "High-alpha pitch-up (see docstring)", cobra, 25),
    M("acceleration", "Burner acceleration run", acceleration, 40),
    M("vertical-tailslide", "Vertical to zero airspeed", vertical_tailslide, 45,
      entry_speed=550.0),
    M("tumble", "Departure and recovery", tumble, 45),
    M("inverted-pass", "Inverted level pass at -1g", inverted_pass, 30),
    M("loop", "360 loop", loop, 45, entry_speed=550.0),
    M("barrel-roll", "Barrel roll", barrel_roll, 30),
    M("crossing-pass", "Level crossing pass", crossing_pass, 20),
    M("max-power-climb", "Max power climb", max_power_climb, 60),
    M("rolling-descent", "Rolling descent", rolling_descent, 40),
    M("final-burner-pass", "Final burner pass", final_burner_pass, 20),
    M("break", "Pitch-out break to downwind", break_to_downwind, 40),
    M("landing", "Gear down, approach, flare, rollout", landing, 120),
]
BY_NAME = {m.name: m for m in SHOW}


def recover(jet, max_secs=60.0, want_speed=ENTRY_SPEED_KT):
    """Return to level flight, on speed, before the next maneuver.

    An earlier version only rolled the wings level at ~1g, which does NOT
    bring the nose back: confirmed live, a break turn inherited +68deg of
    pitch from a preceding 7g pull-up, entered nose-high and departed
    vertically instead of turning. Recovery has to fly the nose to the
    horizon AND rebuild energy, because every maneuver here assumes it
    starts from level flight at a speed that can actually carry its g.
    """
    end = time.time() + max_secs
    while time.time() < end:
        t = jet.telem
        if (abs(t["pitch_deg"]) < 4.0 and abs(t["roll_deg"]) < 5.0
                and t["airspeed_kt"] >= want_speed):
            return True
        # Roll upright FIRST: pulling toward the horizon while inverted
        # drives the nose further down, not up.
        upright = abs(t["roll_deg"]) < 25.0
        # If the nose is level but the jet is slow, sitting at 1g will never
        # build speed -- descend for it. Energy is altitude OR airspeed; the
        # only way to buy one when out of the other is to spend the other.
        target_pitch = 0.0
        if upright and abs(t["pitch_deg"]) < 10.0 and t["airspeed_kt"] < want_speed:
            target_pitch = -8.0
        g = jet.hold_nose(target_pitch) if upright else 1.0
        jet.step(max(0.0, min(5.0, g)), jet.roll_to(0.0), throttle=1.0)
    return False


def fly(jet, m, entry=True):
    print(f"\n[{m.name}] {m.desc}")
    if m.needs_entry and entry:
        if not recover(jet, want_speed=m.entry_speed):
            print("    (entry gate not met -- flying it anyway from current state)")
    jet.zero_integrators()
    jet.floor_breached = False
    m.start = time.time()
    t0 = dict(jet.telem)
    done = False
    while m.elapsed() < m.cap:
        if m.fn(jet, m) is True:
            done = True
            break
    t = jet.telem
    print(
        f"    {'OK ' if done else 'CAP'} {m.elapsed():5.1f}s  "
        f"hdg {t0['heading_deg']:5.1f}->{t['heading_deg']:5.1f} "
        f"(turned {jet.turned_deg:+6.0f})  "
        f"roll {t['roll_deg']:+6.1f} (rolled {jet.rolled_deg:+6.0f})  "
        f"pitch {t['pitch_deg']:+6.1f} (looped {jet.looped_deg:+6.0f})  "
        f"alt {t['altitude_m']/M_PER_FT:6.0f}ft  spd {t['airspeed_kt']:3.0f}kt  "
        f"aoa {t['alpha_deg']:+5.1f}"
        + ("  [FLOOR]" if jet.floor_breached else "")
    )
    return done


def main():
    args = sys.argv[1:]
    if "--list" in args:
        for m in SHOW:
            print(f"  {m.name:20s} {m.desc}")
        return
    names = [a for a in args if not a.startswith("-")]
    for n in names:
        if n not in BY_NAME:
            raise SystemExit(f"unknown maneuver {n!r}; try --list")
    program = [BY_NAME[n] for n in names] if names else SHOW

    t = Telnet()
    startup(t)
    t.close()

    jet = Jet(NativeBridge())
    print(f"\n=== F-15 AIRSHOW: {len(program)} states ===")
    results = []
    try:
        for m in program:
            results.append((m.name, fly(jet, m)))
    except Crashed as e:
        # Do not keep flying a wreck. Previously nothing watched for this and
        # a destroyed aircraft was "flown" until the wall-clock timeout, with
        # the log simply going silent -- which reads like a hang, not a crash.
        print(f"\n*** CRASHED during [{m.name}]: {e}")
        print("*** Aborting show.")
        results.append((m.name, False))
    finally:
        jet.nb.close()

    ok = sum(1 for _, d in results if d)
    print(f"\n=== {ok}/{len(results)} states met their completion condition ===")
    for name, d in results:
        if not d:
            print(f"    hit time cap: {name}")


def _self_check():
    assert abs(Jet.level_turn_g(0.0) - 1.0) < 1e-9
    assert abs(Jet.level_turn_g(80.0) - 5.759) < 0.01  # the break turn's 5.8g
    assert Jet.level_turn_g(60.0) > Jet.level_turn_g(45.0)  # monotonic in bank
    assert abs(ROLL_RATE_LIMIT_DPS / CAS_ROLL_RATE_FULL_DPS - 1.0) < 1e-9
    # CAS sign convention: pulling g is negative elevator, pushing is positive.
    assert g_to_elevator(7.0) < 0 and g_to_elevator(-1.0) > 0
    assert 360.0 / ROLL_RATE_LIMIT_DPS < 2.0  # full roll under 2s
    # Every state must be uniquely named and have a positive time cap.
    assert len(BY_NAME) == len(SHOW) == 23, f"{len(SHOW)} states"
    assert all(m.cap > 0 and callable(m.fn) for m in SHOW)
    # Integrating heading must survive wrap -- the bug that overshot a turn
    # to 223deg was a wrapped comparison peaking at 180 and falling back.
    total = 0.0
    for a, b in [(350, 10), (10, 350), (179, -179)]:
        total += angle_diff_deg(b, a)
    assert abs(total - (20 - 20 + 2)) < 1e-6, total


if __name__ == "__main__":
    _self_check()
    main()
