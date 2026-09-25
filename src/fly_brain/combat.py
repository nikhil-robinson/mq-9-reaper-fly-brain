"""Combat policy for the F-15 dogfight: pure decisions, no FlightGear.

Deliberately importable and testable with no sim running. The arena
(scripts/dogfight.py) owns every socket and property write; this module only
answers "where do I point, how fast do I want to be, and may I shoot".

Output convention matches the calibrated decode layer the project already has,
and the one the connectome must eventually drive: a g-demand and a roll-rate
demand, never raw surface deflections.
"""
import math
from collections import namedtuple

from fly_brain.geo import angle_diff_deg, bearing_deg, haversine_m, offset_point

KT_TO_MS = 0.5144

# AIM-9 envelope, from the F-15 add-on's own fox2.nas parameters.
AIM9_MIN_NM = 0.2       # below this the warhead has no time to arm
AIM9_MAX_NM = 9.0
AIM9_FOV_DEG = 27.5     # seeker field of view
AIM9_MIN_TGT_KT = 15.0  # fox2.nas refuses a class-A target slower than this
# Gun: M61A1. Short range, and it only hits with the pipper genuinely on.
GUN_MAX_NM = 0.6
GUN_FOV_DEG = 2.5
# Max instantaneous turn rate is at corner speed: slower and there is not
# enough lift to pull g, faster and the turn radius blows up.
# The Bombable AI spawns and typically cruises around 400kt (see
# data/bombable_dogfight.xml). 430kt left almost no speed margin to actually
# close on or out-turn an opponent already near that speed, especially once
# our own jet dips into RESET/RECOVER during real maneuvering (which costs
# energy the same way it would for the AI). Raised for real speed
# superiority, not just parity.
CORNER_SPEED_KT = 500.0

STALL_SPEED_KT = 180.0          # 1g stall speed; n_max = (V/Vs)^2
CAS_ROLL_RATE_FULL_DPS = 213.1  # aileron=1.0 commands this via the roll CAS
MAX_G = 9.0
ENGAGE_RANGE_NM = 5.0
K_BANK_TO_RATE = 2.5            # bank error -> commanded roll rate
MAX_BANK_DEG = 80.0
HARD_TURN_FRACTION = 0.9        # fraction of available g to pull while
                                 # actively turning to engage -- a fighter
                                 # pulls hard, not the bare minimum to hold
                                 # altitude
NEAR_180_DEADBAND_DEG = 20.0     # within this of +-180, keep the committed
                                 # turn direction rather than re-deriving sign

# Altitude terms, carried over from the intercept law where they were proven.
# A dogfight still has to not fly into the ground: confirmed live that
# commanding only 1/cos(bank) -- correct for a coordinated level turn and
# nothing else -- walked both jets from 17900ft down to 8400ft in 45s, on
# terrain that sits at ~5400ft.
K_ALT_G = 1.0 / 4000.0   # g of extra pull per ft of altitude error
K_CLIMB_G = 0.02         # g per ft/s of climb rate (damping)
# Ground-avoidance floor, in AGL. A FIXED deck is not enough -- the airshow
# work established this the hard way: the jet passed 4000ft AGL at 673kt in a
# 42deg dive and hit the ground anyway, because a 5g pull-out at that speed
# needs a radius of v^2/((n-1)g) ~= 8000ft. The floor has to scale with energy
# and dive angle. FLOOR_MIN_FT is only the absolute lower bound.
FLOOR_MIN_FT = 1500.0
ROLL_RATE_LIMIT_DPS = 213.1
M_PER_FT = 0.3048
# Fraction of available lift spent on the turn itself, leaving the rest for
# holding altitude. Without this the pilot commands a bank it cannot sustain.
TURN_LIFT_BUDGET = 0.8
# Energy floor. Below this the jet stops turning and gets its speed back.
# Confirmed live: holding a 63-74deg bank at 14500ft decayed 418kt -> 296kt
# even at full throttle, and available g fell 4.18 -> 1.90 with it, until the
# bank being held needed more lift than existed. A turning fight is an energy
# fight, and a controller that only ever points at the enemy loses it.
# Genuinely near-stall only, and CAPPED in duration -- see step()'s
# RESET handling for why. A real fighter turning to engage a single,
# comparably-slow opponent (this Bombable AI cruises around 300-400kt)
# essentially never needs a deliberate multi-second disengage-and-extend;
# it only needs to not literally stall. Multiple live runs with a wider
# floor/recover band (330/400, then 300/360) got stuck in RESET for 30-90+
# real seconds at a stretch, either slow-circling or (once that was fixed)
# ballooning on leftover vertical momentum while decelerating, and never
# closing back in on the target during that whole window.
# Confirmed live: 200/230kt let the jet cruise an ENTIRE fight at 230-260kt
# average (never above 421kt across a ~300-sample run) without ever
# triggering a real extend, because it was always above this floor -- far
# below CORNER_SPEED_KT=430, and far less capable of hard maneuvering than
# it should be. The floor has to be low enough to not fight the RESET_STEPS
# cap (which bounds how long an extend can run once triggered, fixing the
# EARLIER bug of extending forever), but high enough to actually engage when
# speed sags meaningfully below corner speed rather than just above stall.
ENERGY_FLOOR_KT = 320.0
ENERGY_RECOVER_KT = 400.0   # hysteresis: recover to here before fighting again
RESET_MAX_STEPS = 40        # hard cap: ~2s of sim time at 20Hz, however fast
                            # the arena's own loop actually runs -- RESET
                            # commits to a short, decisive unload, not an
                            # open-ended extend the geometry can wander during


MAX_LEAD_TIME_SEC = 20.0


def lead_bearing_deg(own_lat, own_lon, own_speed_kt,
                     tgt_lat, tgt_lon, tgt_heading_deg, tgt_speed_kt):
    """Bearing to where the target WILL be, not where it is.

    Pure pursuit (pointing at the target's current position) is right for a
    fixed waypoint and wrong for a dogfight: against a maneuvering jet it
    degenerates into a lag chase that never closes. Estimate time-to-go from
    present range and own speed, advance the target along its own velocity
    vector, and aim there. One iteration is enough at dogfight ranges.

    t_go = range/speed BLOWS UP at long range with near-parallel headings
    (near-zero closure): confirmed live at 14nm with the target on a similar
    heading, t_go came out near 140s, extrapolating the target ~26km along its
    track -- an offset_point() displacement of the same order as the range
    itself, which walks the "lead point" onto roughly the opposite side of the
    local great circle from the real target. The result was a bearing ~275deg
    off the true one, which CombatPilot then chased with full authority,
    producing a violent, divergent roll oscillation (observed: bank swinging
    +179 -> -178 -> +167 deg over 4 seconds) that nearly flew into terrain.
    Real lead pursuit is only meaningful over the few seconds it actually
    takes a shot to arrive, not over a straight-line "closure" projection at
    long stand-off range -- so t_go is capped at a sane maximum, well inside
    which the small-angle extrapolation this formula assumes remains valid.
    """
    rng_m = haversine_m(own_lat, own_lon, tgt_lat, tgt_lon)
    own_ms = max(1.0, own_speed_kt * KT_TO_MS)
    t_go = min(MAX_LEAD_TIME_SEC, rng_m / own_ms)
    lead_lat, lead_lon = offset_point(tgt_lat, tgt_lon, tgt_heading_deg,
                                      tgt_speed_kt * KT_TO_MS * t_go)
    return bearing_deg(own_lat, own_lon, lead_lat, lead_lon)


def corner_throttle(airspeed_kt, corner_kt=CORNER_SPEED_KT):
    """Throttle that drives airspeed toward corner speed.

    Proportional, clamped to a valid throttle. Being fast is not free in a
    turning fight: above corner the radius grows and the fight is lost by
    geometry rather than by energy.
    """
    err = corner_kt - airspeed_kt
    return max(0.0, min(1.0, 0.55 + err / 300.0))


def aim9_in_envelope(range_nm, bearing_err_deg, target_speed_kt, has_lock):
    """Whether firing an AIM-9 now is legitimate.

    Without explicit discipline an AI empties its rails at zero range and
    nothing ever hits. Every one of these is a real launch precondition in
    weapons.nas/fox2.nas, not a heuristic.
    """
    return (has_lock
            and AIM9_MIN_NM <= range_nm <= AIM9_MAX_NM
            and abs(bearing_err_deg) <= AIM9_FOV_DEG
            and target_speed_kt >= AIM9_MIN_TGT_KT)


def gun_in_envelope(range_nm, bearing_err_deg):
    """Whether the gun solution is real -- for the endgame inside the AIM-9's
    minimum arming range, where a missile is useless."""
    return range_nm <= GUN_MAX_NM and abs(bearing_err_deg) <= GUN_FOV_DEG


def is_lethal(miss_distance_m, lethal_radius_m=12.0):
    """Did a detonation at this miss distance kill the target.

    FlightGear reports the hit; it has no damage model, so the consequence is
    ours. 12m is an AIM-9-scale fragmentation radius.
    """
    return miss_distance_m <= lethal_radius_m


Command = namedtuple("Command", "g_cmd roll_rate_dps throttle "
                                "fire_missile fire_gun flares state")


def alt_err_g(pilot, telem):
    """Altitude-hold contribution to the g demand, shared by every state."""
    return ((pilot.target_alt_ft - telem["alt_ft"]) * K_ALT_G
            - telem["climb_rate_fps"] * K_CLIMB_G)


class CombatPilot:
    """One dogfight policy, flown by BOTH jets.

    Instantiated twice with different aggression, so the enemy is exactly as
    capable as we are and there is only one controller to debug.

    Emits a g-demand and a roll-rate demand -- never surface deflections --
    because that is what this airframe's CAS actually accepts, and what the
    connectome would eventually have to produce in its place.
    """

    def __init__(self, aggression=1.0, callsign="", target_alt_ft=18000.0):
        self.aggression = aggression
        self.callsign = callsign
        self.target_alt_ft = target_alt_ft
        self.state = "SEARCH"
        self.flares_used = 0
        self.missiles_fired = 0
        self._low_energy = False
        self._reset_steps = 0  # how long we have been in RESET, see step()
        self._turn_dir = 1.0  # committed turn direction, see step()

    def g_available(self, airspeed_kt):
        """Real aerodynamic limit: stall speed scales as sqrt(n)."""
        return max(1.0, (airspeed_kt / STALL_SPEED_KT) ** 2)

    def recovery_floor_ft(self, airspeed_kt, pitch_deg, roll_deg):
        """AGL actually needed to pull out of the CURRENT dive.

        Ported from the airshow controller, which needed it for exactly this
        reason. Two phases, and the first is the one that is easy to forget:

        1. Roll upright. You cannot pull your way out while banked hard -- the
           lift vector points sideways, and inverted it points the nose further
           DOWN -- so the recovery rolls level first, descending at 1g
           throughout. Omitting this term is why a guard that fired at a
           geometrically "sufficient" altitude still hit the ground.
        2. The pull-out arc itself, radius v^2/((n-1)g).
        """
        v = airspeed_kt * 0.514444                      # m/s
        n = max(1.5, min(5.0, 0.8 * self.g_available(airspeed_kt)))
        gamma = math.radians(max(0.0, -pitch_deg))      # dive angle
        roll_s = abs(roll_deg) / ROLL_RATE_LIMIT_DPS
        lost_rolling_m = v * math.sin(gamma) * roll_s
        radius_m = v * v / ((n - 1.0) * 9.81)
        lost_pulling_m = radius_m * (1.0 - math.cos(gamma))
        need_ft = (lost_rolling_m + lost_pulling_m) / M_PER_FT
        return max(FLOOR_MIN_FT, need_ft * 1.5 + 500.0)

    def max_sustainable_bank_deg(self, airspeed_kt):
        """Steepest bank that can be held WITHOUT losing altitude.

        A level turn at bank phi needs n = 1/cos(phi), and only
        (V/Vstall)^2 g is available, so phi is capped at
        acos(1 / (budget * n_available)).

        This is the fix for a real crash, not a refinement: at 313kt with the
        target 178deg behind, the pilot commanded the full 80deg bank, which
        needs 5.76g to stay level while only 3.02g existed. The jet descended
        through the whole turn and hit terrain. Limiting g alone does not
        prevent it -- the BANK has to be limited too, or the controller is
        asking for a turn the wing cannot pay for.
        """
        n = TURN_LIFT_BUDGET * self.g_available(airspeed_kt)
        if n <= 1.0:
            return 0.0      # cannot sustain any banked turn at this speed
        return math.degrees(math.acos(1.0 / n))

    def _roll_to(self, desired_bank_deg, roll_deg, airspeed_kt=None):
        limit = MAX_BANK_DEG
        if airspeed_kt is not None:
            limit = min(MAX_BANK_DEG, self.max_sustainable_bank_deg(airspeed_kt))
        bank = max(-limit, min(limit, desired_bank_deg))
        rate = K_BANK_TO_RATE * angle_diff_deg(bank, roll_deg)
        return max(-CAS_ROLL_RATE_FULL_DPS, min(CAS_ROLL_RATE_FULL_DPS, rate))

    def step(self, telem, threat):
        aim = lead_bearing_deg(telem["lat"], telem["lon"], telem["airspeed_kt"],
                               telem["tgt_lat"], telem["tgt_lon"],
                               telem["tgt_heading_deg"], telem["tgt_speed_kt"])
        bearing_err = angle_diff_deg(aim, telem["heading_deg"])
        rng = telem["tgt_range_nm"]
        avail = self.g_available(telem["airspeed_kt"])

        # GROUND AVOIDANCE FIRST -- ahead of even missile evasion. The
        # ground kills faster than an AIM-9, and a break turn into a
        # missile while low is exactly how the spiral-into-terrain
        # happens. Uses AGL, not MSL: terrain here Uses AGL, not MSL: terrain here
        # sits at ~5400ft, so an MSL deck is meaningless.
        floor_ft = self.recovery_floor_ft(telem["airspeed_kt"],
                                          telem.get("pitch_deg", 0.0),
                                          telem["roll_deg"])
        if telem.get("agl_ft", 1e9) < floor_ft:
            self.state = "RECOVER"
            if abs(telem["roll_deg"]) > 30.0:
                # Roll upright, fast but NOT a bare full-rate sign flip.
                #
                # A bang-bang "-copysign(MAX_RATE, roll)" reversing hard the
                # instant roll crosses zero is only stable if the airframe
                # tracks the commanded rate with zero lag. It does not:
                # confirmed live, this exact bang-bang law produced a
                # persistent limit-cycle oscillation through the recovery
                # (roll logged at +161 -> +100 -> +35 -> +98 -> -132 -> +102
                # -> -168 -> +167 deg over ~5 seconds, never actually settling
                # near 0) while altitude kept bleeding underneath it, ending
                # in a belly-flop into terrain at spd~12kt. Using _roll_to's
                # own proportional approach to bank=0 keeps the full rate cap
                # while damping as the error shrinks, so it can actually
                # converge instead of perpetually overshooting past level.
                return Command(
                    g_cmd=1.0,
                    roll_rate_dps=self._roll_to(0.0, telem["roll_deg"],
                                                telem["airspeed_kt"]),
                    throttle=1.0, fire_missile=False, fire_gun=False,
                    flares=0, state=self.state)
            return Command(
                g_cmd=max(0.3, min(0.8 * avail, 5.0)),
                roll_rate_dps=self._roll_to(0.0, telem["roll_deg"],
                                            telem["airspeed_kt"]),
                throttle=1.0, fire_missile=False, fire_gun=False,
                flares=0, state=self.state)

        # EVADE next: a missile in the air outranks any shot we might take.
        if threat.get("missile_inbound"):
            self.state = "EVADE"
            # Break INTO the threat so the missile must pull lead and bleed
            # energy, driving it toward the 3/9 line.
            brg = threat.get("missile_bearing_deg")
            if brg is None:
                brg = bearing_err
            desired_bank = MAX_BANK_DEG if brg >= 0 else -MAX_BANK_DEG
            self.flares_used += 1
            return Command(
                g_cmd=min(avail, MAX_G),
                roll_rate_dps=self._roll_to(desired_bank, telem["roll_deg"],
                                            telem["airspeed_kt"]),
                throttle=1.0,
                fire_missile=False,
                fire_gun=False,
                flares=1,
                state=self.state,
            )

        # ENERGY before geometry. Hysteresis so it does not chatter in and out
        # of the fight at the threshold, PLUS a hard step cap: without one, a
        # wide hysteresis band let RESET run for 30-90+ real seconds straight
        # in live testing, moving the geometry so far in the meantime that it
        # never converged back into a fight. A committed pilot takes the
        # unload it needs and gets back to business -- it does not wait for a
        # perfect energy number while the target wanders off.
        if telem["airspeed_kt"] < ENERGY_FLOOR_KT and not self._low_energy:
            self._low_energy = True
            self._reset_steps = 0  # only reset on the TRANSITION into RESET,
                                    # not every step while still slow -- doing
                                    # it unconditionally meant the step cap
                                    # below could never fire for a jet that
                                    # simply never got fast again on its own.
        elif telem["airspeed_kt"] > ENERGY_RECOVER_KT or self._reset_steps > RESET_MAX_STEPS:
            self._low_energy = False
        if self._low_energy:
            self._reset_steps += 1
            self.state = "RESET"
            # EXTEND STRAIGHT, wings level, to rebuild energy as fast as
            # possible. A shallow constant bank toward the target's bearing
            # is NOT an extend -- it is a slow circle: confirmed live, holding
            # a fixed -25deg bank toward a target sitting 60-95deg off the
            # nose kept the jet in a wide, closed loop for 30+ seconds while
            # range climbed from 11nm to 17nm and away, never once actually
            # closing or re-engaging. Unloading (low g) and rolling wings
            # level converts the energy budget straight into airspeed instead
            # of spending it holding a turn that goes nowhere useful.
            # g_cmd is a bare 1.0 (true unloaded flight), NOT 1.0 plus the
            # altitude-hold correction term used everywhere else. Confirmed
            # live: adding alt_err_g here kept demanding climb/descent
            # correction g the entire time, which is directly counter to an
            # extend -- any g above the true minimum induces drag and costs
            # exactly the energy this state exists to rebuild. The result was
            # a monotonic speed bleed from 280kt to 144kt over 78 REAL
            # seconds of RESET that never once recovered, because the
            # "extend" was still quietly loaded the whole time.
            return Command(
                g_cmd=1.0,
                roll_rate_dps=self._roll_to(0.0, telem["roll_deg"],
                                            telem["airspeed_kt"]),
                throttle=1.0,
                fire_missile=False, fire_gun=False, flares=0, state=self.state)

        self.state = "ENGAGE" if rng <= ENGAGE_RANGE_NM else "INTERCEPT"

        # Turn hard enough to point at the lead solution, no harder.
        #
        # COMMIT TO A TURN DIRECTION near +-180deg bearing error. A target
        # directly behind is equally reachable turning either way, and
        # angle_diff_deg's sign there is essentially sub-degree measurement
        # noise -- confirmed live, bearing_err flickered between +178 and
        # -179 on a target sitting almost dead astern, which flipped
        # desired_bank between +80 and -80deg every 20Hz step and produced a
        # real, sustained bank oscillation (+164 -> +145 -> +157 -> -19deg)
        # that bled 3200ft of altitude.
        #
        # The deadband is around the +-180 BOUNDARY itself, not around zero:
        # +178 and -179 are only 3deg apart in reality but land on opposite
        # sides of the wrap, while +178 and +90 (also both "large") are
        # unambiguous and must be free to re-derive direction normally. So
        # only freeze the committed direction when bearing_err sits within
        # NEAR_180_DEADBAND_DEG of +-180; everywhere else the sign is
        # trustworthy and is taken fresh every step.
        if abs(abs(bearing_err) - 180.0) > NEAR_180_DEADBAND_DEG:
            self._turn_dir = 1.0 if bearing_err >= 0 else -1.0
        desired_bank = self._turn_dir * min(MAX_BANK_DEG,
                                            abs(bearing_err) * 2.0 * self.aggression)
        # PULL HARD, not just enough to stay level. This is a fighter turning
        # to engage, not an airliner holding altitude -- level_turn_g alone
        # (1/cos(bank)) is only the MINIMUM g to avoid descending, and
        # commanding just that produced slow, cautious turns (logged g=1.0-2.0
        # through most engagements against an available envelope up to 9g).
        # Pull most of the available envelope instead, so the turn radius is
        # actually tight: n = 1/cos(phi) still has to be respected as a FLOOR
        # (you cannot pull less than that without descending), but the target
        # is HARD_TURN_FRACTION of whatever the airframe can sustain, on top
        # of that floor -- plus the same altitude-error and climb-rate
        # correction terms so this does not reopen the "walks downhill while
        # turning level" bug the floor term originally fixed.
        level_turn_g = 1.0 / max(0.2, math.cos(math.radians(telem["roll_deg"])))
        alt_err = self.target_alt_ft - telem["alt_ft"]
        hard_pull_g = max(level_turn_g, avail * HARD_TURN_FRACTION)
        g_cmd = (hard_pull_g * self.aggression
                 + alt_err * K_ALT_G
                 - telem["climb_rate_fps"] * K_CLIMB_G)
        g_cmd = min(min(avail, MAX_G), g_cmd)
        g_cmd = max(0.3, g_cmd)

        shoot_missile = aim9_in_envelope(rng, bearing_err,
                                         telem["tgt_speed_kt"], telem["has_lock"])
        shoot_gun = gun_in_envelope(rng, bearing_err)
        if shoot_missile:
            self.missiles_fired += 1

        return Command(
            g_cmd=g_cmd,
            roll_rate_dps=self._roll_to(desired_bank, telem["roll_deg"],
                                        telem["airspeed_kt"]),
            # Full throttle while actively engaging -- a fighter turning to
            # kill does not manage toward a cruise setpoint mid-fight.
            # corner_throttle's proportional law is for energy management
            # (RESET), not for a jet that is currently pulling hard to close.
            throttle=1.0,
            fire_missile=shoot_missile,
            fire_gun=shoot_gun,
            flares=0,
            state=self.state,
        )
