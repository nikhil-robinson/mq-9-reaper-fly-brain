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
CORNER_SPEED_KT = 430.0

STALL_SPEED_KT = 180.0          # 1g stall speed; n_max = (V/Vs)^2
CAS_ROLL_RATE_FULL_DPS = 213.1  # aileron=1.0 commands this via the roll CAS
MAX_G = 9.0
ENGAGE_RANGE_NM = 5.0
K_BANK_TO_RATE = 2.5            # bank error -> commanded roll rate
MAX_BANK_DEG = 80.0

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


def lead_bearing_deg(own_lat, own_lon, own_speed_kt,
                     tgt_lat, tgt_lon, tgt_heading_deg, tgt_speed_kt):
    """Bearing to where the target WILL be, not where it is.

    Pure pursuit (pointing at the target's current position) is right for a
    fixed waypoint and wrong for a dogfight: against a maneuvering jet it
    degenerates into a lag chase that never closes. Estimate time-to-go from
    present range and own speed, advance the target along its own velocity
    vector, and aim there. One iteration is enough at dogfight ranges.
    """
    rng_m = haversine_m(own_lat, own_lon, tgt_lat, tgt_lon)
    own_ms = max(1.0, own_speed_kt * KT_TO_MS)
    t_go = rng_m / own_ms
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
                # Roll upright at the FULL rate, unloaded. Pulling while banked
                # hard just turns; every tenth of a second here is altitude, so
                # do not trickle it through a proportional gain.
                return Command(
                    g_cmd=1.0,
                    roll_rate_dps=-math.copysign(ROLL_RATE_LIMIT_DPS,
                                                 telem["roll_deg"]),
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

        self.state = "ENGAGE" if rng <= ENGAGE_RANGE_NM else "INTERCEPT"

        # Turn hard enough to point at the lead solution, no harder.
        desired_bank = max(-MAX_BANK_DEG,
                           min(MAX_BANK_DEG, bearing_err * 2.0 * self.aggression))
        # A level turn at bank phi needs n = 1/cos(phi) just to hold altitude,
        # plus a correction for altitude error and damping on climb rate --
        # without those two terms the jet descends steadily even while turning
        # "level", which walked both jets 9500ft downhill in 45s.
        level_turn_g = 1.0 / max(0.2, math.cos(math.radians(telem["roll_deg"])))
        alt_err = self.target_alt_ft - telem["alt_ft"]
        g_cmd = (level_turn_g * self.aggression
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
            throttle=corner_throttle(telem["airspeed_kt"]),
            fire_missile=shoot_missile,
            fire_gun=shoot_gun,
            flares=0,
            state=self.state,
        )
