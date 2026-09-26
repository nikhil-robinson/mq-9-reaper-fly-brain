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
# fox2.nas's own seeker can technically see out to 9nm, but its guidance
# repeatedly failed to converge on a maneuvering target from that far in
# practice -- confirmed in FlightGear's own log, all "Passed target" misses,
# miss distance tracking launch range almost exactly: 1.92nm->111m miss,
# 1.67nm->30m, 1.33nm->2.5m, 1.00nm->3.3m. Tightened to the range band where
# the miss distance was actually small enough to matter.
# 4.0, not 1.5. The seeker's OWN limit, read live off
# Current_srm.max_fire_range_nm in the sim, is 9.0nm -- 1.5 was a
# self-imposed restriction six times tighter than the weapon, and it forced
# every shot into the close-in regime where a fast jet overshoots and the
# nose cannot stay on the target (logged: bearing error stuck at -120deg
# while range grew). At 2-4nm the nose sweeps through the target repeatedly
# and the seeker's 27.5deg FOV has real opportunities. Kept well under the
# weapon's 9nm so a shot still has to be a genuine solution, not a hopeful
# launch at the horizon.
AIM9_MAX_NM = 2.0
AIM9_FOV_DEG = 27.5     # seeker field of view
# Lead is capped well inside the seeker FOV: a target we are pointing at must
# always be a target the seeker can actually see. See lead_bearing_deg.
MAX_LEAD_DEG = 18.0
# Firing boresight, much tighter than the seeker's FOV -- see
# aim9_in_envelope for the three wasted rails that set this.
AIM9_FIRE_FOV_DEG = 12.0
# Lead tapers to zero (pure pursuit) at PURE_PURSUIT_NM and is full at
# LEAD_FULL_NM -- see lead_bearing_deg for the deadlock this breaks.
PURE_PURSUIT_NM = 1.0
LEAD_FULL_NM = 3.0
# Inside this range the fight is about angles, not closure -- see the ENGAGE
# throttle for the overshoot this prevents.
CLOSE_FIGHT_NM = 2.0
# Overtake margin inside knife range: enough closure to arrive behind a
# bandit, not enough to fly past it. See the ENGAGE throttle.
CLOSE_OVERTAKE_KT = 100.0
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
# AIM-9 seeker FOV, read live off Current_srm.fcs_fov in the sim.
NOSE_ON_DEG = 27.5

STALL_SPEED_KT = 180.0          # 1g stall speed; n_max = (V/Vs)^2
CAS_ROLL_RATE_FULL_DPS = 213.1
# Roll rate we will actually ask for. Well under the CAS's full authority --
# see _roll_to for the departure this prevents.
MAX_COMBAT_ROLL_RATE_DPS = 110.0  # aileron=1.0 commands this via the roll CAS
MAX_G = 9.0
ENGAGE_RANGE_NM = 5.0
MAX_MISSILES = 4                # real AIM-9 rail count -- an empty rail can't fire
FIRE_COOLDOWN_STEPS = 60        # 3s at 20Hz: time for one shot to fly clear
                                 # before re-evaluating -- without this,
                                 # aim9_in_envelope() is re-checked every
                                 # 1/20s and stays true for many consecutive
                                 # steps, so every step in the envelope
                                 # released another round instead of one shot
                                 # per pass.
# 6.0, not 2.5. Bank error -> commanded roll rate, and at 2.5 the jet could
# not roll fast enough to fight. Confirmed live: commanded 80deg of bank with
# actual roll at -59deg -- a 21deg error -- produced only 2.5*21 = 53deg/s of
# demand, which the decode layer turns into aileron 53/213 = 0.25, and the
# measured roll rate was 2.3deg/s (-59 -> -66deg over three seconds). Meanwhile
# the target sat 83deg off the nose at 1.07nm and the pull was stuck at
# 1.8-2.3g, because g only turns the aircraft once it is BANKED: the flight
# path limiter correctly refuses to let a hard pull at shallow bank become a
# climb, so a slow roll caps the whole turn.
#
# Roll is therefore the real inner loop of the fight and it has to snap. At
# 6.0 a 21deg error commands 126deg/s and a 45deg error saturates -- still
# strictly proportional, so it damps as it converges and does not reintroduce
# the bang-bang limit cycle documented in the RECOVER branch (that was a
# full-rate SIGN FLIP at zero crossing, a different thing entirely).
K_BANK_TO_RATE = 6.0            # bank error -> commanded roll rate
MAX_BANK_DEG = 85.0
# Above this AGL a descending turn is a tactic, not a hazard, so the
# level-flight bank cap is lifted. RECOVER owns everything below it.
SAFE_TO_DESCEND_AGL_FT = 6000.0
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
K_CLIMB_G = 0.02         # g per ft/s of climb-rate ERROR (damping)
K_ALT_TO_CLIMB = 0.1     # ft/s of demanded climb per ft of altitude error
MAX_DESIRED_CLIMB_FPS = 150.0   # ~9000ft/min, a real fighter climb
# 0.75, not 2.0. At 2.0 this term ate the entire hard-turn margin: worked out
# from run 20 step 280 and matching the log exactly -- hard_pull computed to
# 4.78g, the jet was 700ft above the bandit while climbing 199fps, the
# correction hit its full -2.0g clamp, and g came out at 2.78. That is
# precisely path_hold_g at 70deg of bank, i.e. a PATH-HOLDING turn rather than
# a max-performance one, and the bearing error sat at -46 to -58deg all the
# way from 2.10nm down to 0.76nm -- always outside the seeker's 27.5deg FOV,
# so no lock was ever possible however close the jet got.
#
# A hard turn with the nose up climbs; that is not a fault to be corrected
# away mid-fight. clamp_flight_path already has full authority over a genuine
# runaway climb (it can bunt to -1g), so this term only has to nudge the
# engagement altitude, which is all its own comment ever claimed for it.
MAX_ALT_CORRECTION_G = 0.75  # nudge the engagement altitude, never override
                             # the turn -- see above


def path_hold_g(telem, roll_deg=0.0):
    """Load factor that holds the CURRENT flight path -- NOT 1.0.

    THE cause of the endless climb. The F-15's pitch axis is a load-factor
    command system (Systems/f-15-cas.xml closes on accelerations/Nz), and a
    load factor says nothing about where the flight path is going:

        dgamma/dt = g * (n * cos(phi) - cos(gamma)) / V

    The path only stops curving when n*cos(phi) == cos(gamma). Commanding a
    bare n=1.0 at gamma=+20deg does not level off -- it HOLDS +20deg of climb
    forever, and in fact steepens slightly, since g*(1 - cos 20)/V > 0. That
    is exactly the observed signature: wings level, roll -2 to -8, altitude
    climbing almost linearly and undiminished for 29 seconds no matter what
    was commanded. Every "neutral" in this file was 1.0 or 1/cos(phi), both
    of which assume gamma == 0 and so quietly command a climb to continue.

    gamma comes from the FDM's own climb rate and true airspeed, so this
    needs no new telemetry.
    """
    v_fps = max(100.0, telem["airspeed_kt"] * 1.68781)
    sin_gamma = max(-1.0, min(1.0, telem["climb_rate_fps"] / v_fps))
    gamma = math.asin(sin_gamma)
    return math.cos(gamma) / max(0.2, math.cos(math.radians(roll_deg)))


# Hard ceiling on the flight path angle, enforced over EVERY state's command.
#
# A per-state g number cannot do this job: RESET deliberately commands a bare
# path-hold g with no altitude correction (for good energy reasons, see its
# comment), and path-hold at gamma=+60deg faithfully HOLDS +60deg of climb --
# logged live climbing 33657ft -> 39671ft while airspeed fell 245 -> 157kt.
# Holding the path is the right command only when the path is acceptable.
# This is the one place that decides it is not, so no future state can
# reintroduce the bug.
# A FIXED ceiling is a fixed climb. Confirmed live: with MAX_CLIMB_ANGLE_DEG
# a constant 15deg, the limiter worked exactly as written and the jet simply
# parked against it -- g_cmd pinned at 0.94-0.96 for hundreds of steps while
# gamma sat at 14.4deg and altitude walked 14846ft -> 20899ft over 27s. A cap
# only stops the path getting STEEPER; it never brings it back. So the
# ceiling is driven by altitude error instead: at the target altitude the
# allowed climb angle is zero (level off), above it the ceiling goes NEGATIVE
# and commands a descent, and only well below it does the full climb open up.
# 30, not 15. A hard turn with the nose up climbs steeply for a few seconds
# and that is not a fault. At 15 this limiter -- not the altitude term -- was
# the thing capping the fight: run 20 step 280, reproduced exactly, gamma was
# 16.3deg in a 70deg-banked turn, just over the ceiling, so g was capped at
# cos(16.3)-0.015*1.3 over cos(70) = 2.748 against a computed hard_pull of
# 4.78. The bearing error then sat at -46..-58deg from 2.10nm all the way to
# 0.76nm, permanently outside the seeker's 27.5deg FOV.
#
# The runaway this exists to stop was gamma=59deg SUSTAINED, and it is still
# caught hard: at 59deg the cap comes out at 0.068 vertical, a real bunt. The
# total climb is bounded by COMBAT_CEILING_MARGIN_FT regardless, which is the
# limit that actually matters.
MAX_CLIMB_ANGLE_DEG = 30.0
# Floored at 0, NOT negative: this is a climb limiter, not an altitude
# controller. Letting the allowance go negative made it command a 0.2g bunt
# in level flight merely for being 5000ft high, which fights the unloaded
# extend RESET exists for. Level off here; getting back DOWN is
# alt_correction_g's job, which already has the gain and the clamp for it.
# Resolved: it mattered (see below) -- the floor is now negative, still a
# single loop rather than a second altitude controller.
# -10, not 0. "Level off but never descend" left the jet stranded ABOVE the
# bandit, and that does not just cost turn performance -- it blinds the radar.
# Probed live at textbook geometry (1.17nm, 9deg off, both inside the firing
# gates): the jet was at 24091ft against a bandit at 16500ft, and while
# armament.contact was still RED1 its position had gone STALE at 3.46nm,
# because a ~45deg depression angle puts the target outside the radar's scan
# volume. fox2's search() then evaluates that stale bearing, FOV_check fails,
# srm.Tgt stays NIL and the seeker sits at 0 forever. No lock, no shot, with
# perfect numbers in our own log.
#
# So the limiter must be able to command a real descent to get back to the
# bandit's altitude. This is the upgrade path the original ponytail note here
# called out; it is now load-bearing rather than cosmetic.
MIN_ALLOWED_GAMMA_DEG = -10.0
# A DOGFIGHT NEEDS VERTICAL WORKING ROOM.
#
# This used to allow zero climb the moment the jet reached its target
# altitude, and that starved the fight: confirmed live, the jet pulled 6.6g
# at 70deg of bank (a good turn), which naturally climbs, so the limiter cut
# g to 1.18 on the very next sample, the turn stopped dead and the bearing
# error just sat at -44deg. Pull, get cut, coast, repeat -- the limiter had
# become the reason it could not point at anything.
#
# So give the fight a BLOCK to manoeuvre in rather than a hard floor at the
# entry altitude. Full climb freedom up to COMBAT_CEILING_MARGIN_FT above
# the target, ramping to level-off over the last 3000ft. That still kills
# the runaway zoom this limiter exists for (14655 -> 39671ft was 25000ft of
# climb, five times this whole allowance) while leaving the vertical room a
# turning fight actually uses.
COMBAT_CEILING_MARGIN_FT = 5000.0
# How far above the engagement's starting altitude we will chase a climbing
# bandit -- see _step_unlimited for the thin-air stalemate this prevents.
MAX_CHASE_CLIMB_FT = 6000.0
K_ALT_GAMMA = 15.0 / 3000.0   # deg of allowed climb per ft under the ceiling
# 0.015, not 0.04. At 0.04 a 14deg overshoot subtracted 0.58g of VERTICAL
# lift, which even after dividing by cos(80deg) left only 1.96g -- still not
# enough to turn with. 0.015 leaves ~4.3g at that bank while the vertical
# component stays below cos(gamma), so the jet levels off AND fights.
K_GAMMA_G = 0.015         # g of extra bunt per degree of overshoot
MIN_BUNT_G = -1.0         # F-15 is stressed to -3g; -1 is a conservative push


def flight_path_deg(telem):
    """Flight path angle from the FDM's own climb rate and true airspeed."""
    v_fps = max(100.0, telem["airspeed_kt"] * 1.68781)
    return math.degrees(math.asin(
        max(-1.0, min(1.0, telem["climb_rate_fps"] / v_fps))))


def allowed_gamma_deg(telem, target_alt_ft):
    """Flight path angle the jet is allowed to hold, from altitude error.

    Zero at the target altitude, negative above it, opening up to a full
    climb only when genuinely low. This is the outer loop of the standard
    altitude -> flight-path -> load-factor cascade.
    """
    room_ft = (target_alt_ft + COMBAT_CEILING_MARGIN_FT) - telem["alt_ft"]
    # NEVER COMMAND A DESCENT NEAR THE GROUND.
    #
    # The negative floor exists to get back DOWN to a bandit that out-climbed
    # us; it must not help chase one into the dirt. Run 33 followed a bandit to
    # the deck and crashed (agl=2ft, spd=19kt, closest approach 0.01nm) with
    # the old level-off-only rule, so a commanded dive clearly needs its own
    # guard rather than relying on RECOVER to catch every case.
    floor = MIN_ALLOWED_GAMMA_DEG
    if telem.get("agl_ft", 1e9) < SAFE_TO_DESCEND_AGL_FT:
        floor = 0.0
    return max(floor, min(MAX_CLIMB_ANGLE_DEG, room_ft * K_ALT_GAMMA))


def clamp_flight_path(cmd, telem, target_alt_ft):
    """Cap g so the flight path returns to allowed_gamma_deg.

    Above the allowance the g cap goes BELOW cos(gamma), which is what
    actually curves the nose back down -- cos(gamma) alone only stops it
    curving up. Never RAISES a command, so a state asking for less than the
    cap keeps what it asked for (a deliberate dive is untouched).
    """
    gamma = flight_path_deg(telem)
    gamma_max = allowed_gamma_deg(telem, target_alt_ft)
    if gamma <= gamma_max:
        return cmd
    # The cap is on the VERTICAL component of lift, n*cos(phi) -- not on n.
    # Capping total g strangled the turn: confirmed live, with the cap applied
    # to n directly the jet sat at g=0.93-0.98 through an entire engagement
    # while bearing error GREW from -25deg to -89deg as range closed to
    # 0.69nm. It closed the range and then flew straight past, every time,
    # because a 0.97g command at 45deg of bank is barely a turn at all.
    #
    # Dividing by cos(phi) is the level-turn relation: at 80deg of bank the
    # jet may pull 5.5g and still not climb, because only 0.95 of it points
    # up. Altitude is held by the vertical component alone, which leaves the
    # horizontal component -- the part that actually turns the aircraft --
    # completely free.
    roll_deg = telem.get("roll_deg", 0.0)
    vertical_cap = math.cos(math.radians(gamma)) - K_GAMMA_G * (gamma - gamma_max)
    g_cap = vertical_cap / max(0.2, math.cos(math.radians(roll_deg)))
    return cmd._replace(g_cmd=max(MIN_BUNT_G, min(cmd.g_cmd, g_cap)))
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
    """Bearing of the COLLISION COURSE to the target.

    Proper lead is an ANGLE, not an extrapolated position, and it does not
    depend on range at all. On a collision course our velocity component
    across the line of sight must cancel the target's:

        Vm * sin(L) = Vt * sin(theta)

    where theta is the angle between the target's velocity and the LOS, and L
    is the lead angle to add to the LOS. asin bounds it automatically, and
    when the target is faster across the LOS than we can match, it saturates
    at a pure beam aim instead of producing nonsense.

    THE OLD VERSION EXTRAPOLATED A POSITION and that is what stopped the jet
    turning. It used t_go = range/own_speed and advanced the target along its
    track for that long. At dogfight range against a fast crossing bandit the
    displacement is the same order as the range itself: logged live at 1.89nm
    with a 400kt target, t_go came to 14.6s and the target was pushed 3000m
    (1.6nm) along its heading -- so the "lead point" landed somewhere near our
    own nose, desired_bank collapsed to roughly zero, and the aircraft rolled
    wings-level with the bandit 40deg off. Probed in the sim mid-fight:
    /controls/flight/aileron read -0.027, -0.010, +0.016 -- no roll command at
    all -- while bank drifted 66 -> 59 -> 43deg and the target sat off the
    nose. The controller was not failing to turn; it believed it was already
    pointing correctly.

    The long-range failure the old docstring described (t_go blowing up near
    parallel headings, bank swinging +179 -> -178deg) cannot happen here:
    there is no time term to blow up.
    """
    los_deg = bearing_deg(own_lat, own_lon, tgt_lat, tgt_lon)
    own_kt = max(1.0, own_speed_kt)
    # Target velocity across the LOS, as a fraction of our speed.
    theta = math.radians(angle_diff_deg(tgt_heading_deg, los_deg))
    across = (tgt_speed_kt * math.sin(theta)) / own_kt
    lead_deg = math.degrees(math.asin(max(-1.0, min(1.0, across))))
    # NEVER LEAD THE TARGET OUT OF THE SEEKER'S FIELD OF VIEW.
    #
    # A pure collision-course lead can be large: worked out from run 17's own
    # log (LOS 4deg, bandit heading 45deg at 400kt, us at 478kt) it comes to
    # 33.3deg. That is geometrically correct for an intercept and tactically
    # useless here, for two reasons.
    #
    # First, the AIM-9 seeker's field of view is 27.5deg (read live off
    # Current_srm.fcs_fov). Aiming 33deg off the target puts the bandit
    # OUTSIDE the seeker cone even when the jet is perfectly on aim -- it can
    # never lock, which is exactly what every run showed.
    #
    # Second, the pilot's bearing error becomes the lead-point error, not the
    # target error: -21deg instead of the -54deg to the actual bandit. That
    # commands 41deg of bank instead of 85, and it shrinks as the turn
    # proceeds, so the jet rolls out mid-fight (logged: -38 -> -12deg) while
    # the bandit is still well off the nose.
    #
    # Capping the lead keeps the intercept benefit at the aspects where it is
    # small and real, and degenerates to pure pursuit exactly when the lead
    # would otherwise throw the shot away. MAX_LEAD_DEG sits inside the FOV
    # with margin, so a target we are pointing at is always a target the
    # seeker can see.
    # TAPER TO PURE PURSUIT AS RANGE CLOSES.
    #
    # A fixed lead cap makes the firing gate unreachable. The steering loop
    # settles where the nose is on the AIM POINT, so the steady-state LOS
    # error to the bandit is about the lead angle itself: with the cap at
    # 18deg, runs 23 and 24 both stabilised at a best pointing of 23deg and
    # never once got inside the 12deg firing gate -- 26 locks, zero shots.
    # Capping the lead (needed to keep the target in the seeker's FOV) and
    # gating the shot on boresight are individually right and deadlock
    # together.
    #
    # Lead pursuit is for the intercept; the endgame is pure pursuit, nose on
    # the bandit. Taper between them by range, so the jet still cuts the
    # corner on the way in and points straight at the target once it is close
    # enough to shoot.
    rng_nm = haversine_m(own_lat, own_lon, tgt_lat, tgt_lon) / 1852.0
    taper = ((rng_nm - PURE_PURSUIT_NM)
             / max(1e-6, LEAD_FULL_NM - PURE_PURSUIT_NM))
    cap = MAX_LEAD_DEG * max(0.0, min(1.0, taper))
    lead_deg = max(-cap, min(cap, lead_deg))
    return (los_deg + lead_deg) % 360.0


def corner_throttle(airspeed_kt, corner_kt=CORNER_SPEED_KT):
    """Throttle that drives airspeed toward corner speed.

    Proportional, clamped to a valid throttle. Being fast is not free in a
    turning fight: above corner the radius grows and the fight is lost by
    geometry rather than by energy.
    """
    # AT OR BELOW CORNER SPEED, EVERYTHING YOU HAVE.
    #
    # The old proportional law returned only 0.90 at 394kt, and a 4-5g turn at
    # that speed bleeds energy faster than 90% thrust replaces it. Combined
    # with cutting full throttle inside knife range (which was right, to stop
    # the overshoot) it produced an energy death spiral with no way back:
    # logged live, the jet sank from 15686ft/394kt into 300+ steps of RECOVER,
    # wings level, commanding g=0.80 -- LESS THAN 1g -- and flew into the
    # ground. It was not a control failure. At 155kt the available g is
    # (155/180)^2 = 0.74, i.e. below its own 1g stall speed, so no command
    # could have saved it.
    #
    # Corner speed is a TARGET, not a ceiling to creep up on: below it, full
    # thrust; only above it does pulling power buy a tighter turn.
    if airspeed_kt <= corner_kt:
        return 1.0
    return max(0.0, min(1.0, 0.55 - (airspeed_kt - corner_kt) / 300.0))


def aim9_in_envelope(range_nm, bearing_err_deg, target_speed_kt, has_lock):
    """Whether firing an AIM-9 now is legitimate.

    Without explicit discipline an AI empties its rails at zero range and
    nothing ever hits. Every one of these is a real launch precondition in
    weapons.nas/fox2.nas, not a heuristic.
    """
    # A LOCK IS NOT A SHOT. The seeker needs its full 27.5deg FOV just to SEE
    # the target; a round that actually guides needs the bandit near boresight
    # and close in. Run 22 launched all three AIM-9s at 3.64nm, 3.80nm and
    # 2.10nm with 21-31deg of LOS error -- legal by the old envelope, and all
    # three missed: 52 locks, 3 launches, 1% damage, 0 kills. With only four
    # rounds for three bandits, a long-range off-boresight shot at a hard
    # maneuvering target is a rail thrown away. The same run closed to 0.11nm,
    # so the tight shot is there to be taken.
    return (has_lock
            and AIM9_MIN_NM <= range_nm <= AIM9_MAX_NM
            and abs(bearing_err_deg) <= AIM9_FIRE_FOV_DEG
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
        self._fight_base_alt_ft = None
        self.state = "SEARCH"
        self.flares_used = 0
        self.missiles_fired = 0
        self._low_energy = False
        self._reset_steps = 0  # how long we have been in RESET, see step()
        self._turn_dir = 1.0  # committed turn direction, see step()
        self._fire_cooldown = 0  # steps remaining before another shot may fire

    def g_available(self, airspeed_kt):
        """Real aerodynamic limit: stall speed scales as sqrt(n).

        NO 1.0 FLOOR. The floor was a lie about the physics, and it hid the
        one condition that actually kills the aircraft: below the 1g stall
        speed the wing cannot even hold level flight, and every consumer of
        this number was told otherwise. That is how RECOVER came to command
        0.8g (0.8 x a floored 1.0) while sinking -- logged live for 300+ steps,
        wings level, from 6540ft into the ground at 150-210kt. At 155kt the
        honest answer is (155/180)^2 = 0.74.

        Reporting it truthfully makes the g cap bind correctly at low speed, so
        the controller stops demanding lift the wing cannot produce; the way
        out of that corner is thrust and unloading, not a bigger g command.
        """
        return (airspeed_kt / STALL_SPEED_KT) ** 2

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

    def _roll_to(self, desired_bank_deg, roll_deg, airspeed_kt=None,
                 agl_ft=None):
        """Roll-rate demand toward a bank angle.

        The level-flight bank cap applies only when there is no altitude to
        spend. max_sustainable_bank_deg answers "what bank can I hold WITHOUT
        DESCENDING", which is the right question next to the ground and the
        wrong one with two miles of air underneath: a fighter wins angles by
        putting the lift vector into the turn and letting the nose fall, not
        by holding altitude.

        Confirmed live in run 15 -- a genuine turning fight at last (3.3-3.7g
        at 76-79deg of bank, bearing error finally coming back from -90 to
        -87deg), but a stalemate at 1.42nm, because the cap held bank at
        ~78deg, the jet climbed 14874 -> 17941ft against a bandit at 16500,
        and the flight-path limiter then throttled it to 3.3g when 5.9g was
        available at that speed. Capping bank for level flight forced the
        climb that cost the turn.

        RECOVER already owns the low-altitude case (see recovery_floor_ft), so
        below SAFE_TO_DESCEND_AGL_FT keep the conservative cap and above it
        allow the full bank.
        """
        limit = MAX_BANK_DEG
        if airspeed_kt is not None and (agl_ft is None
                                        or agl_ft < SAFE_TO_DESCEND_AGL_FT):
            limit = min(MAX_BANK_DEG, self.max_sustainable_bank_deg(airspeed_kt))
        bank = max(-limit, min(limit, desired_bank_deg))
        rate = K_BANK_TO_RATE * angle_diff_deg(bank, roll_deg)
        # CAP THE ROLL RATE WELL BELOW THE CAS'S FULL AUTHORITY.
        #
        # Clamping at CAS_ROLL_RATE_FULL_DPS (213deg/s) let the controller
        # command the airframe into roll-coupled departure. Probed live
        # mid-fight: p reached 3.598 rad/s (206deg/s) with the jet inverted at
        # roll=+141deg and accelerations/Nz reading 22.8g, then 12.0, 10.6,
        # 10.9, 11.4 -- while our own g_cmd was between 1 and 6.6. The
        # aircraft was not tracking our command at all; it was departing, and
        # the g-meter spikes are the inertial coupling of a fast roll at
        # alpha. That is the "roll collapse" seen in every run's log
        # (-63 -> -42 -> +1 -> -44deg): the jet snapping past the target bank,
        # through inverted, and back.
        #
        # Raising K_BANK_TO_RATE to 6.0 cured the opposite problem (rolling at
        # 2.3deg/s, unable to fight) and overshot into this one. Real fighters
        # limit roll rate at high AoA for exactly this reason, so cap the
        # DEMAND rather than detune the gain: the loop stays responsive to
        # large errors and simply refuses to ask for a departure.
        return max(-MAX_COMBAT_ROLL_RATE_DPS,
                   min(MAX_COMBAT_ROLL_RATE_DPS, rate))

    def step(self, telem, threat):
        """Decide this frame's command, then hard-limit the flight path.

        _step_unlimited picks the tactical command; clamp_flight_path is a
        final authority over all of them. See MAX_CLIMB_ANGLE_DEG for why no
        per-state g tuning can replace it.
        """
        return clamp_flight_path(self._step_unlimited(telem, threat),
                                 telem, self.target_alt_ft)

    def _step_unlimited(self, telem, threat):
        if self._fire_cooldown > 0:
            self._fire_cooldown -= 1

        # FIGHT AT THE BANDIT'S ALTITUDE, not the one we happened to start at.
        #
        # target_alt_ft was set once, at handover, to OUR entry altitude, and
        # never moved. The bandits orbit at 16500ft while the jet took the
        # stick at ~14900, so the controller spent every engagement defending
        # an altitude 1600ft BELOW the aircraft it was chasing -- it climbed
        # to reach them, then braked itself for climbing.
        #
        # Logged live and reproduced exactly by hand: at step 580 hard_pull
        # computed to 4.25g, alt_correction hit its full -2.0g clamp because
        # the jet was 4600ft "too high", and g_cmd came out at 2.25 -- the
        # precise value in the log, while the target sat 110deg off the nose
        # and the range opened. Both altitude brakes (alt_correction here and
        # clamp_flight_path's ceiling) were fighting the turn for doing the
        # one thing an intercept has to do.
        #
        # An intercept's reference altitude IS the target's. Track it, so
        # alt_err goes to zero when co-altitude and the combat block centres
        # on the fight instead of on wherever we entered.
        tgt_alt = telem.get("tgt_alt_ft")
        if tgt_alt is not None and tgt_alt > 0:
            # DO NOT FOLLOW A BANDIT INTO AIR YOU CANNOT FIGHT IN.
            #
            # Tracking the target's altitude (see below) is right, but
            # unbounded it lets a climbing bandit drag the fight up until the
            # wing runs out. Run 27: the jet followed RED2 up to 28128ft at
            # 366kt CAS -- a 4.1g budget in thin air -- and sat at 85deg of
            # bank unable to point, bearing error swinging -107 to -66deg at
            # 4.5nm. Cap the reference at a fighting ceiling above where the
            # engagement began; below that ceiling it still tracks normally.
            if self._fight_base_alt_ft is None:
                self._fight_base_alt_ft = telem["alt_ft"]
            self.target_alt_ft = min(
                tgt_alt, self._fight_base_alt_ft + MAX_CHASE_CLIMB_FT)
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
                    g_cmd=path_hold_g(telem),
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
                                            telem["airspeed_kt"],
                                            telem.get("agl_ft")),
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
                g_cmd=path_hold_g(telem),
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
        # BANK STEEPLY: bank is what turns g into a TURN instead of a climb.
        #
        # At 2.0 this ran far too shallow once the lead cap shrank the
        # steering error. Run 21, logged: bearing error ~-35deg raw, about
        # -17deg to the aim point after the 18deg lead cap, so desired_bank
        # came out at only 34deg -- and a 4g pull at 45deg of bank puts
        # 4*cos(45) = 2.8g straight into the vertical. The jet climbed
        # 14887 -> 18025ft, went above the 16500ft bandit, and the altitude
        # terms then braked it back to 0.9-1.4g while the bearing error grew
        # from -35 to -56deg.
        #
        # At 4.0 a 17deg error commands 68deg of bank and 21deg saturates at
        # the limit. The same 4g pull at 80deg is only 0.7g vertical -- nearly
        # all of it goes into pointing the nose, which is the entire objective.
        desired_bank = self._turn_dir * min(MAX_BANK_DEG,
                                            abs(bearing_err) * 4.0 * self.aggression)
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
        level_turn_g = path_hold_g(telem, telem["roll_deg"])
        alt_err = self.target_alt_ft - telem["alt_ft"]
        # THE PULL MUST BE EARNED WITH BANK.
        #
        # Confirmed live with real FCS telemetry, not inference: at step 100
        # we commanded g_cmd=3.98, the airframe's own fcs/stick-commanded-g
        # read 3.79 and accelerations/Nz read 4.03 -- the command chain is
        # accurate end to end and the jet delivers exactly the g we ask for.
        # The problem was WHERE that g pointed. Bank was only -9 to -21deg,
        # so cos(roll) ~ 0.96 of a 4g pull went straight into the vertical:
        # a 4g LOOP, not a turn. flight-path/gamma-deg confirms it -- 27.8deg
        # at step 100, 59.4deg by step 220, and the jet zoomed from 14655ft
        # to 39671ft and stalled at 157kt.
        #
        # Only the horizontal component of lift, n*sin(phi), turns the
        # aircraft. At 9deg of bank a hard pull buys essentially no turn rate
        # and costs the entire energy budget. So scale the excess above the
        # path-holding baseline by how much of it would actually go into the
        # turn. At wings level this collapses to level_turn_g (pure path
        # hold, no climb); at 60deg bank it is 87% of the hard pull; at 90deg
        # it is all of it. This also fixes itself: not pulling 4g at low bank
        # leaves the roll axis the authority to actually GET to the bank
        # angle, so the jet rolls first and then pulls, which is the correct
        # order and what makes the turn tight instead of vertical.
        turn_fraction = abs(math.sin(math.radians(telem["roll_deg"])))
        hard_pull_g = level_turn_g + max(
            0.0, avail * HARD_TURN_FRACTION - level_turn_g) * turn_fraction
        # The altitude/climb-rate correction is only meant to nudge the
        # engagement altitude and damp oscillation (see
        # test_climb_rate_is_damped_so_altitude_does_not_oscillate) -- it is
        # NOT meant to fight the vertical rate a hard pursuit turn naturally
        # produces. Uncapped, a jet pulling ~4.8g with the nose up easily
        # climbs at 150-200fps, and -climb_rate*K_CLIMB_G alone then
        # subtracts 3-4g, gutting the turn down near the 0.3g floor -- logged
        # live as g=0.87-1.5 through most of an engagement despite a large
        # bearing error, the jet flying nearly straight while the AI kept
        # maneuvering hard. Clamp the correction so it can steer the
        # engagement altitude but never cancel out the intended pull.
        # DAMP THE CLIMB-RATE ERROR, NOT THE CLIMB RATE.
        #
        # These two terms used to fight each other whenever the jet had to
        # change altitude -- which, now that it fights at the bandit's
        # altitude, is most of an engagement. Worked out by hand from run 14
        # step 60 and matching the log exactly: 1535ft below the bandit gave
        # +0.38g of altitude demand, while the 123fps climb that was closing
        # that very gap produced -2.46g of damping. Net went negative, g_cmd
        # floored at 0.30, the pull died and the turn died with it -- the log
        # shows roll collapsing (-38 -> -12deg) in lockstep with every g=0.30
        # sample.
        #
        # Damping exists to stop an UNWANTED vertical rate, so it has to
        # measure against the rate we actually want. Ask for a climb rate
        # proportional to the altitude error, then damp the difference. At
        # the target altitude desired is zero and this reduces to the old
        # behaviour, so a runaway climb is still caught just as hard.
        desired_climb_fps = max(-MAX_DESIRED_CLIMB_FPS,
                                min(MAX_DESIRED_CLIMB_FPS,
                                    alt_err * K_ALT_TO_CLIMB))
        alt_correction_g = (desired_climb_fps
                            - telem["climb_rate_fps"]) * K_CLIMB_G
        alt_correction_g = max(-MAX_ALT_CORRECTION_G,
                               min(MAX_ALT_CORRECTION_G, alt_correction_g))
        g_cmd = hard_pull_g * self.aggression + alt_correction_g
        g_cmd = min(min(avail, MAX_G), g_cmd)
        # NEVER UNLOAD BELOW THE CURRENT FLIGHT PATH WHILE ENGAGING.
        #
        # The 0.3 floor let the controller fall into a self-sustaining hole,
        # reproduced at the same point in three consecutive runs (steps 60-80:
        # g=0.30, roll -8 to -13deg). Once bank collapses, turn_fraction goes
        # with it, hard_pull drops to ~1.7g, alt_correction hits its full
        # -2.0g clamp, and the sum floors at 0.3 -- but at 0.3g there is no
        # turn, so the bank never comes back and the jet stays in the hole
        # while the bandit walks away.
        #
        # path_hold_g is the g that merely holds the flight path at this bank:
        # asking for less is unloading, which is only ever right in RESET (a
        # deliberate extend, which has its own branch and does not come
        # through here). Flooring here breaks the feedback loop without
        # capping anything -- a real bunt from the flight-path limiter still
        # applies afterwards, because clamp_flight_path runs last.
        g_cmd = max(path_hold_g(telem, telem["roll_deg"]), g_cmd)
        # RE-APPLY THE PHYSICAL CAP AFTER THE FLOOR.
        #
        # path_hold_g divides by cos(phi), so at extreme bank it asks for
        # enormous numbers -- 11.5g at 85deg -- and the floor above was
        # overriding the aerodynamic limit that had already been applied. The
        # wing cannot make g it does not have: at 28128ft and 366kt CAS the
        # budget is (366/180)^2 = 4.1g, and commanding more just buffets and
        # bleeds speed, which is exactly the state run 27 got stuck in (85deg
        # of bank, 366kt, unable to point).
        g_cmd = min(min(avail, MAX_G), g_cmd)
        g_cmd = max(0.3, g_cmd)

        # SHOOT ON THE TRUE LINE OF SIGHT, NOT THE LEAD POINT.
        #
        # bearing_err is the error to the LEAD point -- correct for steering,
        # wrong for deciding a shot. The seeker looks at the target, not at
        # where we are aiming. Confirmed live in run 19: two AIM-9s were
        # launched at a logged brg_err of -40deg, because after the 18deg lead
        # cap the pilot's own bearing_err read about -22deg and passed the
        # 27.5deg FOV test. Both missiles went off with the bandit 40deg off
        # boresight -- outside the seeker's view -- and neither guided: 14
        # lock samples, 2 launches, 0 damage, 0 kills.
        #
        # Steer with lead, shoot on LOS.
        los_err = angle_diff_deg(
            bearing_deg(telem["lat"], telem["lon"],
                        telem["tgt_lat"], telem["tgt_lon"]),
            telem["heading_deg"])
        shoot_missile = (aim9_in_envelope(rng, los_err, telem["tgt_speed_kt"],
                                          telem["has_lock"])
                         and self._fire_cooldown <= 0
                         and self.missiles_fired < MAX_MISSILES)
        shoot_gun = gun_in_envelope(rng, los_err)
        if shoot_missile:
            self.missiles_fired += 1
            self._fire_cooldown = FIRE_COOLDOWN_STEPS

        # Speed to fly inside knife range -- the bandit's, plus an overtake
        # margin. See the throttle comment below.
        close_ref_kt = min(CORNER_SPEED_KT,
                           telem["tgt_speed_kt"] + CLOSE_OVERTAKE_KT)
        return Command(
            g_cmd=g_cmd,
            roll_rate_dps=self._roll_to(desired_bank, telem["roll_deg"],
                                        telem["airspeed_kt"],
                                        telem.get("agl_ft")),
            # CORNER SPEED WHEN TURNING, FULL THROTTLE WHEN POINTING.
            #
            # The old comment here said a fighter turning to kill does not
            # manage toward a setpoint mid-fight. That is exactly backwards,
            # and it is why this jet never got a shot: corner speed IS the
            # speed for maximum turn rate, and above it the fight is lost by
            # geometry no matter how hard the pull.
            #
            # Confirmed live: at full throttle throughout an engagement the
            # jet ran to 672-692kt -- ~190kt past corner -- with bank pinned
            # at the -80deg limit and 5g, and the bearing error simply STUCK
            # at -116 to -131deg while range GREW from 2.07nm to 3.03nm. Turn
            # radius is V^2/(g*sqrt(n^2-1)): at 690kt and 5g that is ~1.4nm,
            # a 2.8nm circle, so a target 2nm away is geometrically
            # unreachable. At corner (500kt) and 9g it is ~0.4nm.
            #
            # So slow to corner while the nose is off the target, and push it
            # up only once pointed, where speed buys closure instead of
            # radius. NOSE_ON_DEG is the AIM-9 seeker's own FOV (27.5deg,
            # read live off Current_srm.fcs_fov) -- inside it there is a shot
            # to take, outside it there is only a turn to win.
            # Full throttle only when speed buys CLOSURE. Inside knife-fight
            # range with the nose still off, speed buys turn radius instead --
            # run 23 held a steady -23..-26deg at 1.1nm (inside the seeker
            # cone, never inside the 12deg firing gate), stayed at full
            # throttle because 25deg counts as "nose-on", and drove straight
            # through the overshoot: bearing error snapped to -121deg, and the
            # recovery cost 6000ft of climb and 120kt.
            # los_err, not bearing_err: "am I pointed at the bandit" is a
            # line-of-sight question, exactly as for the firing gate. Using
            # the lead error here read 7deg while the target sat 25deg off,
            # so the jet counted itself as pointed and kept the power in.
            # INSIDE KNIFE RANGE, NEVER FULL THROTTLE -- pointed or not.
            #
            # Being well aimed used to license full power, which is how the
            # jet flew THROUGH the bandit: once the endgame became pure
            # pursuit the nose points straight at it, so full throttle closes
            # the last mile far too fast, the bearing error snaps to
            # +-120-170deg as it passes, the bandit leaves the 27.5deg seeker
            # cone and the lock dies. Observed exactly that way -- a lock
            # comes up, the jet flies away from the target, the lock drops.
            #
            # A gun/missile pass is flown at a speed that lets you STAY
            # behind, so inside CLOSE_FIGHT_NM manage energy regardless of
            # aim; full power is for closing from outside it.
            # INSIDE KNIFE RANGE, MATCH THE BANDIT'S SPEED, NOT CORNER.
            #
            # Corner speed is 500kt and "full thrust at or below corner" is
            # right for a turning fight -- but against a SLOW bandit in knife
            # range it is how you fly straight past. Logged live: RED1 down to
            # 158kt while we held 500kt, so the close-range rule asked for full
            # afterburner and the bearing error was -155deg at 0.52nm. We had
            # the range gate and threw the shot away on closure.
            #
            # So inside CLOSE_FIGHT_NM the reference speed becomes the
            # target's, plus an overtake margin: enough to close and stay
            # behind, not enough to pass. Outside it, corner speed still rules
            # because the fight there is about turn performance. The same
            # full-thrust-when-slow guard applies to the lower reference, so
            # this cannot reproduce the energy death spiral.
            throttle=(corner_throttle(telem["airspeed_kt"], close_ref_kt)
                      if rng <= CLOSE_FIGHT_NM
                      else (corner_throttle(telem["airspeed_kt"])
                            if abs(los_err) > NOSE_ON_DEG else 1.0)),
            fire_missile=shoot_missile,
            fire_gun=shoot_gun,
            flares=0,
            state=self.state,
        )

def _selfcheck_flight_path_limiter():
    """The regression that flew the jet from 14655ft to 39671ft at 157kt,
    and then the one where it parked at a constant 14.4deg climb."""
    C = Command(g_cmd=4.0, roll_rate_dps=0.0, throttle=1.0, fire_missile=False,
                fire_gun=False, flares=0, state="ENGAGE")
    TGT = 15000.0

    def tel(kt, gamma_deg, alt):
        return {"airspeed_kt": kt, "alt_ft": alt, "roll_deg": 0.0,
                "climb_rate_fps": kt * 1.68781 * math.sin(math.radians(gamma_deg))}

    # ONCE THE COMBAT BLOCK IS USED UP, any climb is capped. This is the fix
    # for the jet parking against a fixed ceiling and climbing forever at
    # 14.4deg -- note the test is at the TOP of the block, not at the entry
    # altitude, because a fight is allowed to climb inside its block.
    topped_out = tel(520.0, 14.4, 15000.0 + COMBAT_CEILING_MARGIN_FT)
    assert clamp_flight_path(C._replace(g_cmd=0.95), topped_out, TGT).g_cmd < 0.9

    # ABOVE the combat block, climbing is shut off entirely.
    high = tel(520.0, 2.0, 21000.0)             # 6000ft over a 15000ft target
    assert allowed_gamma_deg(high, TGT) == 0.0  # level off, do not force a dive

    # INSIDE the block the fight is left alone -- this is what stopped the
    # limiter strangling a good 6.6g/70deg turn down to 1.18g.
    in_block = tel(470.0, 8.0, 15000.0)
    assert allowed_gamma_deg(in_block, TGT) == MAX_CLIMB_ANGLE_DEG
    assert clamp_flight_path(C._replace(g_cmd=6.6), in_block, TGT).g_cmd == 6.6
    assert clamp_flight_path(C, high, TGT).g_cmd < 1.0

    # WELL BELOW it a real climb is allowed -- the limiter must not stop the
    # jet recovering altitude it actually needs.
    low = tel(450.0, 10.0, 10000.0)
    assert allowed_gamma_deg(low, TGT) > 10.0
    assert clamp_flight_path(C, low, TGT).g_cmd == 4.0  # untouched

    # The original failure: gamma=59.4deg must BUNT, not merely stop pulling.
    out = clamp_flight_path(C._replace(g_cmd=3.087),
                            tel(429.0, 59.4, 19589.0), TGT).g_cmd
    assert out < 0.0 and out >= MIN_BUNT_G, out

    # A dive is never touched, and the cap never RAISES a command.
    assert clamp_flight_path(C._replace(g_cmd=0.5),
                             tel(450.0, -20.0, 15000.0), TGT).g_cmd == 0.5

    # THE TURN MUST SURVIVE THE CAP. Same climbing state, but banked hard:
    # the cap is on the vertical component, so a hard turn keeps most of its
    # g. Capping total g instead pinned the jet at ~0.95g and it flew past
    # every target while the bearing error grew.
    climbing = tel(520.0, 14.4, 15000.0 + COMBAT_CEILING_MARGIN_FT)
    level_cap = clamp_flight_path(C, climbing, TGT).g_cmd
    climbing_banked = dict(climbing, roll_deg=-80.0)
    banked_cap = clamp_flight_path(C, climbing_banked, TGT).g_cmd
    assert banked_cap > 3.5, banked_cap        # a real turn is still allowed
    assert banked_cap > level_cap * 4.0        # ~1/cos(80deg) more than level
    # ...and it still levels off: vertical component stays under cos(gamma).
    assert banked_cap * math.cos(math.radians(80.0)) < math.cos(math.radians(14.4))


def _selfcheck_path_hold_g():
    """The regression that let the jet climb for 29s straight."""
    level = {"airspeed_kt": 400.0, "climb_rate_fps": 0.0}
    assert abs(path_hold_g(level) - 1.0) < 1e-6  # level: n=1 is correct

    # 20deg climb at 400kt TAS -> 231 fps vertical. Holding the path needs
    # cos(20)=0.94g, NOT 1.0g. The old 1.0 is what kept the nose up.
    climbing = {"airspeed_kt": 400.0, "climb_rate_fps": 400.0 * 1.68781 * math.sin(math.radians(20.0))}
    assert abs(path_hold_g(climbing) - math.cos(math.radians(20.0))) < 1e-6
    assert path_hold_g(climbing) < 1.0  # commands LESS than 1g while climbing

    # Banked: still divided by cos(phi), so a level turn floor survives.
    assert path_hold_g(level, 60.0) > 1.9
    # Diving: needs MORE than the climbing case, less than 1.0 is wrong here.
    diving = {"airspeed_kt": 400.0, "climb_rate_fps": -200.0}
    assert path_hold_g(diving) < 1.0 and path_hold_g(diving) > 0.9
    # Degenerate telemetry must not divide by zero or blow up.
    assert 0.0 < path_hold_g({"airspeed_kt": 0.0, "climb_rate_fps": 9999.0}) <= 1.0


if __name__ == "__main__":
    _selfcheck_path_hold_g()
    _selfcheck_flight_path_limiter()
    print("combat self-checks OK")
