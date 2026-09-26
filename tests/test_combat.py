import math
from fly_brain.combat import (
    AIM9_FIRE_FOV_DEG,
    AIM9_FOV_DEG,
    CORNER_SPEED_KT,
    CombatPilot,
    RESET_MAX_STEPS as RESET_MAX_STEPS_TEST,
    aim9_in_envelope,
    corner_throttle,
    gun_in_envelope,
    is_lethal,
    CAS_ROLL_RATE_FULL_DPS,
    MAX_BANK_DEG,
    MAX_COMBAT_ROLL_RATE_DPS,
    MAX_LEAD_DEG,
    MIN_ALLOWED_GAMMA_DEG,
    MIN_BUNT_G,
    allowed_gamma_deg,
    lead_bearing_deg,
    path_hold_g,
)
from fly_brain.geo import angle_diff_deg, bearing_deg, offset_point


def test_lead_bearing_equals_pure_bearing_for_a_stationary_target():
    pure = bearing_deg(37.0, -115.0, 37.05, -115.0)
    lead = lead_bearing_deg(37.0, -115.0, 400.0, 37.05, -115.0, 0.0, 0.0)
    assert abs(angle_diff_deg(lead, pure)) < 0.5


def test_lead_bearing_aims_ahead_of_a_crossing_target():
    # Target due north of us, running due east. We must aim EAST of its
    # current position, i.e. at a larger bearing than pure pursuit.
    pure = bearing_deg(37.0, -115.0, 37.05, -115.0)
    lead = lead_bearing_deg(37.0, -115.0, 400.0, 37.05, -115.0, 90.0, 300.0)
    assert angle_diff_deg(lead, pure) > 5.0


def test_lead_bearing_mirrors_for_the_opposite_crossing():
    pure = bearing_deg(37.0, -115.0, 37.05, -115.0)
    lead = lead_bearing_deg(37.0, -115.0, 400.0, 37.05, -115.0, 270.0, 300.0)
    assert angle_diff_deg(lead, pure) < -5.0


def test_corner_throttle_accelerates_when_slow_and_backs_off_when_fast():
    assert corner_throttle(300.0) > 0.9
    assert corner_throttle(600.0) < 0.4

    # FULL thrust at corner speed, not a mid-range setting. This assertion
    # used to require 0.2-0.9 here, which is the "creep up to corner"
    # behaviour that starved the turn: holding corner through a 4-5g pull
    # needs near-max thrust, and at 394kt the old law gave only 0.90 while the
    # turn bled energy faster than that replaced it. The jet sank from
    # 15686ft/394kt into 300+ steps of RECOVER at g=0.80 and hit the ground --
    # at 155kt its available g was 0.74, below its own 1g stall speed.
    assert corner_throttle(CORNER_SPEED_KT) == 1.0
    assert corner_throttle(CORNER_SPEED_KT - 50.0) == 1.0
    # Above corner it still backs off, monotonically.
    assert (corner_throttle(CORNER_SPEED_KT + 40.0)
            > corner_throttle(CORNER_SPEED_KT + 120.0))
    assert corner_throttle(CORNER_SPEED_KT + 40.0) < 1.0


def test_corner_throttle_is_always_a_valid_throttle():
    for v in (0.0, 120.0, 430.0, 800.0, 2000.0):
        assert 0.0 <= corner_throttle(v) <= 1.0


def test_aim9_envelope_requires_lock_range_fov_and_a_moving_target():
    assert aim9_in_envelope(1.0, 5.0, 300.0, True)
    assert not aim9_in_envelope(1.0, 5.0, 300.0, False), "no lock"
    assert not aim9_in_envelope(0.1, 5.0, 300.0, True), "too close to arm"
    assert not aim9_in_envelope(12.0, 5.0, 300.0, True), "beyond max range"
    assert not aim9_in_envelope(1.0, 40.0, 300.0, True), "outside seeker FOV"
    assert not aim9_in_envelope(1.0, 5.0, 5.0, True), "target under 15kt"


def test_aim9_envelope_is_symmetric_in_bearing_error():
    # Symmetry is the point here; the bound is the FIRING boresight, which is
    # tighter than the seeker FOV -- a lock is not a shot.
    assert aim9_in_envelope(1.0, AIM9_FIRE_FOV_DEG - 1, 300.0, True)
    assert aim9_in_envelope(1.0, -(AIM9_FIRE_FOV_DEG - 1), 300.0, True)
    assert AIM9_FIRE_FOV_DEG < AIM9_FOV_DEG
    assert not aim9_in_envelope(1.0, AIM9_FOV_DEG + 1, 300.0, True)


def test_gun_envelope_is_close_range_and_tightly_aimed():
    assert gun_in_envelope(0.3, 1.0)
    assert not gun_in_envelope(0.3, 10.0), "pipper not on target"
    assert not gun_in_envelope(2.0, 1.0), "out of gun range"


def test_is_lethal_only_inside_the_lethal_radius():
    assert is_lethal(5.0)
    assert not is_lethal(200.0)
    assert is_lethal(30.0, lethal_radius_m=50.0)


BASE = dict(lat=37.0, lon=-115.0, alt_ft=18000.0, agl_ft=12000.0,
            heading_deg=0.0, roll_deg=0.0, pitch_deg=0.0,
            airspeed_kt=430.0, climb_rate_fps=0.0,
            tgt_lat=37.05, tgt_lon=-115.0, tgt_heading_deg=180.0,
            tgt_speed_kt=400.0, tgt_range_nm=3.0, has_lock=False)
NO_THREAT = dict(missile_inbound=False, missile_bearing_deg=None)


def _telem(**over):
    t = dict(BASE)
    t.update(over)
    return t


def test_evade_takes_priority_over_everything():
    p = CombatPilot(callsign="BLUE1")
    cmd = p.step(_telem(tgt_range_nm=1.0, has_lock=True),
                 dict(missile_inbound=True, missile_bearing_deg=90.0))
    assert cmd.state == "EVADE"
    assert cmd.flares > 0, "must release flares while evading"
    assert not cmd.fire_missile, "not shooting while defending"


def test_evade_breaks_hard():
    p = CombatPilot(callsign="BLUE1")
    cmd = p.step(_telem(), dict(missile_inbound=True, missile_bearing_deg=90.0))
    assert cmd.g_cmd > 4.0, "a break turn is a high-g maneuver"


def test_engage_inside_five_miles():
    p = CombatPilot(callsign="BLUE1")
    assert p.step(_telem(tgt_range_nm=3.0), NO_THREAT).state == "ENGAGE"


def test_intercept_beyond_five_miles():
    p = CombatPilot(callsign="BLUE1")
    assert p.step(_telem(tgt_range_nm=7.0), NO_THREAT).state == "INTERCEPT"


def test_fires_missile_only_with_lock_in_envelope():
    p = CombatPilot(callsign="BLUE1")
    assert p.step(_telem(tgt_range_nm=1.0, has_lock=True), NO_THREAT).fire_missile
    assert not p.step(_telem(tgt_range_nm=1.0, has_lock=False), NO_THREAT).fire_missile


def test_fires_gun_when_very_close_and_aligned():
    p = CombatPilot(callsign="BLUE1")
    # target dead ahead (we point north, it is due north of us) and close
    cmd = p.step(_telem(tgt_range_nm=0.3, tgt_heading_deg=0.0,
                        tgt_speed_kt=400.0), NO_THREAT)
    assert cmd.fire_gun


def test_g_command_never_exceeds_available_lift():
    p = CombatPilot(callsign="BLUE1")
    # at 200kt, (200/180)^2 = 1.23g available -- a break turn must be clipped
    cmd = p.step(_telem(airspeed_kt=200.0),
                 dict(missile_inbound=True, missile_bearing_deg=90.0))
    assert cmd.g_cmd <= 1.24, "stall guard must clip the demand"


def test_commands_are_always_within_actuator_limits():
    p = CombatPilot(callsign="BLUE1")
    for rng in (0.1, 0.5, 3.0, 8.0, 20.0):
        for spd in (150.0, 430.0, 700.0):
            cmd = p.step(_telem(tgt_range_nm=rng, airspeed_kt=spd), NO_THREAT)
            assert 0.0 <= cmd.throttle <= 1.0
            assert abs(cmd.roll_rate_dps) <= 213.1
            # MIN_BUNT_G, not 0: unloading below zero g is a legitimate,
            # in-envelope command (the F-15 is stressed to -3g and the CAS
            # g-demand table goes to -4) and is exactly what the flight-path
            # limiter uses to push the nose back down from a steep climb.
            assert MIN_BUNT_G <= cmd.g_cmd <= 9.0




def test_descending_jet_is_commanded_more_g_than_a_level_one():
    """The bug Phase 1 exposed: commanding only 1/cos(bank) holds altitude in a
    coordinated level turn and nowhere else, so both jets walked 9500ft
    downhill in 45s over terrain at ~5400ft."""
    p = CombatPilot(callsign="BLUE1", target_alt_ft=18000.0)
    level = p.step(_telem(alt_ft=18000.0, climb_rate_fps=0.0), NO_THREAT)
    sinking = p.step(_telem(alt_ft=14000.0, climb_rate_fps=-80.0), NO_THREAT)
    assert sinking.g_cmd > level.g_cmd


def test_climb_rate_is_damped_so_altitude_does_not_oscillate():
    p = CombatPilot(callsign="BLUE1", target_alt_ft=18000.0)
    steady = p.step(_telem(alt_ft=17000.0, climb_rate_fps=0.0), NO_THREAT)
    climbing = p.step(_telem(alt_ft=17000.0, climb_rate_fps=+100.0), NO_THREAT)
    assert climbing.g_cmd < steady.g_cmd, "already climbing: ease off"


def test_recovery_floor_scales_with_speed_and_dive_angle():
    """A fixed deck is not enough: the airshow work recorded the jet passing
    4000ft AGL at 673kt in a 42deg dive and hitting the ground anyway, because
    a 5g pull-out at that speed needs ~8000ft of radius."""
    p = CombatPilot(callsign="BLUE1")
    level_slow = p.recovery_floor_ft(300.0, 0.0, 0.0)
    diving_fast = p.recovery_floor_ft(673.0, -42.0, 0.0)
    # 673kt in a 42deg dive computes ~4360ft of recovery room against the
    # 1500ft absolute floor -- the point is that it scales, not the exact ratio.
    assert diving_fast > 3000.0, "a fast steep dive needs real room"
    assert diving_fast > 2.5 * level_slow
    assert level_slow >= 1500.0, "never below the absolute floor"


def test_recovery_floor_includes_the_time_spent_rolling_upright():
    p = CombatPilot(callsign="BLUE1")
    wings_level = p.recovery_floor_ft(500.0, -30.0, 0.0)
    steeply_banked = p.recovery_floor_ft(500.0, -30.0, 120.0)
    assert steeply_banked > wings_level, "rolling upright costs altitude first"


def test_near_the_ground_and_banked_it_rolls_upright_before_pulling():
    """This is the spiral-into-terrain fix: pulling while banked hard just
    turns, and inverted it points the nose further down.

    The roll-to-level command is a DAMPED proportional law toward bank=0, not
    a bang-bang -copysign(MAX_RATE, roll) that reverses at full rate the
    instant roll crosses zero -- that bang-bang law produced a real limit
    cycle in flight (roll oscillating +161/-132/+167deg, never settling) that
    ended in a crash. So a strong but not-yet-saturated error (70deg) is
    expected to command a large, correctly-signed rate without necessarily
    hitting the exact 213.1 cap -- saturation is checked separately at a
    larger error in test_recover_roll_upright_does_not_bang_bang_across_zero.
    """
    p = CombatPilot(callsign="BLUE1")
    cmd = p.step(_telem(agl_ft=800.0, roll_deg=70.0, pitch_deg=-20.0,
                        airspeed_kt=400.0), NO_THREAT)
    assert cmd.state == "RECOVER"
    assert cmd.roll_rate_dps < -100.0, "fast, not a trickle"
    assert cmd.g_cmd == 1.0, "unloaded while rolling"
    assert cmd.throttle == 1.0


def test_near_the_ground_and_upright_it_pulls_hard():
    p = CombatPilot(callsign="BLUE1")
    cmd = p.step(_telem(agl_ft=800.0, roll_deg=5.0, pitch_deg=-20.0,
                        airspeed_kt=400.0), NO_THREAT)
    assert cmd.state == "RECOVER"
    assert cmd.g_cmd > 2.0, "now pull"
    assert not cmd.fire_missile and not cmd.fire_gun


def test_ground_avoidance_outranks_an_incoming_missile():
    p = CombatPilot(callsign="BLUE1")
    cmd = p.step(_telem(agl_ft=600.0, roll_deg=60.0, pitch_deg=-25.0,
                        airspeed_kt=400.0),
                 dict(missile_inbound=True, missile_bearing_deg=90.0))
    assert cmd.state == "RECOVER", "the ground kills faster than the missile"


def test_altitude_terms_never_break_the_actuator_limits():
    p = CombatPilot(callsign="BLUE1", target_alt_ft=18000.0)
    for alt in (2000.0, 9000.0, 18000.0, 40000.0):
        for cr in (-300.0, 0.0, +300.0):
            cmd = p.step(_telem(alt_ft=alt, climb_rate_fps=cr), NO_THREAT)
            # MIN_BUNT_G, not 0: unloading below zero g is a legitimate,
            # in-envelope command (the F-15 is stressed to -3g and the CAS
            # g-demand table goes to -4) and is exactly what the flight-path
            # limiter uses to push the nose back down from a steep climb.
            assert MIN_BUNT_G <= cmd.g_cmd <= 9.0
            assert abs(cmd.roll_rate_dps) <= 213.1


def test_bank_is_limited_to_what_the_wing_can_sustain():
    """The second crash: at 313kt with the target 178deg behind, the pilot
    commanded the full 80deg bank, which needs 5.76g to hold altitude while
    only 3.02g was available. The jet descended through the turn and hit
    terrain. Limiting g alone does not prevent that -- bank must be limited."""
    p = CombatPilot(callsign="BLUE1")
    slow = p.max_sustainable_bank_deg(313.0)
    fast = p.max_sustainable_bank_deg(600.0)
    assert slow < 70.0, "313kt cannot sustain a 70deg+ bank"
    assert fast > slow, "more speed buys more bank"


def test_sustainable_bank_is_zero_near_stall():
    p = CombatPilot(callsign="BLUE1")
    assert p.max_sustainable_bank_deg(180.0) == 0.0, "1g available: no banked turn"
    assert p.max_sustainable_bank_deg(100.0) == 0.0


def test_a_slow_jet_chasing_a_target_behind_it_does_not_command_a_doomed_bank():
    p = CombatPilot(callsign="BLUE1", target_alt_ft=15000.0)
    # target almost directly behind, at the speed that crashed it
    cmd = p.step(_telem(alt_ft=15000.0, airspeed_kt=313.0, heading_deg=0.0,
                        tgt_lat=36.95, tgt_lon=-115.0, tgt_heading_deg=0.0,
                        tgt_range_nm=10.0), NO_THREAT)
    sustainable = p.max_sustainable_bank_deg(313.0)
    # the commanded roll rate must be driving toward a sustainable bank, so at
    # zero current roll it cannot exceed K_BANK_TO_RATE * that limit
    assert abs(cmd.roll_rate_dps) <= 2.5 * sustainable + 1e-6


def test_a_slow_jet_stops_turning_and_rebuilds_energy():
    """A turning fight is an energy fight. Confirmed live: holding 63-74deg of
    bank decayed 418kt to 296kt at full throttle, and available g fell with it
    until the bank needed more lift than existed.

    RESET extends wings-level (bank target 0), not a shallow constant bank
    toward the target's bearing -- a fixed bank is a slow circle, not an
    extend. Confirmed live: a -25deg "extend" toward a target 60-95deg off
    the nose held the jet in a wide closed loop for 30+ seconds, with range
    climbing from 11nm to 17nm and never closing."""
    p = CombatPilot(callsign="BLUE1", target_alt_ft=15000.0)
    cmd = p.step(_telem(alt_ft=15000.0, agl_ft=9000.0, airspeed_kt=180.0,
                        roll_deg=40.0,
                        tgt_lat=37.05, tgt_lon=-114.90), NO_THREAT)
    assert cmd.state == "RESET"
    assert cmd.throttle == 1.0
    assert cmd.roll_rate_dps < 0, "rolling back toward level from +40deg"
    assert not cmd.fire_missile


def test_energy_state_has_hysteresis_so_it_does_not_chatter():
    p = CombatPilot(callsign="BLUE1", target_alt_ft=15000.0)
    p.step(_telem(airspeed_kt=300.0), NO_THREAT)          # enters RESET
    mid = p.step(_telem(airspeed_kt=360.0), NO_THREAT)     # between the bands
    assert mid.state == "RESET", "must not re-enter the fight at 360kt"
    back = p.step(_telem(airspeed_kt=420.0), NO_THREAT)    # above recover band
    assert back.state in ("ENGAGE", "INTERCEPT")


def test_lead_bearing_does_not_blow_up_at_long_range_low_closure():
    """Confirmed live: at 14nm with a near-parallel target heading, the
    unclamped t_go extrapolated the target ~26km along its track -- an
    offset roughly as large as the range itself -- producing a lead bearing
    ~275deg off the true one. CombatPilot then chased that phantom bearing
    with full roll authority and nearly flew into terrain (bank swinging
    +179 -> -178 -> +167deg over 4 seconds)."""
    own_lat, own_lon, own_spd = 37.20, -115.90, 360.0
    tgt_lat, tgt_lon = 37.30, -115.75
    pure = bearing_deg(own_lat, own_lon, tgt_lat, tgt_lon)
    for tgt_hdg in (238, 239, 240, 245, 250):
        lead = lead_bearing_deg(own_lat, own_lon, own_spd, tgt_lat, tgt_lon,
                                tgt_hdg, 360.0)
        assert abs(angle_diff_deg(lead, pure)) < 30.0, (
            "lead bearing must stay near the true bearing at long range, "
            "not swing across the whole compass")


def test_lead_survives_where_it_matters_and_yields_to_boresight_up_close():
    """Lead must survive at intercept ranges, where cutting the corner is the
    whole point -- but NOT inside the endgame. This used to assert a real lead
    angle at 0.6nm; two runs showed that is exactly what makes the firing gate
    unreachable, because the steering loop settles with the nose on the aim
    point (best pointing 23deg, 26 locks, zero shots). See
    test_lead_tapers_to_pure_pursuit_so_the_firing_gate_is_reachable."""
    # Mid-range crossing target: lead pursuit is still doing real work.
    mid = offset_point(37.0, -115.0, 0.0, 2.5 * 1852.0)
    pure = bearing_deg(37.0, -115.0, mid[0], mid[1])
    lead = lead_bearing_deg(37.0, -115.0, 400.0, mid[0], mid[1], 90.0, 400.0)
    assert abs(angle_diff_deg(lead, pure)) > 3.0

    # Knife range: pure pursuit, so a shot is actually possible.
    close = offset_point(37.0, -115.0, 0.0, 0.6 * 1852.0)
    pure_c = bearing_deg(37.0, -115.0, close[0], close[1])
    lead_c = lead_bearing_deg(37.0, -115.0, 400.0, close[0], close[1], 90.0, 400.0)
    assert abs(angle_diff_deg(lead_c, pure_c)) == 0.0


def test_recover_roll_upright_does_not_bang_bang_across_zero():
    """The bang-bang law -copysign(MAX_RATE, roll) reverses at full rate the
    INSTANT roll crosses zero. Confirmed live this produced a persistent limit
    cycle through a real recovery (roll +161 -> +100 -> +35 -> +98 -> -132 ->
    +102 -> -168 -> +167deg over ~5s, never settling) that ended in a
    belly-flop crash. The fix must actually DAMP as roll approaches zero."""
    p = CombatPilot(callsign="BLUE1", target_alt_ft=13000.0)
    # deep in the ground-recovery envelope, banked hard
    far = p.step(_telem(agl_ft=500.0, roll_deg=100.0, pitch_deg=-10.0,
                        airspeed_kt=300.0), NO_THREAT)
    near = p.step(_telem(agl_ft=500.0, roll_deg=10.0, pitch_deg=-10.0,
                         airspeed_kt=300.0), NO_THREAT)
    assert far.state == "RECOVER" and near.state == "RECOVER"
    # closer to level must command a SMALLER rate, not the same max rate --
    # that is what lets it actually converge instead of oscillating forever.
    assert abs(near.roll_rate_dps) < abs(far.roll_rate_dps)
    # Still SATURATED far from level -- but at the combat roll-rate cap, not
    # the CAS's full 213deg/s. Probed live, commanding that much rate rolled
    # the jet into coupled departure (p=206deg/s, inverted at +141deg, Nz
    # reading 22.8g against a g_cmd of 1-6.6). Saturating is the point here;
    # the magnitude is bounded by what the airframe survives.
    assert abs(far.roll_rate_dps) == MAX_COMBAT_ROLL_RATE_DPS
    assert MAX_COMBAT_ROLL_RATE_DPS < CAS_ROLL_RATE_FULL_DPS


def test_turn_direction_commits_near_180deg_bearing_ambiguity():
    """A target directly astern is equally reachable turning either way, and
    angle_diff_deg's sign there is essentially sub-degree measurement noise.
    Confirmed live: bearing_err flickered +178/-179 on a target sitting almost
    dead astern, flipping the bank target +80/-80deg every 20Hz step and
    producing a real, sustained oscillation that bled 3200ft of altitude.

    The deadband is around the +-180 BOUNDARY, not around zero: a clearly
    signed error (e.g. target 45deg off the nose) must still re-derive
    direction normally, so the first call here establishes a real direction
    from an unambiguous geometry before the ambiguous headings are fed in.
    """
    p = CombatPilot(callsign="BLUE1", target_alt_ft=15000.0)
    # Establish a clear rightward turn from an unambiguous 45deg-right target.
    p.step(_telem(heading_deg=0.0, tgt_lat=37.0001, tgt_lon=-114.9999,
                  tgt_heading_deg=180.0), NO_THREAT)
    assert p._turn_dir == 1.0
    # Now feed a near-180 bearing error that flickers sign call to call, as
    # sub-degree telemetry noise would. The COMMITTED direction must not flip.
    signs_seen = set()
    for hdg in (178.0, -179.0, 179.5, -178.5, 179.9, -179.9):
        cmd = p.step(_telem(heading_deg=hdg, tgt_lat=37.0001, tgt_lon=-115.0,
                            tgt_heading_deg=180.0), NO_THREAT)
        if abs(cmd.roll_rate_dps) > 1.0:
            signs_seen.add(cmd.roll_rate_dps > 0)
    assert len(signs_seen) <= 1, (
        "turn direction must not flip sign near the +-180deg ambiguity")


def test_engage_pulls_hard_g_not_just_enough_to_hold_altitude():
    """This is a fighter turning to engage, not an airliner holding altitude.
    Confirmed live: commanding only level_turn_g (1/cos(bank), the bare
    minimum to avoid descending) produced slow, cautious turns logging
    g=1.0-2.0 against an available envelope up to 9g -- much wider turn radius
    and much slower to point at the target than the airframe allows."""
    p = CombatPilot(callsign="BLUE1", target_alt_ft=15000.0)
    cmd = p.step(_telem(alt_ft=15000.0, roll_deg=45.0, airspeed_kt=450.0,
                        tgt_range_nm=3.0), NO_THREAT)
    avail = p.g_available(450.0)
    level_turn_g = 1.0 / math.cos(math.radians(45.0))
    assert cmd.g_cmd > level_turn_g + 1.0, (
        "must pull well beyond the bare altitude-hold minimum")
    assert cmd.g_cmd > 0.7 * min(avail, 9.0), "must use most of the envelope"


def test_engage_uses_full_throttle_not_a_cruise_setpoint():
    """A fighter closing for a kill does not manage toward a cruise speed
    mid-turn -- that throttle policy belongs to RESET (energy management),
    not to an active engagement."""
    p = CombatPilot(callsign="BLUE1", target_alt_ft=15000.0)
    fast = p.step(_telem(airspeed_kt=600.0, tgt_range_nm=3.0), NO_THREAT)
    slow = p.step(_telem(airspeed_kt=350.0, tgt_range_nm=3.0), NO_THREAT)
    assert fast.throttle == 1.0
    assert slow.throttle == 1.0


def test_reset_is_truly_unloaded_not_altitude_corrected():
    """RESET's g_cmd must be a bare 1.0, not 1.0 plus the altitude-hold
    correction term used everywhere else. Confirmed live: adding that term
    kept demanding small climb/descent corrections the entire time, which is
    directly counter to an extend -- any g above the true minimum induces
    drag. The result was a monotonic speed bleed from 280kt to 144kt over 78
    real seconds that never recovered, because the "extend" was still
    quietly loaded the whole time.

    RESET still gets no altitude CORRECTION term -- that is what this test
    guards. It is not exempt from the flight-path LIMITER, though: a bare
    path-holding g at gamma=+60deg faithfully holds +60deg of climb, which
    flew the jet from 33657ft to 39671ft at 157kt. So the unloaded command
    stands whenever the flight path is acceptable, and is capped when it is
    not."""
    p = CombatPilot(callsign="BLUE1", target_alt_ft=15000.0)
    # Level flight: RESET is unloaded, and the limiter does not touch it.
    low = p.step(_telem(alt_ft=8000.0, airspeed_kt=180.0), NO_THREAT)
    high = p.step(_telem(alt_ft=20000.0, airspeed_kt=180.0), NO_THREAT)
    assert low.state == "RESET" and high.state == "RESET"
    assert low.g_cmd == 1.0
    assert high.g_cmd == 1.0, "level and above target: still an unloaded extend"

    # Climbing hard while already above the target IS capped -- the extend
    # must not become a zoom climb.
    climbing = p.step(_telem(alt_ft=20000.0, airspeed_kt=180.0,
                             climb_rate_fps=200.0), NO_THREAT)
    assert climbing.state == "RESET" and climbing.g_cmd < 1.0


def test_reset_is_hard_capped_in_duration():
    """Without a cap, a wide hysteresis band let RESET run for 30-90+ real
    seconds straight in live testing -- either slow-circling (before the
    wings-level fix) or ballooning on leftover vertical momentum while
    decelerating (after it), and the geometry moved so far in the meantime
    that the fight never converged. A committed pilot takes the unload it
    needs and gets back to business."""
    p = CombatPilot(callsign="BLUE1", target_alt_ft=15000.0)
    telem_slow = _telem(airspeed_kt=100.0)  # stays well under the floor
    for _ in range(RESET_MAX_STEPS_TEST + 1):
        cmd = p.step(telem_slow, NO_THREAT)
        assert cmd.state == "RESET"
    # one more step past the cap must exit RESET even though airspeed never
    # crossed ENERGY_RECOVER_KT
    cmd = p.step(telem_slow, NO_THREAT)
    assert cmd.state != "RESET"


def test_engage_slows_to_corner_when_the_nose_is_off_target():
    """Confirmed live: full throttle throughout an engagement ran the jet to
    ~690kt, ~190kt past corner. Bank pinned at the -80deg limit at 5g, and
    bearing error STUCK at -116..-131deg while range GREW 2.07 -> 3.03nm --
    turn radius at 690kt/5g is ~1.4nm, so a target 2nm away is geometrically
    unreachable. Slow down to out-turn it; speed up only once pointed."""
    p = CombatPilot(callsign="BLUE1", aggression=1.0)
    fast_and_off = _telem(airspeed_kt=690.0, tgt_lat=37.30, tgt_lon=-115.70)
    cmd = p.step(fast_and_off, NO_THREAT)
    brg = angle_diff_deg(bearing_deg(fast_and_off["lat"], fast_and_off["lon"],
                                     fast_and_off["tgt_lat"], fast_and_off["tgt_lon"]),
                         fast_and_off["heading_deg"])
    if abs(brg) > 27.5:                     # nose off target
        assert cmd.throttle < 1.0, "must slow toward corner to out-turn"
        assert cmd.throttle == corner_throttle(690.0)
    # Corner logic itself: fast -> pull power, slow -> full power.
    assert corner_throttle(690.0) < corner_throttle(500.0) <= corner_throttle(300.0)
    assert corner_throttle(300.0) == 1.0


def test_roll_snaps_hard_enough_to_actually_fight():
    """Roll is the inner loop of a turning fight: g only turns the aircraft
    once it is banked, and the flight-path limiter rightly refuses to let a
    hard pull at shallow bank become a climb. Confirmed live at the old 2.5
    gain -- 80deg commanded vs -59deg actual rolled at 2.3deg/s while the
    target sat 83deg off the nose at 1.07nm and the pull stayed at 1.8g."""
    p = CombatPilot(callsign="BLUE1")
    rate = p._roll_to(-80.0, -59.0, 500.0)      # the exact logged geometry
    assert abs(rate) > 100.0, rate               # must be a real roll demand
    assert abs(rate) <= 213.1                    # never beyond CAS authority
    # Still proportional: it must damp as it converges, not bang-bang.
    assert abs(p._roll_to(-80.0, -78.0, 500.0)) < abs(rate)
    assert p._roll_to(0.0, 0.0, 500.0) == 0.0


def test_intercept_reference_altitude_follows_the_bandit():
    """The controller used to defend its ENTRY altitude forever. With bandits
    orbiting 1600ft above where the jet took the stick, that made every
    altitude term brake the turn for doing the one thing an intercept must do.
    Reproduced from the log: hard_pull 4.25g - 2.0g alt clamp = the exact
    2.25g commanded while the target sat 110deg off and the range opened."""
    p = CombatPilot(callsign="BLUE1", target_alt_ft=14900.0)
    p.step(_telem(alt_ft=14900.0, tgt_alt_ft=16500.0), NO_THREAT)
    assert p.target_alt_ft == 16500.0, "must fight at the bandit's altitude"

    # A missing or nonsense target altitude must not wipe out the reference.
    p.step(_telem(alt_ft=14900.0, tgt_alt_ft=0.0), NO_THREAT)
    assert p.target_alt_ft == 16500.0


def test_lead_does_not_collapse_at_dogfight_range():
    """The exact geometry that stopped the jet turning: 1.89nm, 400kt bandit
    crossing. The old position-extrapolation lead pushed the aim point 1.6nm
    along the target's track -- nearly the whole range -- so the aim landed
    near our own nose, desired_bank went to ~0 and the jet rolled level with
    the target 40deg off (probed live: aileron -0.027/-0.010/+0.016)."""
    own_lat, own_lon = 37.26, -115.80
    # Bandit ~1.9nm away on a crossing heading.
    tgt_lat, tgt_lon = offset_point(own_lat, own_lon, 40.0, 1.89 * 1852.0)
    los = bearing_deg(own_lat, own_lon, tgt_lat, tgt_lon)
    aim = lead_bearing_deg(own_lat, own_lon, 465.0,
                           tgt_lat, tgt_lon, 45.0, 400.0)
    # Lead must be a bounded offset from the LOS, never a wild new direction.
    assert abs(angle_diff_deg(aim, los)) < 90.0

    # A head-on / tail-on bandit needs no lead at all.
    for hdg in (los, (los + 180.0) % 360.0):
        straight = lead_bearing_deg(own_lat, own_lon, 465.0,
                                    tgt_lat, tgt_lon, hdg, 400.0)
        assert abs(angle_diff_deg(straight, los)) < 1e-6

    # The lead GEOMETRY is range-independent -- it comes from aspect and the
    # speed ratio, not from a time-to-go extrapolation, which is the bug this
    # law replaced. (The CAP on it does taper with range, to hand the endgame
    # over to pure pursuit; at this aspect the raw angle is under both caps,
    # so the underlying geometry shows through unchanged.)
    far_lat, far_lon = offset_point(own_lat, own_lon, 40.0, 10.0 * 1852.0)
    far_aim = lead_bearing_deg(own_lat, own_lon, 465.0,
                               far_lat, far_lon, 45.0, 400.0)
    far_los = bearing_deg(own_lat, own_lon, far_lat, far_lon)
    assert abs(angle_diff_deg(aim, los)
               - angle_diff_deg(far_aim, far_los)) < 0.5

    # A crossing bandit must pull the aim toward its direction of travel.
    right = lead_bearing_deg(own_lat, own_lon, 465.0,
                             tgt_lat, tgt_lon, (los + 90.0) % 360.0, 400.0)
    left = lead_bearing_deg(own_lat, own_lon, 465.0,
                            tgt_lat, tgt_lon, (los - 90.0) % 360.0, 400.0)
    # A crossing bandit pulls the aim toward its direction of travel, but
    # never further than MAX_LEAD_DEG -- the seeker has to keep it in view.
    assert 0.0 < angle_diff_deg(right, los) <= MAX_LEAD_DEG + 1e-9
    assert -MAX_LEAD_DEG - 1e-9 <= angle_diff_deg(left, los) < 0.0
    assert MAX_LEAD_DEG < AIM9_FOV_DEG, "aim must stay inside the seeker cone"


def test_climb_damping_does_not_fight_a_needed_climb():
    """Run 14 step 60, worked out by hand and matching the log exactly: the
    jet was 1535ft BELOW the bandit (+0.38g of altitude demand) while the
    123fps climb closing that very gap produced -2.46g of damping. Net went
    negative, g_cmd floored at 0.30 and the turn died with the pull."""
    p = CombatPilot(callsign="BLUE1", aggression=1.0)
    climbing_to_reach = _telem(alt_ft=14965.0, tgt_alt_ft=16500.0,
                               climb_rate_fps=123.0, airspeed_kt=479.0,
                               roll_deg=-38.0)
    cmd = p.step(climbing_to_reach, NO_THREAT)
    assert cmd.g_cmd > 1.5, cmd.g_cmd   # a real pull, not the 0.30 floor

    # A climb NOT wanted is still damped hard -- this must not become a
    # licence to zoom once the jet is already at the bandit's altitude.
    runaway = _telem(alt_ft=16500.0, tgt_alt_ft=16500.0,
                     climb_rate_fps=150.0, airspeed_kt=479.0, roll_deg=0.0)
    p2 = CombatPilot(callsign="BLUE1", aggression=1.0)
    level = _telem(alt_ft=16500.0, tgt_alt_ft=16500.0,
                   climb_rate_fps=0.0, airspeed_kt=479.0, roll_deg=0.0)
    assert p2.step(runaway, NO_THREAT).g_cmd < p2.step(level, NO_THREAT).g_cmd


def test_hard_turn_may_descend_when_there_is_altitude_to_spend():
    """Run 15 was a real turning fight but a stalemate at 1.42nm: the
    level-flight bank cap held bank at ~78deg, the jet climbed 14874 ->
    17941ft against a bandit at 16500, and the flight-path limiter then
    throttled it to 3.3g when 5.9g was available. Capping bank for LEVEL
    flight forced the climb that cost the turn."""
    p = CombatPilot(callsign="BLUE1")
    level_cap = p.max_sustainable_bank_deg(438.0)
    assert level_cap < MAX_BANK_DEG, "premise: level flight caps below the max"

    # Sitting just past the level-flight cap, where the two regimes differ
    # (from wings-level both saturate the rate clamp and prove nothing).
    at_bank = -(level_cap + 1.0)

    # High up, the cap is lifted: keep rolling toward the full bank.
    high = p._roll_to(-MAX_BANK_DEG, at_bank, 438.0, agl_ft=12000.0)
    assert high < -1.0, high          # still commanding MORE bank

    # Near the deck the cap applies and it rolls back OUT to what it can hold.
    low = p._roll_to(-MAX_BANK_DEG, at_bank, 438.0, agl_ft=1000.0)
    assert low > 0.0, low             # unwinding toward level-sustainable

    # Unknown AGL must stay conservative, not silently unlock full bank.
    assert p._roll_to(-MAX_BANK_DEG, at_bank, 438.0) == low


def test_engage_never_unloads_itself_into_a_no_turn_hole():
    """Reproduced at the same point in three consecutive runs (steps 60-80:
    g=0.30, roll -8 to -13deg). Bank collapses -> turn_fraction collapses ->
    hard_pull drops to ~1.7g -> alt_correction hits its -2.0g clamp -> the sum
    floors at 0.3g. At 0.3g there is no turn, so the bank never comes back and
    the jet sits in the hole while the bandit walks away."""
    p = CombatPilot(callsign="BLUE1", aggression=1.0)
    hole = _telem(alt_ft=15048.0, tgt_alt_ft=16500.0, climb_rate_fps=250.0,
                  airspeed_kt=475.0, roll_deg=-8.0)
    cmd = p.step(hole, NO_THREAT)
    # Out of the hole by a wide margin. clamp_flight_path still runs after
    # this and may bunt below path_hold_g to bring a steep climb down -- that
    # is the limiter doing its job, not the collapse this guards against.
    assert cmd.g_cmd > 0.8, cmd.g_cmd

    # With the flight path already acceptable, the floor applies in full.
    settled = _telem(alt_ft=16500.0, tgt_alt_ft=16500.0, climb_rate_fps=0.0,
                     airspeed_kt=475.0, roll_deg=-8.0)
    p2 = CombatPilot(callsign="BLUE1", aggression=1.0)
    assert (p2.step(settled, NO_THREAT).g_cmd
            >= path_hold_g(settled, settled["roll_deg"]) - 1e-9)


def test_never_commands_a_departure_roll_rate():
    """Probed live mid-fight at the old 213deg/s clamp: p reached 3.598 rad/s
    (206deg/s), the jet was inverted at roll=+141deg, and accelerations/Nz
    read 22.8g then 12.0, 10.6, 10.9 -- while g_cmd was between 1 and 6.6.
    The airframe was departing, not tracking. The g-meter spikes are the
    inertial coupling of a fast roll at alpha, and the snap through inverted
    is the 'roll collapse' in every run's log."""
    p = CombatPilot(callsign="BLUE1")
    # Huge bank errors, every regime: the demand must always stay bounded.
    for roll in (-180.0, -90.0, 0.0, 90.0, 179.0):
        for spd in (250.0, 450.0, 650.0):
            for agl in (1000.0, 20000.0):
                r = p._roll_to(85.0, roll, spd, agl_ft=agl)
                assert abs(r) <= MAX_COMBAT_ROLL_RATE_DPS, (roll, spd, agl, r)


def test_does_not_shoot_at_the_lead_point():
    """Run 19 fired two AIM-9s at a true bearing error of -40deg: after the
    18deg lead cap the pilot's steering error read ~-22deg and passed the
    27.5deg FOV test, so both rounds left the rail with the bandit outside
    the seeker's view. 14 locks, 2 launches, 0 damage, 0 kills."""
    p = CombatPilot(callsign="BLUE1")
    own_lat, own_lon = 37.26, -115.80
    # Bandit dead ahead in RANGE but 40deg off our nose, crossing hard so the
    # lead angle saturates and pulls the aim point back toward the nose.
    tgt_lat, tgt_lon = offset_point(own_lat, own_lon, 40.0, 2.0 * 1852.0)
    telem = _telem(lat=own_lat, lon=own_lon, heading_deg=0.0,
                   tgt_lat=tgt_lat, tgt_lon=tgt_lon,
                   tgt_heading_deg=130.0, tgt_speed_kt=400.0,
                   airspeed_kt=470.0, has_lock=True)
    los_err = angle_diff_deg(bearing_deg(own_lat, own_lon, tgt_lat, tgt_lon),
                             telem["heading_deg"])
    assert abs(los_err) > AIM9_FOV_DEG, "premise: target outside seeker FOV"
    assert not p.step(telem, NO_THREAT).fire_missile, "must not shoot blind"


def test_altitude_correction_cannot_eat_the_hard_turn():
    """Run 20 step 280, reproduced by hand: hard_pull 4.78g, jet 700ft above
    the bandit climbing 199fps, correction clamped at -2.0g, g_cmd 2.78 --
    exactly path_hold_g at 70deg bank. A path-holding turn, not a fighting
    one, and the bearing error stayed at -46..-58deg from 2.10nm to 0.76nm,
    always outside the seeker's 27.5deg FOV."""
    p = CombatPilot(callsign="BLUE1", aggression=1.0)
    t = _telem(alt_ft=17208.0, tgt_alt_ft=16500.0, climb_rate_fps=199.0,
               airspeed_kt=419.0, roll_deg=-70.0)
    cmd = p.step(t, NO_THREAT)
    # Must be a real pull, meaningfully above merely holding the flight path.
    assert cmd.g_cmd > path_hold_g(t, t["roll_deg"]) + 0.5, cmd.g_cmd


def test_banks_steeply_enough_that_g_turns_instead_of_climbing():
    """Run 21: raw bearing error ~-35deg became ~-17deg to the aim point after
    the 18deg lead cap, and at the old gain of 2.0 that commanded only 34deg
    of bank. A 4g pull at 45deg puts 2.8g straight into the vertical -- the
    jet climbed 14887 -> 18025ft, went above the 16500ft bandit, and the
    altitude terms braked it to 0.9-1.4g while the error grew to -56deg."""
    p = CombatPilot(callsign="BLUE1", aggression=1.0)
    # Modest steering error must still produce a genuinely steep bank.
    p._turn_dir = -1.0
    bank = -min(MAX_BANK_DEG, 17.0 * 4.0)
    assert abs(bank) > 60.0, bank
    # At that bank the vertical share of a hard pull is small.
    assert 4.0 * math.cos(math.radians(abs(bank))) < 2.0


def test_does_not_waste_rails_on_long_off_boresight_shots():
    """Run 22 launched all three AIM-9s at 3.64nm, 3.80nm and 2.10nm with
    21-31deg of LOS error -- legal by the old envelope, and all three missed:
    52 locks, 3 launches, 1% damage, 0 kills. Four rounds, three bandits."""
    assert not aim9_in_envelope(3.64, -31.0, 400.0, True)
    assert not aim9_in_envelope(3.80, -21.0, 400.0, True)
    assert not aim9_in_envelope(2.10, -23.0, 400.0, True)
    # The close, near-boresight shot the same run repeatedly had available
    # (it closed to 0.11nm) is still taken.
    assert aim9_in_envelope(1.20, -8.0, 400.0, True)


def test_does_not_accelerate_into_a_close_range_overshoot():
    """Run 23 held -23..-26deg at 1.1nm -- inside the seeker cone, never
    inside the 12deg firing gate -- and stayed at FULL throttle because 25deg
    counts as nose-on. It drove through the overshoot: bearing error snapped
    to -121deg and the recovery cost 6000ft of climb and 120kt."""
    p = CombatPilot(callsign="BLUE1", aggression=1.0)

    def at(range_nm, brg_off_deg):
        tgt = offset_point(37.26, -115.80, brg_off_deg, range_nm * 1852.0)
        return _telem(lat=37.26, lon=-115.80, heading_deg=0.0,
                      tgt_lat=tgt[0], tgt_lon=tgt[1], tgt_heading_deg=0.0,
                      tgt_speed_kt=400.0, airspeed_kt=520.0,
                      tgt_range_nm=range_nm)   # the pilot reads range here

    # Close, nose still well off the firing gate: manage speed, do not add it.
    assert p.step(at(1.1, 25.0), NO_THREAT).throttle == corner_throttle(520.0)
    # Close and PERFECTLY aimed: still manage speed. Being pointed used to
    # license full power, which flew the jet through the bandit -- the lock
    # came up and then died as the target left the seeker cone astern.
    assert p.step(at(1.1, 2.0), NO_THREAT).throttle == corner_throttle(520.0)
    # Far away with the same angle: speed buys closure, so use it.
    assert p.step(at(4.5, 5.0), NO_THREAT).throttle == 1.0


def test_lead_tapers_to_pure_pursuit_so_the_firing_gate_is_reachable():
    """Runs 23 and 24 both stabilised at a best pointing of 23deg and never
    got inside the 12deg firing gate -- 26 locks, zero shots. The steering
    loop settles with the nose on the AIM point, so the steady-state LOS error
    is about the lead angle itself: a fixed 18deg cap and a 12deg boresight
    gate deadlock each other."""
    own_lat, own_lon = 37.26, -115.80

    def lead_at(range_nm):
        t = offset_point(own_lat, own_lon, 30.0, range_nm * 1852.0)
        los = bearing_deg(own_lat, own_lon, t[0], t[1])
        aim = lead_bearing_deg(own_lat, own_lon, 470.0, t[0], t[1], 120.0, 400.0)
        return abs(angle_diff_deg(aim, los))

    # Far out: full lead, so the intercept still cuts the corner.
    assert lead_at(4.0) == MAX_LEAD_DEG
    # Closing: monotonically tapering.
    assert lead_at(3.0) > lead_at(2.0) > lead_at(1.5) > lead_at(1.0)
    # Inside the endgame: pure pursuit, nose on the bandit, so a LOS error
    # under the firing gate is actually achievable.
    assert lead_at(1.0) == 0.0
    assert lead_at(0.6) == 0.0
    assert lead_at(1.5) < AIM9_FIRE_FOV_DEG


def test_descends_to_a_high_bandit_but_never_dives_at_the_ground():
    """Being stranded ABOVE the bandit blinds the radar, not just the turn.
    Probed live at textbook geometry (1.17nm, 9deg off, both firing gates
    satisfied): the jet sat at 24091ft against a 16500ft bandit, and
    armament.contact -- still RED1 -- had gone STALE at 3.46nm because a ~45deg
    depression angle puts the target outside the radar scan volume. fox2's
    search() then fails FOV_check on that stale bearing, srm.Tgt stays NIL and
    the seeker never leaves 0.

    The descent that fixes that must not chase a bandit into the dirt: run 33
    followed one to the deck and crashed at agl=2ft, spd=19kt."""
    high = dict(airspeed_kt=400.0, climb_rate_fps=0.0, alt_ft=24000.0,
                roll_deg=0.0, agl_ft=18000.0)
    assert allowed_gamma_deg(high, 16500.0) == MIN_ALLOWED_GAMMA_DEG
    assert MIN_ALLOWED_GAMMA_DEG < 0.0, "must be able to come back down"

    near_deck = dict(high, agl_ft=1500.0)
    assert allowed_gamma_deg(near_deck, 16500.0) == 0.0, "no dive at the ground"

    # Below the bandit, a real climb is still allowed regardless of AGL.
    low = dict(high, alt_ft=15000.0, agl_ft=1500.0)
    assert allowed_gamma_deg(low, 16500.0) > 0.0


def test_never_flies_itself_below_its_own_1g_stall_speed():
    """Runs 33 and 34 both ended in terrain the same way: the jet bled energy
    in the turn until its available g fell BELOW 1.0, at which point no
    command could recover it -- 300+ steps of RECOVER at g=0.80, wings level,
    descending from 6540ft into the ground at 150-210kt.

    The guard is thrust: whenever speed is at or under corner, every state
    that manages energy must be asking for all of it."""
    p = CombatPilot(callsign="BLUE1")
    # The regime that killed it: slow, low, nose off target.
    assert corner_throttle(155.0) == 1.0
    assert corner_throttle(394.0) == 1.0
    # And the physics that made it unrecoverable, so the premise is explicit.
    assert p.g_available(155.0) < 1.0
    assert p.g_available(CORNER_SPEED_KT) > 4.0


def test_does_not_blow_past_a_slow_bandit_in_knife_range():
    """Logged live: RED1 bled down to 158kt while we held 500kt. Corner speed
    IS 500, so "full thrust at or below corner" asked for full afterburner in
    knife range and the bearing error was -155deg at 0.52nm -- range gate
    satisfied, shot thrown away on closure. 17 locks, 0 shots."""
    p = CombatPilot(callsign="BLUE1", aggression=1.0)

    def at(range_nm, tgt_kt, own_kt):
        tgt = offset_point(37.26, -115.80, 2.0, range_nm * 1852.0)
        return _telem(lat=37.26, lon=-115.80, heading_deg=0.0,
                      tgt_lat=tgt[0], tgt_lon=tgt[1], tgt_heading_deg=0.0,
                      tgt_speed_kt=tgt_kt, airspeed_kt=own_kt,
                      tgt_range_nm=range_nm)

    # Fast us, slow bandit, knife range: chop the power, do not fly past.
    assert p.step(at(0.9, 158.0, 500.0), NO_THREAT).throttle < 0.2
    # Same geometry but WE are slow: full thrust, so the energy spiral that
    # flew two runs into the ground cannot come back.
    assert p.step(at(0.9, 158.0, 200.0), NO_THREAT).throttle == 1.0
    # Co-speed bandit in knife range still gets full thrust to stay with it.
    assert p.step(at(0.9, 450.0, 450.0), NO_THREAT).throttle == 1.0
