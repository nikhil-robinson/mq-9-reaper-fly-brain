import math
from fly_brain.combat import (
    AIM9_FOV_DEG,
    CORNER_SPEED_KT,
    CombatPilot,
    RESET_MAX_STEPS as RESET_MAX_STEPS_TEST,
    aim9_in_envelope,
    corner_throttle,
    gun_in_envelope,
    is_lethal,
    lead_bearing_deg,
)
from fly_brain.geo import angle_diff_deg, bearing_deg


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
    assert 0.2 < corner_throttle(CORNER_SPEED_KT) < 0.9


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
    assert aim9_in_envelope(1.0, AIM9_FOV_DEG - 1, 300.0, True)
    assert aim9_in_envelope(1.0, -(AIM9_FOV_DEG - 1), 300.0, True)
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
            assert 0.0 < cmd.g_cmd <= 9.0




def test_nose_down_jet_is_commanded_more_g_than_a_level_one():
    """The bug Phase 1 exposed: commanding only 1/cos(bank) holds altitude in a
    coordinated level turn and nowhere else, so both jets walked 9500ft
    downhill in 45s over terrain at ~5400ft. Superseded by pitch-hold (ported
    from f15_airshow.py's hold_nose(), the only proven working hard-turn law
    in this codebase, after four altitude/climb-rate/gamma-based attempts
    each failed or introduced a new failure mode -- see
    PITCH_RECOVER_G_PER_DEG's definition for the full history): the pitch
    error itself, not altitude or climb rate, drives the correction."""
    p = CombatPilot(callsign="BLUE1")
    level = p.step(_telem(pitch_deg=0.0), NO_THREAT)
    nose_down = p.step(_telem(pitch_deg=-20.0), NO_THREAT)
    assert nose_down.g_cmd > level.g_cmd


def test_nose_up_jet_is_commanded_less_g_than_a_level_one():
    p = CombatPilot(callsign="BLUE1")
    level = p.step(_telem(pitch_deg=0.0), NO_THREAT)
    nose_up = p.step(_telem(pitch_deg=+20.0), NO_THREAT)
    assert nose_up.g_cmd < level.g_cmd, "already pitched up: ease off"


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


def test_pitch_hold_never_breaks_the_actuator_limits():
    p = CombatPilot(callsign="BLUE1")
    for alt in (2000.0, 9000.0, 18000.0, 40000.0):
        for pitch in (-60.0, 0.0, +60.0):
            cmd = p.step(_telem(alt_ft=alt, pitch_deg=pitch), NO_THREAT)
            assert 0.0 < cmd.g_cmd <= 9.0
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


def test_lead_bearing_capped_extrapolation_still_leads_a_close_crossing_target():
    """The cap must not defeat lead pursuit at ranges where it actually
    matters -- a close, fast crossing target still needs a real lead angle."""
    pure = bearing_deg(37.0, -115.0, 37.01, -115.0)
    lead = lead_bearing_deg(37.0, -115.0, 400.0, 37.01, -115.0, 90.0, 400.0)
    assert angle_diff_deg(lead, pure) > 3.0


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
    assert abs(far.roll_rate_dps) == 213.1, "still full rate far from level"


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


def test_engage_does_not_pull_hard_g_while_still_wings_level():
    """User-reported and confirmed live: step 0 of a real engagement logged
    roll=-2 (essentially wings level) with g=6.98 -- a huge pull with almost
    no bank is not a turn toward the target, it is a pure vertical loop
    ("points to the sky and climbs"). A large bearing error commands a large
    desired bank, but the hard pull must ramp in with actual bank progress,
    not apply the instant the turn starts."""
    p = CombatPilot(callsign="BLUE1", target_alt_ft=15000.0)
    # Target far off to one side (large bearing error -> large desired
    # bank), jet still essentially wings-level.
    cmd = p.step(_telem(alt_ft=15000.0, roll_deg=-2.0, heading_deg=0.0,
                        tgt_lat=37.0, tgt_lon=-114.9, tgt_heading_deg=90.0,
                        airspeed_kt=475.0, tgt_range_nm=2.0), NO_THREAT)
    assert cmd.g_cmd < 2.0, "must roll into the bank before pulling hard"


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
    quietly loaded the whole time."""
    p = CombatPilot(callsign="BLUE1", target_alt_ft=15000.0)
    # far below and far above the (irrelevant, for RESET) target altitude
    low = p.step(_telem(alt_ft=8000.0, airspeed_kt=180.0), NO_THREAT)
    high = p.step(_telem(alt_ft=20000.0, airspeed_kt=180.0), NO_THREAT)
    assert low.state == "RESET" and high.state == "RESET"
    assert low.g_cmd == 1.0
    assert high.g_cmd == 1.0


def test_reset_pushes_nose_down_when_pitched_up_instead_of_a_flat_one_g():
    """A bare 1g only holds the CURRENT flight path -- it is only "level"
    when the jet is already near level. User-reported and confirmed live:
    after a hard pull, RESET's flat 1g left the jet "pointing at the sky and
    climbing" for 25+ real seconds, 476->248kt. Two corrections were tried
    and both failed (see PITCH_RECOVER_G_PER_DEG's definition for the full
    history, including a mathematically "correct" n=cos(gamma) neutral that
    only HELD an already-vertical climb instead of fixing it). RESET now
    uses the same proven pitch-hold as ENGAGE/INTERCEPT, ported from
    f15_airshow.py's hold_nose(): actively drive pitch toward level."""
    p = CombatPilot(callsign="BLUE1")
    level = p.step(_telem(airspeed_kt=180.0, pitch_deg=0.0), NO_THREAT)
    nose_up = p.step(_telem(airspeed_kt=180.0, pitch_deg=60.0), NO_THREAT)
    assert level.state == "RESET" and nose_up.state == "RESET"
    assert level.g_cmd == 1.0
    assert nose_up.g_cmd < level.g_cmd, "must push over, not hold the climb"


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
