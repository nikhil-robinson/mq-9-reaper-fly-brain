from fly_brain.combat import (
    AIM9_FOV_DEG,
    CORNER_SPEED_KT,
    CombatPilot,
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
    assert aim9_in_envelope(2.0, 5.0, 300.0, True)
    assert not aim9_in_envelope(2.0, 5.0, 300.0, False), "no lock"
    assert not aim9_in_envelope(0.1, 5.0, 300.0, True), "too close to arm"
    assert not aim9_in_envelope(12.0, 5.0, 300.0, True), "beyond max range"
    assert not aim9_in_envelope(2.0, 40.0, 300.0, True), "outside seeker FOV"
    assert not aim9_in_envelope(2.0, 5.0, 5.0, True), "target under 15kt"


def test_aim9_envelope_is_symmetric_in_bearing_error():
    assert aim9_in_envelope(2.0, AIM9_FOV_DEG - 1, 300.0, True)
    assert aim9_in_envelope(2.0, -(AIM9_FOV_DEG - 1), 300.0, True)
    assert not aim9_in_envelope(2.0, AIM9_FOV_DEG + 1, 300.0, True)


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
    cmd = p.step(_telem(tgt_range_nm=2.0, has_lock=True),
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
    assert p.step(_telem(tgt_range_nm=2.0, has_lock=True), NO_THREAT).fire_missile
    assert not p.step(_telem(tgt_range_nm=2.0, has_lock=False), NO_THREAT).fire_missile


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
    turns, and inverted it points the nose further down."""
    p = CombatPilot(callsign="BLUE1")
    cmd = p.step(_telem(agl_ft=800.0, roll_deg=70.0, pitch_deg=-20.0,
                        airspeed_kt=400.0), NO_THREAT)
    assert cmd.state == "RECOVER"
    assert abs(cmd.roll_rate_dps) == 213.1, "full rate, not a trickle"
    assert cmd.roll_rate_dps < 0, "rolling back toward level from +70deg"
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
