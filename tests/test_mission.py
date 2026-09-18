from fly_brain.mission import MissionPhase, next_phase


def test_stays_in_takeoff_below_altitude_threshold():
    phase = next_phase(
        MissionPhase.TAKEOFF, altitude_m=50.0, airspeed_mps=70.0, waypoints_remaining=3
    )
    assert phase == MissionPhase.TAKEOFF


def test_stays_in_takeoff_below_airspeed_threshold():
    phase = next_phase(
        MissionPhase.TAKEOFF, altitude_m=200.0, airspeed_mps=40.0, waypoints_remaining=3
    )
    assert phase == MissionPhase.TAKEOFF


def test_advances_from_takeoff_to_cruise_once_airborne_and_fast_enough():
    phase = next_phase(
        MissionPhase.TAKEOFF, altitude_m=200.0, airspeed_mps=70.0, waypoints_remaining=3
    )
    assert phase == MissionPhase.CRUISE


def test_stays_in_cruise_while_waypoints_remain():
    phase = next_phase(
        MissionPhase.CRUISE, altitude_m=1000.0, airspeed_mps=150.0, waypoints_remaining=2
    )
    assert phase == MissionPhase.CRUISE


def test_advances_from_cruise_to_landing_when_no_waypoints_remain():
    phase = next_phase(
        MissionPhase.CRUISE, altitude_m=1000.0, airspeed_mps=150.0, waypoints_remaining=0
    )
    assert phase == MissionPhase.LANDING


def test_landing_is_terminal():
    phase = next_phase(
        MissionPhase.LANDING, altitude_m=0.0, airspeed_mps=0.0, waypoints_remaining=0
    )
    assert phase == MissionPhase.LANDING


def test_custom_thresholds_are_respected():
    phase = next_phase(
        MissionPhase.TAKEOFF,
        altitude_m=80.0,
        airspeed_mps=65.0,
        waypoints_remaining=1,
        min_takeoff_altitude_m=50.0,
        min_takeoff_airspeed_mps=60.0,
    )
    assert phase == MissionPhase.CRUISE
