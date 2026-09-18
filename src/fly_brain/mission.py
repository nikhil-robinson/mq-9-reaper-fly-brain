"""Mission phase state machine (PLAN.md Section 6): scripted TAKEOFF and
LANDING, brain-driven CRUISE. Deliberately has no dependency on
FlightGearBridge or the brain modules -- it's pure transition logic over
telemetry values the caller already has."""

from enum import Enum, auto


class MissionPhase(Enum):
    TAKEOFF = auto()
    CRUISE = auto()
    LANDING = auto()


def next_phase(
    current_phase,
    altitude_m,
    airspeed_mps,
    waypoints_remaining,
    min_takeoff_altitude_m=150.0,
    min_takeoff_airspeed_mps=60.0,
):
    if current_phase == MissionPhase.TAKEOFF:
        if altitude_m >= min_takeoff_altitude_m and airspeed_mps >= min_takeoff_airspeed_mps:
            return MissionPhase.CRUISE
        return MissionPhase.TAKEOFF

    if current_phase == MissionPhase.CRUISE:
        if waypoints_remaining <= 0:
            return MissionPhase.LANDING
        return MissionPhase.CRUISE

    return MissionPhase.LANDING
