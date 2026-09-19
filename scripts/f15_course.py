"""Multi-gate tower course for the F-15: fly between pairs of scaled towers.

Each gate is TWO SEPARATE tower objects (scripts/f15_props.py's tower_big.xml
-- a real FlightGear cooling tower model, scaled 4x uniformly) placed
independently apart, so gap and height are fully decoupled (unlike the
earlier attempt to reuse the real Red Bull gate mesh, which bakes both
pylons into one fixed-offset object). The aircraft flies through the gap
between each pair.

History, briefly (see git log / SESSION.md for the full story): the first
version used the real Red Bull Air Race gate model (RedbullRedGate.ac) at
real 18m scale, requiring precise low-altitude tracking near real terrain --
this caused repeated live crashes chasing that precision. Scaling that same
mesh 1000x in HEIGHT ONLY rendered as a broken horizontal streak (confirmed
live via a sim screenshot): an 18m-footprint, 59449ft-tall shape is too
extreme an aspect ratio for the renderer. Scaling it 50x UNIFORMLY fixed the
aspect-ratio problem but the mesh's own baked-in pylon spacing (~14m) could
not be adjusted independently of height. Switching to the simpler cooling
tower model (2 named parts sharing one origin, vs the gate's 20) and placing
TWO of them per gate, at whatever spacing chosen in code, solves both
problems: sane aspect ratio (scaled uniformly, same proportions as the real,
correctly-rendering tower) and fully independent height/spacing control.
"""

import math

from f15_intercept import angle_diff_deg, bearing_deg, haversine_m, offset_point

FT = 0.3048  # metres per foot

# Cooling tower (Models/Power/coolingtower.ac), measured live from its own
# AC3D vertex extents: 101.42m wide/deep, 131.33m tall, only 2 named parts
# ("Cube" the base, "Cylinder" the tall part) sharing one coordinate origin
# (contiguous Y ranges, centered X/Z) -- confirmed live this scales and
# renders cleanly, unlike the more complex Red Bull gate mesh. Scaled 4x
# uniformly (see f15_props.py's tower_big.xml): a real, sane aspect ratio,
# just bigger, so no rendering artifacts.
TOWER_SCALE = 8.0  # MUST match tower_big.xml's scale animation -- see below
TOWER_HEIGHT_FT = 131.33 * TOWER_SCALE / FT  # ~3447ft
TOWER_FOOTPRINT_RADIUS_M = 101.42 * TOWER_SCALE / 2.0  # ~406m
# This file said 4.0 while tower_big.xml scaled 8.0, so every collision
# cylinder was HALF the size of the tower actually drawn in the sim: 203m
# radius against a real 406m one. That is why the gap looked far tighter in
# the sim than the numbers here claimed -- the numbers were describing a
# tower half the size of the one on screen.

# Gate width is the two towers' CENTER-TO-CENTER spacing, fully independent
# of their height since they are two separate placed objects. At the real
# 406m footprint radius, the old 900m spacing left a 88m clear gap, not the
# 494m the old comment claimed -- an F-15 at 450kt covers 88m in 0.4s, which
# is why the gate looked (and was) impassably tight. 2000m leaves a real
# ~1188m clear gap while the towers still read as one gate.
GATE_WIDTH_M = 2000.0

# Fly height above the REAL ground at each gate, comfortably WITHIN the
# tower's ~1723ft height (not above or below it) -- about 45% of the way
# up, giving real margin both above ground and below the tower tops even
# with a few hundred feet of altitude-tracking oscillation (the vertical
# guidance is not perfectly precise; see f15_obstacle_run.py's K_ALT_G
# history for the tuning story).
GATE_FLY_AGL_FT = 800.0

# GATE_SPACING_M: a 35deg-bank turn at 450kt has a radius of v^2/(g tan35)
# ~= 7800m. At the old 3000m spacing the jet could not physically correct a
# lateral error before the next gate arrived -- once off the line it stayed
# off it, which is exactly the "flies past the left side of the gates"
# behaviour seen live. Spacing has to be at least a turn radius.
GATE_SPACING_M = 8000.0  # ~35s between gates at 450kt (231 m/s)
N_GATES = 6
CLEARANCE_M = 150.0  # miss distance that counts as "cleared, not hit"
GATE_PASS_RADIUS_M = 250.0  # generous: a pass through the gate, not a precise slot


class Obstacle:
    __slots__ = ("lat", "lon", "base_ft", "top_ft", "radius_m", "kind", "label")

    def __init__(self, lat, lon, base_ft, height_ft, radius_m, kind, label):
        self.lat, self.lon = lat, lon
        self.base_ft, self.top_ft = base_ft, base_ft + height_ft
        self.radius_m, self.kind, self.label = radius_m, kind, label

    def miss_distance_m(self, lat, lon, alt_ft):
        """Signed clearance: negative means inside the cylinder (a hit)."""
        horiz = haversine_m(lat, lon, self.lat, self.lon) - self.radius_m
        vert = (alt_ft - self.top_ft) * FT
        below = (self.base_ft - alt_ft) * FT
        return max(horiz, vert, below)


class Gate:
    __slots__ = ("lat", "lon", "alt_ft", "label", "kind")

    def __init__(self, lat, lon, alt_ft, label, kind):
        self.lat, self.lon, self.alt_ft = lat, lon, alt_ft
        self.label, self.kind = label, kind


def _flat_terrain_ft(lat, lon):
    """Default terrain function for offline testing -- flat sea level.

    The live runner passes a real geo.elevation()-backed function instead
    (see f15_obstacle_run.py); this default exists only so build_course and
    its self-check work without a running FlightGear instance.
    """
    return 0.0


def build_course(lat, lon, heading_deg, terrain_fn=_flat_terrain_ft, difficulty=None):
    """Lay N_GATES tower-pair gates ahead in a straight line.

    Each gate is two Obstacle towers, GATE_WIDTH_M apart (perpendicular to
    the course heading), each placed at its OWN real ground elevation (they
    can differ slightly if terrain slopes). The single Gate waypoint sits
    at the midpoint between them, GATE_FLY_AGL_FT above the midpoint's
    ground elevation. Returns (obstacles, gates, target_point).
    `difficulty` is accepted and ignored -- kept so callers built against
    the old signature still work.
    """
    obstacles, gates = [], []
    perp_hdg = (heading_deg + 90.0) % 360.0
    d = 10000.0  # run-in: room to settle onto the line before gate-1
    for i in range(N_GATES):
        cy, cx = offset_point(lat, lon, heading_deg, d)
        center_ground_ft = terrain_fn(cy, cx)
        fly_alt_ft = center_ground_ft + GATE_FLY_AGL_FT

        for side, tag in ((-1.0, "L"), (1.0, "R")):
            ty, tx = offset_point(cy, cx, perp_hdg, side * GATE_WIDTH_M / 2.0)
            tower_ground_ft = terrain_fn(ty, tx)
            obstacles.append(Obstacle(ty, tx, tower_ground_ft, TOWER_HEIGHT_FT,
                                      TOWER_FOOTPRINT_RADIUS_M, "tower",
                                      f"gate-{i + 1}{tag}"))
        gates.append(Gate(cy, cx, fly_alt_ft, f"gate-{i + 1}", "gate"))
        d += GATE_SPACING_M

    d += 4000.0  # run beyond the last gate before the target
    ty, tx = offset_point(lat, lon, heading_deg, d)
    target_ground_ft = terrain_fn(ty, tx)
    return obstacles, gates, (ty, tx, target_ground_ft + GATE_FLY_AGL_FT)


def course_length_m(gates, start_lat, start_lon):
    total, py, px = 0.0, start_lat, start_lon
    for g in gates:
        total += haversine_m(py, px, g.lat, g.lon)
        py, px = g.lat, g.lon
    return total


def _self_check():
    lat, lon, hdg = 37.0, -115.8, 90.0
    obstacles, gates, target = build_course(lat, lon, hdg)

    assert len(gates) == N_GATES, len(gates)
    assert len(obstacles) == N_GATES * 2, len(obstacles)  # two towers per gate

    # With flat terrain (0ft), every gate should fly at exactly GATE_FLY_AGL_FT.
    for g in gates:
        assert abs(g.alt_ft - GATE_FLY_AGL_FT) < 1e-6, g.alt_ft

    # The fly altitude must sit WITHIN the tower's height span, not above or
    # below it -- flying "through the gate" only means something if the
    # towers actually extend above and below the flight path.
    assert 0.0 < GATE_FLY_AGL_FT < TOWER_HEIGHT_FT, (GATE_FLY_AGL_FT, TOWER_HEIGHT_FT)

    # The two towers of gate 1 must be on opposite sides of its gate
    # waypoint, GATE_WIDTH_M apart -- not stacked on top of each other.
    left, right = obstacles[0], obstacles[1]
    assert left.label == "gate-1L" and right.label == "gate-1R"
    gap = haversine_m(left.lat, left.lon, right.lat, right.lon)
    assert abs(gap - GATE_WIDTH_M) < 1.0, gap

    # There must be a REAL clear gap between the towers' own footprints --
    # otherwise "fly through the gate" would mean flying through solid rock.
    clear_gap_m = GATE_WIDTH_M - 2 * TOWER_FOOTPRINT_RADIUS_M
    assert clear_gap_m > 300.0, clear_gap_m

    # Hits must actually register as hits (inside a tower's real footprint).
    assert left.miss_distance_m(left.lat, left.lon, left.base_ft + 5) < 0

    # Flying through the CENTRE of the gate, at the intended altitude, must
    # clear BOTH towers.
    fly_alt = left.base_ft + GATE_FLY_AGL_FT
    assert left.miss_distance_m(gates[0].lat, gates[0].lon, fly_alt) > 0
    assert right.miss_distance_m(gates[0].lat, gates[0].lon, fly_alt) > 0

    # The pass radius must be generous compared to the clear gap, or "not a
    # precise slot" would be a lie -- but must not be so large it reaches
    # into either tower's own footprint.
    assert GATE_PASS_RADIUS_M < clear_gap_m / 2.0 + CLEARANCE_M

    # A non-flat terrain function must actually change gate altitude --
    # otherwise "real ground elevation" would be dead code.
    obstacles2, gates2, _ = build_course(lat, lon, hdg, terrain_fn=lambda a, b: 5000.0)
    assert abs(gates2[0].alt_ft - (5000.0 + GATE_FLY_AGL_FT)) < 1e-6

    # The course must run away from the start, along the given heading.
    b = bearing_deg(lat, lon, target[0], target[1])
    assert abs(angle_diff_deg(b, hdg)) < 1.0, b
    assert course_length_m(gates, lat, lon) > 10000.0


if __name__ == "__main__":
    _self_check()
    lat, lon, hdg = 37.0, -115.8, 90.0
    obstacles, gates, target = build_course(lat, lon, hdg)
    print(f"SELF-CHECK OK: {N_GATES} gates ({len(obstacles)} towers), "
          f"{course_length_m(gates, lat, lon) / 1000:.1f} km of course, "
          f"towers ~{TOWER_HEIGHT_FT:.0f}ft tall, {GATE_WIDTH_M:.0f}m apart")
    for g in gates:
        print(f"  gate  {g.label:22s} alt {g.alt_ft:7.0f}ft (flat-terrain test)")
