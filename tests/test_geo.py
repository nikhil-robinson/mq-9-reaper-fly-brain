from fly_brain.geo import angle_diff_deg, bearing_deg, haversine_m, offset_point


def test_haversine_known_distance():
    # one degree of latitude is ~111.2 km
    d = haversine_m(37.0, -115.0, 38.0, -115.0)
    assert 110_000 < d < 112_000


def test_bearing_due_north_is_zero():
    assert abs(bearing_deg(37.0, -115.0, 38.0, -115.0)) < 0.5


def test_bearing_due_east_is_ninety():
    assert abs(bearing_deg(37.0, -115.0, 37.0, -114.0) - 90.0) < 1.0


def test_angle_diff_wraps_across_north():
    assert abs(angle_diff_deg(10.0, 350.0) - 20.0) < 1e-6
    assert abs(angle_diff_deg(350.0, 10.0) + 20.0) < 1e-6


def test_offset_point_round_trips_through_bearing_and_range():
    lat, lon = offset_point(37.0, -115.0, 45.0, 10_000.0)
    assert abs(haversine_m(37.0, -115.0, lat, lon) - 10_000.0) < 50.0
    assert abs(angle_diff_deg(bearing_deg(37.0, -115.0, lat, lon), 45.0)) < 0.5
