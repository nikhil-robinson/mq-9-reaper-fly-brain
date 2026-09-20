"""Great-circle geometry, shared by the flight scripts and the combat logic.

Moved out of scripts/f15_intercept.py so src/fly_brain/combat.py can use it
without importing a script that opens sockets -- the combat policy must stay
testable with no FlightGear anywhere in the picture.
"""
import math


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def offset_point(lat, lon, heading_deg, dist_m):
    R = 6371000.0
    brg = math.radians(heading_deg)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    dr = dist_m / R
    p2 = math.asin(math.sin(p1) * math.cos(dr)
                   + math.cos(p1) * math.sin(dr) * math.cos(brg))
    l2 = l1 + math.atan2(math.sin(brg) * math.sin(dr) * math.cos(p1),
                         math.cos(dr) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)
