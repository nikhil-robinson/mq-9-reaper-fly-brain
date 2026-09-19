"""One-time setup step: while FlightGear is already running (with any
protocol -- this only needs telnet), find the nearest real AI target and
generate+deploy a UDP protocol locked to that one target's index.

Streaming all 25 AI target positions in one FlightGear generic protocol
broke the AI scenario's own Nasal script (confirmed live); one target
index at a time is confirmed safe. So target selection has to happen
before the real-time flight loop can use pure UDP for it.

Usage: .venv/bin/python scripts/pick_target_and_fly.py
Then: restart FlightGear (protocol files are only read at launch), then
run scripts/simple_intercept.py, which is pure UDP after this point.
"""

import math

from fly_brain.flightgear_bridge import FlightGearBridge
from fly_brain.protocol_gen import generate_and_deploy


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def main():
    print("Connecting via telnet for one-time target selection...")
    telnet = FlightGearBridge(host="localhost", port=5501)
    own_lat = float(telnet.get("/position/latitude-deg"))
    own_lon = float(telnet.get("/position/longitude-deg"))
    n_models = int(float(telnet.get("/ai/models/count")))
    print(f"Own position: {own_lat:.5f}, {own_lon:.5f}. Scanning {n_models} AI models...")

    best = None
    for i in range(n_models):
        try:
            lat = float(telnet.get(f"/ai/models/aircraft[{i}]/position/latitude-deg"))
            lon = float(telnet.get(f"/ai/models/aircraft[{i}]/position/longitude-deg"))
        except ValueError:
            continue
        dist = haversine_m(own_lat, own_lon, lat, lon)
        if best is None or dist < best[0]:
            best = (dist, i)
    telnet.close()

    if best is None:
        print("No AI targets found.")
        return

    dist, target_idx = best
    print(f"Nearest target: index {target_idx}, dist={dist:.0f}m")
    print("Generating and deploying protocol locked to this target (needs sudo)...")
    generate_and_deploy(target_idx)
    print(f"Done. Restart FlightGear, then run simple_intercept.py -- it's pure UDP now, locked to target[{target_idx}].")


if __name__ == "__main__":
    main()
