"""High-rate telemetry/control link over FlightGear's generic UDP protocol
(data/fly_brain_protocol.xml, generated per-target by protocol_gen.py):
own-ship telemetry, ONE locked AI target's position, flight controls, and
the weapons trigger, all real-time, no telnet in the flight loop at all.

Streaming all 25 AI target positions in one protocol broke FlightGear's
own scenario script (a live Nasal runtime error, "non-objects have no
members", inside scenario_moving_targets_around_KXTA) -- confirmed by
bisecting live that a single aircraft[N] chunk works cleanly, so
protocol_gen.py generates a protocol for exactly one target index,
selected via a one-time telnet lookup before the flight loop starts (see
scripts/pick_target_and_fly.py).

Telnet's PropsChannel is a synchronous debug console: confirmed live, a raw
socket issuing bare `get` commands, bypassing our own client code entirely,
still took ~0.4s per round-trip (server side, not fixable with TCP_NODELAY).
At that rate a fast-rolling jet can roll hundreds of degrees between two
control updates -- that was the real cause of every "roll and crash" seen
while flying via FlightGearBridge for the hot loop, not a flaw in the
control law.

Requires launching fgfs with:
    --generic=socket,out,20,localhost,5502,udp,fly_brain_protocol
    --generic=socket,in,20,localhost,5503,udp,fly_brain_protocol
(fly_brain_protocol.xml must be installed to FlightGear's own Protocol/
directory -- FlightGear only resolves generic protocol names as
"Protocol/<name>.xml" under its own data paths, confirmed live: an absolute
path in --generic is rejected with "not found in any data path", and an
aircraft-local Protocol/ dir is not searched either, only $FG_ROOT's.)

Note: type=double silently breaks value output in this FlightGear build's
ASCII generic protocol (confirmed live: a value stayed frozen at 0 with
type=double, fixed immediately by switching to type=float with an explicit
<format>) -- the protocol XML uses float throughout for this reason.
"""

import socket


class RealtimeBridge:
    def __init__(self, telemetry_port=5502, control_port=5503, host="localhost", timeout=5.0):
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.bind((host, telemetry_port))
        # Non-blocking + poll, not a blocking recv() with settimeout(): the
        # latter reliably hung forever in this environment even with real
        # packets arriving at 20Hz (confirmed live, isolated down to just
        # bind+settimeout+recv, no other code involved) -- socket.timeout
        # was simply never raised and recv() never returned. Cause
        # unconfirmed (this exact setsockopt-free pattern is the one that's
        # actually been proven to work, so it's used as-is rather than
        # further diagnosed).
        self._recv_sock.setblocking(False)
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._send_addr = (host, control_port)
        self._latest = None
        self._timeout = timeout

    def get_telemetry(self):
        """Drain all buffered packets and return the freshest one -- UDP
        can queue up several packets between calls; using the latest
        keeps the control loop from reacting to stale telemetry."""
        import time

        packet = None
        while True:
            try:
                packet = self._recv_sock.recv(8192)
            except BlockingIOError:
                break
        if packet is None:
            if self._latest is None:
                deadline = time.monotonic() + self._timeout
                while packet is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                    try:
                        packet = self._recv_sock.recv(8192)
                    except BlockingIOError:
                        continue
                if packet is None:
                    raise TimeoutError("No telemetry received from FlightGear's generic UDP output")
            else:
                packet = self._latest
        self._latest = packet
        (
            lat, lon, alt_ft, heading_deg, pitch_deg, roll_deg, airspeed_kt,
            tgt_lat, tgt_lon, tgt_alt_ft,
        ) = (float(x) for x in packet.decode().strip().split("\t"))
        return {
            "lat_deg": lat,
            "lon_deg": lon,
            "altitude_m": alt_ft * 0.3048,
            "heading_deg": heading_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "airspeed_mps": airspeed_kt * 0.514444,
            "target_lat_deg": tgt_lat,
            "target_lon_deg": tgt_lon,
            "target_alt_ft": tgt_alt_ft,
        }

    def send_controls(self, aileron, elevator, rudder, throttle, fire=False):
        line = f"{aileron:.4f}\t{elevator:.4f}\t{rudder:.4f}\t{throttle:.4f}\t{1.0 if fire else 0.0:.1f}\n"
        self._send_sock.sendto(line.encode(), self._send_addr)

    def close(self):
        self._recv_sock.close()
        self._send_sock.close()
