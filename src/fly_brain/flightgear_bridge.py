"""Bridge to a live FlightGear instance over its telnet property-tree
protocol (PLAN.md Section 4). Confirmed wire format, live telnet session
against f16-simplified, FlightGear 2020.3.19: every get/set reply is
"{path} = '{value}' ({type})\\r\\n" followed by the prompt "/> "."""

import math
import socket

_FT_TO_M = 0.3048
_KT_TO_MPS = 0.514444


class FlightGearBridge:
    def __init__(self, host="localhost", port=5501, timeout=5.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._buf = b""
        # Some FlightGear telnet sessions send an unsolicited "/> " banner
        # right on connect (observed on one real instance); others send
        # nothing until the first command (observed on another real
        # instance, same FlightGear version -- inconsistent in practice).
        # Drain a banner if one shows up quickly; don't block waiting for
        # one that may never come.
        self._sock.settimeout(0.5)
        try:
            self._read_until_prompt()
        except (socket.timeout, TimeoutError):
            self._buf = b""
        self._sock.settimeout(timeout)

    def _read_until_prompt(self):
        while not self._buf.endswith(b"/> "):
            chunk = self._sock.recv(4096)
            if not chunk:
                break
            self._buf += chunk
        response = self._buf
        self._buf = b""
        return response.decode(errors="replace")

    def _send(self, line):
        self._sock.sendall((line + "\r\n").encode())
        return self._read_until_prompt()

    def get(self, path):
        response = self._send(f"get {path}")
        return response.split("'", 2)[1]

    def set(self, path, value):
        self._send(f"set {path} {value}")

    def get_telemetry(self):
        """Return current flight state from FlightGear.

        Note: heading_rad uses FlightGear's compass convention (0=North,
        clockwise), NOT the math convention (0=East, counter-clockwise) that
        fly_brain.course.bearing_and_distance() expects. To convert heading_rad
        for bearing_and_distance(), use: math.radians(90.0 - heading_deg).
        """
        return {
            "lat_deg": float(self.get("/position/latitude-deg")),
            "lon_deg": float(self.get("/position/longitude-deg")),
            "altitude_m": float(self.get("/position/altitude-ft")) * _FT_TO_M,
            "heading_rad": math.radians(float(self.get("/orientation/heading-deg"))),
            "pitch_rad": math.radians(float(self.get("/orientation/pitch-deg"))),
            "roll_rad": math.radians(float(self.get("/orientation/roll-deg"))),
            "airspeed_mps": float(self.get("/velocities/airspeed-kt")) * _KT_TO_MPS,
        }

    def send_controls(self, aileron, elevator, rudder, throttle):
        self.set("/controls/flight/aileron", aileron)
        self.set("/controls/flight/elevator", elevator)
        self.set("/controls/flight/rudder", rudder)
        self.set("/controls/engines/engine[0]/throttle", throttle)

    def trigger_weapons(self, fire):
        self.set("/controls/armament/trigger", "true" if fire else "false")

    def close(self):
        self._sock.close()
