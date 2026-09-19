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
        # Without this, every get/set round-trip stalls ~0.4s -- Nagle's
        # algorithm on our send colliding with FlightGear's delayed ACK on
        # its end, confirmed live by timing a single `get` before/after
        # this flag (0.39s -> <5ms). At 7+ properties read per control
        # step, that 0.4s-per-call tax alone was slowing the loop enough
        # that a fast-rolling jet could roll hundreds of degrees between
        # two control updates, which looked like a flight-control bug but
        # wasn't one.
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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

    def get(self, path, _retried=False):
        """Read a property. Retries once on a malformed reply -- observed
        live against a real FlightGear instance under sustained load (a
        response missing its quoted value, likely a dropped/delayed TCP
        segment), not just a theoretical risk."""
        response = self._send(f"get {path}")
        parts = response.split("'")
        if len(parts) < 2:
            if _retried:
                raise ValueError(f"malformed reply for {path!r} even after retry: {response!r}")
            return self.get(path, _retried=True)
        return parts[1]

    def set(self, path, value):
        self._send(f"set {path} {value}")

    def run_nasal(self, script):
        """Execute arbitrary Nasal over telnet's undocumented `run` command.

        Needed for properties that `set` creates as the wrong type (e.g.
        heading-bug-deg comes out as string via `set`, breaking the
        autopilot -- confirmed live against the c172p's KAP140) and for
        calling aircraft Nasal functions directly (e.g. the F-15's
        AFCS heading-mode cycler, which isn't a settable property at all).
        """
        self._send(f"run nasal script={script}")

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

    def arm_weapons(self):
        """Arm the master-arm switch. Without this, trigger_weapons() is a
        silent no-op -- confirmed live: firing with master-arm still false
        left /ai/submodels/submodel[2]/count (the cannon's live ammo count)
        completely unchanged after repeated fire commands."""
        self.set("/controls/armament/master-arm", "true")

    def trigger_weapons(self, fire):
        """Fire the M61A1 cannon via its actual JSBSim FCS trigger property.

        /controls/armament/trigger (the pilot's physical trigger switch)
        only does something once fire-control.nas's Nasal state machine has
        a weapon selected via its pylon-selection logic -- confirmed live
        that setting it alone left the cannon's ammo count unchanged after
        repeated fire commands. fdm/jsbsim/fcs/guntrigger is the low-level
        property that actually releases the M61A1 submodel (Nasal/pylons.nas
        wires the cannon's SubModelWeapon straight to this property); setting
        it directly, with master-arm on, verified live via a real ammo-count
        drop (510 -> 509 over 2s of firing).
        """
        self.set("/fdm/jsbsim/fcs/guntrigger", "true" if fire else "false")

    def close(self):
        self._sock.close()
