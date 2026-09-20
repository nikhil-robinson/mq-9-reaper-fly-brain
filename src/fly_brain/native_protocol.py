"""FlightGear's native binary protocols (--native-fdm, --native-ctrls):
fixed, compiled-in C struct layouts, no external protocol XML file needed
at all -- unlike the generic protocol, which turned out to depend on
FlightGear's own Protocol/ directory being writable (true on a normal
install, but NOT true for the flatpak build: /app is read-only by design,
confirmed live, and pointing --fg-root at a writable overlay directory
instead triggered FlightGear's own auto-repair mechanism to re-download
the ~2GB data package, which is not something to do by accident).

Struct layouts and byte order taken directly from FlightGear's own source
(src/Network/net_fdm.hxx, net_ctrls.hxx, native_structs.cxx -- fetched via
`gh api`, not guessed): FG_NET_FDM_VERSION=24, FG_NET_CTRLS_VERSION=27,
both structs use network byte order (big-endian) on the wire, confirmed
from native_structs.cxx's FGCtrls2Props/FGProps2Ctrls calling htonl/htond
on every field when net_byte_order=true (which native_ctrls.cxx always
passes).

FGNetCtrls is NOT a partial-override protocol -- FGCtrls2Props
unconditionally overwrites master-bat/master-alt/magnetos/starter/
speedup/freeze on every single packet (confirmed by reading
native_structs.cxx directly), so a naive zero-filled packet would kill the
engine and/or freeze the simulation on the first send. This module fills
every field with an explicit "keep the aircraft running normally" value,
not just the axes we care about.
"""

import socket
import struct
import time

FG_NET_FDM_VERSION = 24
FG_NET_CTRLS_VERSION = 27

# '>' = network byte order (big-endian), matching FGCtrls2Props/FGProps2FDM.
_FDM_FORMAT = ">2I3d22f5I36f1I4f4I9f1I1i11f"
_FDM_SIZE = struct.calcsize(_FDM_FORMAT)  # 408 bytes

# 3x '4x' = C compiler alignment padding before double fields that Python's
# '>' (standard, no-auto-align) mode won't insert on its own -- confirmed
# live: without these, FlightGear's native-ctrls read(buf,length)==length
# check silently failed (packet length didn't match sizeof(FGNetCtrls)),
# so every control write was silently dropped with zero error output.
_CTRLS_FORMAT = ">1I4x9d19I4x12d4I4d47I4x5d2I11d28I"
_CTRLS_SIZE = struct.calcsize(_CTRLS_FORMAT)  # 744 bytes


def decode_fdm(packet):
    """Unpack a FGNetFDM packet. Only returns the fields we use (position,
    orientation) -- the struct has ~60 fields total, most irrelevant here."""
    if len(packet) != _FDM_SIZE:
        raise ValueError(f"FGNetFDM packet is {len(packet)} bytes, expected {_FDM_SIZE}")
    fields = struct.unpack(_FDM_FORMAT, packet)
    # Field order per _FDM_FORMAT groups: version,padding, lon,lat,alt,
    # agl,phi,theta,psi,alpha,beta, phidot..v_body_w(11), AXYZ(3), stall,slip(2), ...
    version, _padding = fields[0], fields[1]
    longitude, latitude, altitude = fields[2], fields[3], fields[4]
    agl, phi, theta, psi = fields[5], fields[6], fields[7], fields[8]
    alpha = fields[9]  # angle of attack, radians
    thetadot = fields[12]  # pitch rate, rad/s
    vcas = fields[14]  # calibrated airspeed, knots
    # FlightGear's OWN climb rate, ft/s (net_fdm.hxx: ...psidot, vcas,
    # climb_rate, v_north...). Previously the flight loop differentiated
    # altitude against time.monotonic() instead. That is wall-clock, so it
    # silently breaks the moment sim time != wall time (/sim/speed-up), and
    # the K_CLIMB_G damping term it feeds is load-bearing -- session 6's
    # pylon run oscillated 7600->21700ft without it. Take the FDM's value.
    climb_rate_fps = fields[15]
    # 11f block (indices 88-98): visibility, elevator, elevator_trim_tab,
    # left_flap, right_flap, left_aileron, right_aileron, rudder,
    # nose_wheel, speedbrake, spoilers -- confirmed by hand-counting the
    # struct groups in _FDM_FORMAT against FlightGear's own net_fdm.hxx
    # layout (2I,3d,22f,5I,36f,1I,4f,4I,9f,1I,1i,11f). This is the ACTUAL
    # control-surface position, independent of what we last commanded --
    # comparing the two tells us whether our elevator command is reaching
    # the surface at all, or something else (trim, AFCS) is fighting it.
    elevator_pos = fields[89]
    rad2deg = 180.0 / 3.14159265358979323846
    return {
        "version": version,
        "lon_deg": longitude * rad2deg,
        "lat_deg": latitude * rad2deg,
        "altitude_m": altitude,
        "agl_m": agl,  # height above ground -- takeoff rotation and landing flare

        "roll_deg": phi * rad2deg,
        "pitch_deg": theta * rad2deg,
        "heading_deg": psi * rad2deg,
        "alpha_deg": alpha * rad2deg,
        "pitch_rate_dps": thetadot * rad2deg,
        "airspeed_kt": vcas,
        "climb_rate_fps": climb_rate_fps,
        "elevator_pos": elevator_pos,
    }


def encode_ctrls(aileron, elevator, rudder, throttle, gear_down=False,
                 brakes=0.0, flaps=0.0, speedbrake=0.0, speedup=1):
    """Pack a full FGNetCtrls packet. Every field is given an explicit
    safe value -- see module docstring for why zero-filling is dangerous
    (kills the engine, can freeze the sim)."""
    version = FG_NET_CTRLS_VERSION
    # 9d: aileron, elevator, rudder, aileron_trim, elevator_trim,
    # rudder_trim, flaps, spoilers, speedbrake
    d9 = (aileron, elevator, rudder, 0.0, 0.0, 0.0, flaps, 0.0, speedbrake)
    # 19I: flaps_power, flap_motor_ok, num_engines, master_bat[4],
    # master_alt[4], magnetos[4], starter_power[4]
    i19 = (
        1, 1, 2,
        1, 1, 0, 0,  # master_bat: engines 0,1 on
        1, 1, 0, 0,  # master_alt
        0, 0, 0, 0,  # magnetos: this F-15's own Nasal doesn't use the
                     # generic magnetos property and reacts badly to any
                     # nonzero write here (confirmed live: writing 3 got
                     # silently reset to 0 by the aircraft's own logic,
                     # and that reset triggered /controls/engines/engine[0]
                     # /cutoff=true, a property not even in this struct --
                     # a persistent engine-kill that needed a full restart
                     # to clear). Sending 0 from the first packet onward
                     # avoids the 3->0 transition that seems to be the
                     # actual trigger.
        0, 0, 0, 0,  # starter_power: already running, no starter needed
    )
    # 12d: throttle[4], mixture[4], condition[4]
    d12 = (
        throttle, throttle, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0,
    )
    # 4I: fuel_pump_power[4]
    i4 = (1, 1, 0, 0)
    # 4d: prop_advance[4]
    d4 = (1.0, 1.0, 0.0, 0.0)
    # 47I: feed_tank_to[4], reverse[4], engine_ok[4], mag_left_ok[4],
    # mag_right_ok[4], spark_plugs_ok[4], oil_press_status[4],
    # fuel_pump_ok[4], num_tanks, fuel_selector[8], xfer_pump[5], cross_feed
    i47 = (
        0, 0, 0, 0,        # feed_tank_to
        0, 0, 0, 0,        # reverse
        1, 1, 0, 0,        # engine_ok
        1, 1, 0, 0,        # mag_left_ok
        1, 1, 0, 0,        # mag_right_ok
        1, 1, 0, 0,        # spark_plugs_ok
        0, 0, 0, 0,        # oil_press_status (0 = normal)
        1, 1, 0, 0,        # fuel_pump_ok
        4,                 # num_tanks
        1, 1, 1, 1, 0, 0, 0, 0,  # fuel_selector[8]
        0, 0, 0, 0, 0,     # xfer_pump[5]
        0,                 # cross_feed
    )
    # 5d: brake_left, brake_right, copilot_brake_left, copilot_brake_right,
    # brake_parking
    d5 = (brakes, brakes, 0.0, 0.0, 0.0)
    # 2I: gear_handle, master_avionics
    # gear_handle must be a real parameter, not a constant: FGNetCtrls
    # overwrites it on EVERY packet, so setting /controls/gear/gear-down over
    # telnet is silently reverted ~20 times a second. Takeoff and landing are
    # impossible without this. Default stays False (up) to match the in-air
    # start every previous script used.
    i2 = (1 if gear_down else 0, 1)  # gear handle, avionics on
    # 11d: comm_1, comm_2, nav_1, nav_2, wind_speed_kt, wind_dir_deg,
    # turbulence_norm, temp_c, press_inhg, hground, magvar
    d11 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 29.92, 0.0, 0.0)
    # 28I: icing, speedup, freeze, reserved[25]
    # speedup is how many sim seconds pass per wall second. It MUST be
    # sent here rather than set as /sim/speed-up over telnet: FGNetCtrls
    # applies every field on every packet, so a telnet-set speed-up is
    # silently reverted ~20 times a second by our own control stream (the
    # same trap that made /controls/gear/gear-down unsettable). 0 would
    # freeze sim time outright, so it is floored at 1.
    i28 = (0, max(1, int(speedup)), 0) + (0,) * 25

    values = (version,) + d9 + i19 + d12 + i4 + d4 + i47 + d5 + i2 + d11 + i28
    return struct.pack(_CTRLS_FORMAT, *values)


class NativeBridge:
    def __init__(self, fdm_port=5504, ctrls_port=5505, host="localhost"):
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.bind((host, fdm_port))
        self._recv_sock.setblocking(False)
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._send_addr = (host, ctrls_port)
        self._latest = None

    def get_telemetry(self, timeout=5.0):
        packet = None
        while True:
            try:
                packet = self._recv_sock.recv(4096)
            except BlockingIOError:
                break
        if packet is None:
            if self._latest is None:
                # First call right after connecting: the non-blocking drain
                # above can run before FlightGear's first packet arrives.
                # Poll briefly instead of failing immediately -- confirmed
                # live this was needed (get_telemetry() raised on the very
                # first call otherwise, even though packets started
                # flowing a moment later).
                deadline = time.monotonic() + timeout
                while packet is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                    try:
                        packet = self._recv_sock.recv(4096)
                    except BlockingIOError:
                        continue
                if packet is None:
                    raise TimeoutError("No FGNetFDM packet received")
            else:
                packet = self._latest
        self._latest = packet
        return decode_fdm(packet)

    def send_controls(self, aileron, elevator, rudder, throttle,
                      gear_down=False, brakes=0.0, flaps=0.0, speedbrake=0.0,
                      speedup=1):
        packet = encode_ctrls(aileron, elevator, rudder, throttle,
                              gear_down=gear_down, brakes=brakes,
                              flaps=flaps, speedbrake=speedbrake,
                              speedup=speedup)
        self._send_sock.sendto(packet, self._send_addr)

    def close(self):
        self._recv_sock.close()
        self._send_sock.close()
