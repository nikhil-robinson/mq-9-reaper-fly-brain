import socket
import threading

import pytest

from fly_brain.flightgear_bridge import FlightGearBridge

# In-memory property store the mock server answers from, mimicking
# FlightGear's confirmed wire format exactly (see plan Global Constraints):
# every get/set reply is "{path} = '{value}' ({type})\r\n/> ".
_MOCK_PROPS = {
    "/controls/flight/aileron": ("0", "double"),
    "/controls/flight/elevator": ("0", "double"),
    "/controls/flight/rudder": ("0", "double"),
    "/controls/engines/engine[0]/throttle": ("0", "double"),
    "/controls/engines/engine[0]/cutoff": ("true", "bool"),
    "/controls/armament/trigger": ("false", "bool"),
    "/position/longitude-deg": ("-122.3933734", "double"),
    "/position/latitude-deg": ("37.62870187", "double"),
    "/position/altitude-ft": ("1000.0", "double"),
    "/orientation/heading-deg": ("90.0", "double"),
    "/orientation/pitch-deg": ("2.0", "double"),
    "/orientation/roll-deg": ("-1.0", "double"),
    "/velocities/airspeed-kt": ("150.0", "double"),
}


def _run_mock_server(sock, props):
    conn, _ = sock.accept()
    conn.sendall(b"/> ")  # initial banner/prompt, same as real FlightGear
    buf = b""
    with conn:
        while True:
            try:
                chunk = conn.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\r\n" in buf:
                line, buf = buf.split(b"\r\n", 1)
                line = line.decode()
                parts = line.split(" ", 2)
                cmd = parts[0]
                if cmd == "get":
                    path = parts[1]
                    value, typ = props.get(path, ("", "none"))
                    conn.sendall(f"{path} = '{value}' ({typ})\r\n/> ".encode())
                elif cmd == "set":
                    path, value = parts[1], parts[2]
                    _, typ = props.get(path, ("", "double"))
                    props[path] = (value, typ)
                    conn.sendall(f"{path} = '{value}' ({typ})\r\n/> ".encode())
                else:
                    conn.sendall(b"\r\n/> ")


@pytest.fixture
def mock_fg_server():
    props = dict(_MOCK_PROPS)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    thread = threading.Thread(target=_run_mock_server, args=(sock, props), daemon=True)
    thread.start()
    yield port, props
    sock.close()


def test_connect_reads_initial_prompt(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    bridge.close()


def test_get_returns_raw_value_string(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    assert bridge.get("/controls/flight/aileron") == "0"
    bridge.close()


def test_set_then_get_round_trips(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    bridge.set("/controls/flight/aileron", 0.5)
    assert bridge.get("/controls/flight/aileron") == "0.5"
    bridge.close()


def test_get_telemetry_converts_units(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    telem = bridge.get_telemetry()
    assert telem["lat_deg"] == pytest.approx(37.62870187)
    assert telem["lon_deg"] == pytest.approx(-122.3933734)
    assert telem["altitude_m"] == pytest.approx(1000.0 * 0.3048)
    assert telem["heading_rad"] == pytest.approx(1.5707963, abs=1e-4)  # 90 deg
    assert telem["pitch_rad"] == pytest.approx(0.0349066, abs=1e-4)  # 2 deg
    assert telem["roll_rad"] == pytest.approx(-0.0174533, abs=1e-4)  # -1 deg
    assert telem["airspeed_mps"] == pytest.approx(150.0 * 0.514444, abs=1e-3)
    bridge.close()


def test_send_controls_writes_all_four_properties(mock_fg_server):
    port, props = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    bridge.send_controls(aileron=0.3, elevator=-0.2, rudder=0.1, throttle=0.8)
    assert bridge.get("/controls/flight/aileron") == "0.3"
    assert bridge.get("/controls/flight/elevator") == "-0.2"
    assert bridge.get("/controls/flight/rudder") == "0.1"
    assert bridge.get("/controls/engines/engine[0]/throttle") == "0.8"
    bridge.close()


def test_trigger_weapons_true_and_false(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    bridge.trigger_weapons(True)
    assert bridge.get("/controls/armament/trigger") == "true"
    bridge.trigger_weapons(False)
    assert bridge.get("/controls/armament/trigger") == "false"
    bridge.close()


def test_get_retries_once_on_malformed_reply(mock_fg_server):
    """A real live FlightGear run under sustained load produced a reply
    missing its quoted value (dropped/delayed TCP segment); get() must
    retry once rather than crash the whole mission loop on one bad read."""
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    responses = iter(["garbled response with no quotes", "/controls/flight/aileron = '0' (double)\r\n/> "])
    bridge._send = lambda line: next(responses)
    assert bridge.get("/controls/flight/aileron") == "0"
    bridge.close()


def test_get_raises_after_two_malformed_replies(mock_fg_server):
    port, _ = mock_fg_server
    bridge = FlightGearBridge(host="localhost", port=port)
    bridge._send = lambda line: "still garbled, no quotes"
    with pytest.raises(ValueError):
        bridge.get("/controls/flight/aileron")
    bridge.close()
