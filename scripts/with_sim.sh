#!/usr/bin/env bash
# Restart FlightGear clean, wait until the FDM+Nasal are actually ready, then
# run the given python script against it. ALWAYS use this instead of running a
# flight script by hand: a second run against a sim whose aircraft has already
# been flown (or crashed) is not a valid test, and FlightGear's telnet port
# opens well BEFORE the aircraft's Nasal has loaded -- connecting in that
# window silently produces a run where the engines never start.
set -euo pipefail
cd "$(dirname "$0")/.."

# Kill any previous flight script first: it still holds the native-fdm UDP
# port, and the next run dies on bind() with "Address already in use" before
# it ever reaches the sim (hit live).
pkill -9 -f 'python.*f15_intercept|python.*f15_obstacle_run|python.*f15_airshow' 2>/dev/null || true
pkill -9 -f 'flatpak run.*fgfs' 2>/dev/null || true
pkill -9 -f 'bwrap.*fgfs' 2>/dev/null || true
pkill -9 -f 'flightgear.*AppImage' 2>/dev/null || true
pkill -9 -f '[f]gfs' 2>/dev/null || true
sleep 3

# pkill -9 leaves FG_HOME's lock file behind, and the next instance then exits
# a couple of seconds into startup ("flightgear instance already running") with
# nothing useful in the log -- it looks exactly like a hang. Clear it.
rm -f "$HOME/.fgfs/fgfs_lock.pid" "$HOME/.fgfs/fgfs_exclusive.lock" 2>/dev/null || true

# FG_START=air (default) spawns at 15000ft already flying -- right for
# maneuver work. FG_START=ground spawns on the runway, needed for any
# sequence that begins with a takeoff or ends with a landing.
if [ "${FG_START:-air}" = "ground" ]; then
    START_ARGS="--on-ground"
else
    # FG_ALT: aerobatics need room. Terrain here is ~5400ft and the show's
    # hard deck is 4000ft AGL (~9400ft MSL), so a 15000ft start leaves only
    # ~5600ft of working room -- less than a Split-S consumes.
    START_ARGS="--altitude=${FG_ALT:-15000} --vc=300 --in-air"
fi

# AI_SCENARIO: name of a Scenarios/*.xml (no extension) to load a real AI
# aircraft at startup, e.g. static_targets_scenario -- FlightGear then owns
# all of that contact's properties itself, unlike a hand-built /ai/models
# node (see f15_props.py's spawn_target docstring for why that matters).
SCENARIO_ARGS=""
if [ -n "${AI_SCENARIO:-}" ]; then
    SCENARIO_ARGS="--ai-scenario=${AI_SCENARIO}"
fi

LOG="${TMPDIR:-/tmp}/fg.log"
# AppImage + our own writable fgdata, replacing the flatpak. The flatpak's
# FG_ROOT was read-only, which cost real time: AI scenarios are only loadable
# if they are in the catalog FGAIManager builds at startup by globbing
# FG_ROOT/AI, and nothing else is scanned -- byte-identical content loaded
# (rc=1) from FG_ROOT/AI but failed (rc=0) from FG_HOME/AI and from a --data
# dir. With a writable FG_ROOT we just drop scenario XML in FG_ROOT/AI and it
# is picked up normally, no runtime catalog injection needed.
#
# --fg-root is safe HERE specifically because the data package version
# (fgdata_2024_1/version == 2024.1.7) matches the AppImage build exactly. The
# standing warning about --fg-root triggering an unprompted full re-download
# applies to a MISMATCHED or overlay directory that fails FlightGear's
# integrity check, not to a matching data package.
FG_APPIMAGE="${FG_APPIMAGE:-$HOME/Downloads/flightgear-2024.1.7-linux-amd64.AppImage}"
FG_ROOT_DIR="${FG_ROOT_DIR:-$HOME/FlightGear/fgdata_2024_1}"
# BOMBABLE: the addon that makes AI aircraft actually fight -- attack the
# player, take damage and explode. A stock scenario AI aircraft only flies a
# flightplan and cannot fight back, and a hand-built /ai/models node has no
# damage model at all. Set BOMBABLE=0 to launch without it.
BOMBABLE_ADDON="${BOMBABLE_ADDON:-$HOME/FlightGear/addons/Bombable-6.0}"
ADDON_ARGS=""
if [ "${BOMBABLE:-1}" = "1" ] && [ -d "$BOMBABLE_ADDON" ]; then
    ADDON_ARGS="--addon=$BOMBABLE_ADDON"
fi

nohup "$FG_APPIMAGE" \
  --fg-root="$FG_ROOT_DIR" $ADDON_ARGS \
  --fg-aircraft="$HOME/.fgfs/Aircraft" --aircraft=f15c \
  --airport="${FG_AIRPORT:-KXTA}" --runway="${FG_RUNWAY:-03L}" $START_ARGS $SCENARIO_ARGS \
  --telnet=5501 --disable-real-weather-fetch --timeofday=noon \
  --native-fdm=socket,out,20,,5504,udp \
  --native-ctrls=socket,in,20,,5505,udp > "$LOG" 2>&1 &

.venv/bin/python - <<'PY'
import sys, time
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
from f15_intercept import Telnet
for i in range(90):  # 180s cap -- FlightGear has never taken >10s here
    try:
        t = Telnet()
        fdm, n2 = t.get("/sim/signals/fdm-initialized"), t.get("/engines/engine[0]/n2")
        t.close()
        if fdm == "true" and n2 is not None:
            print(f"SIM READY after {i*2}s", flush=True); sys.exit(0)
    except Exception:
        pass
    time.sleep(2)
print("SIM NEVER BECAME READY"); sys.exit(1)
PY

# Hard wall-clock cap. Nothing in this pipeline should be able to hang
# indefinitely: the flight loop blocks on UDP telemetry, the sim can wedge
# under render load, and a run left going holds port 5504 and kills the NEXT
# run at bind() (hit live). Override with RUN_TIMEOUT=<seconds>.
RUN_TIMEOUT="${RUN_TIMEOUT:-2400}"
set +e
timeout --signal=INT --kill-after=15 "$RUN_TIMEOUT" .venv/bin/python -u "$@"  # -u: unbuffered, else a redirected log looks frozen
rc=$?
set -e
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "TIMEOUT: killed after ${RUN_TIMEOUT}s"
fi
pkill -9 -f 'flatpak run.*fgfs' 2>/dev/null || true
pkill -9 -f 'bwrap.*fgfs' 2>/dev/null || true
pkill -9 -f 'flightgear.*AppImage' 2>/dev/null || true
pkill -9 -f '[f]gfs' 2>/dev/null || true
exit "$rc"
