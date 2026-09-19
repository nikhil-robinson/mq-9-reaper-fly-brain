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
pkill -9 -f 'python.*f15_intercept' 2>/dev/null || true
pkill -9 -f 'flatpak run.*fgfs' 2>/dev/null || true
pkill -9 -f 'bwrap.*fgfs' 2>/dev/null || true
sleep 3

LOG="${TMPDIR:-/tmp}/fg.log"
nohup flatpak run --command=fgfs org.flightgear.FlightGear \
  --fg-aircraft="$HOME/.fgfs/Aircraft" --aircraft=f15c \
  --airport=KXTA --runway=03L --altitude=15000 --vc=300 --in-air \
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
exit "$rc"
