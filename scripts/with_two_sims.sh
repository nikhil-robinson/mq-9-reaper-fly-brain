#!/usr/bin/env bash
# Launch TWO FlightGear instances linked by peer-to-peer multiplay, wait until
# both have a live FDM and loaded Nasal, then run the given python script
# against both. Same reasoning as with_sim.sh: telnet opens well before Nasal
# finishes loading, and connecting in that window silently produces a run
# where the engines never start.
#
# No fgms server is needed: --multiplay=out points straight at the other
# instance's --multiplay=in port on localhost.
#
# Ports (fixed; dogfight.py and mp_probe.py assume these):
#            telnet  native-fdm  native-ctrls  mp-in  mp-out  callsign
#   BLUE      5501       5504         5505      5000  ->5001   BLUE1
#   RED       5601       5604         5605      5001  ->5000   RED1
set -euo pipefail
cd "$(dirname "$0")/.."

pkill -9 -f 'python.*dogfight|python.*mp_probe' 2>/dev/null || true
pkill -9 -f 'flightgear.*AppImage' 2>/dev/null || true
pkill -9 -f '[f]gfs' 2>/dev/null || true
sleep 3

FG_APPIMAGE="${FG_APPIMAGE:-$HOME/Downloads/flightgear-2024.1.7-linux-amd64.AppImage}"
FG_ROOT_DIR="${FG_ROOT_DIR:-$HOME/FlightGear/fgdata_2024_1}"

# Two instances on this box is MEMORY-bound, not framerate-bound. Measured:
# fps came in at 57/54 (comfortably over fox2.nas:39's ~25fps guidance
# threshold), but with 7.3GB total RAM the pair pushed usage to 5.6GB plus
# ~3.9GB of swap and the driving python process was killed silently mid-run
# with no traceback. So these flags target FOOTPRINT, not pixels: scenery
# paging and cloud layers are what cost, and we have fps to spare.
VIS_ARGS="--disable-random-objects --disable-ai-traffic
          --visibility=8000 --fog-fastest
          --disable-clouds --disable-clouds3d
          --disable-sound
          --prop:/sim/rendering/shaders/quality-level=0
          --prop:/sim/rendering/max-paged-lod=40
          --prop:/sim/rendering/texture-compression=on
          --geometry=640x480"

launch() {  # $1=callsign $2=telnet $3=fdm $4=ctrls $5=mp_in $6=mp_out $7=lat $8=lon $9=hdg
    nohup "$FG_APPIMAGE" \
      --fg-root="$FG_ROOT_DIR" \
      --fg-aircraft="$HOME/.fgfs/Aircraft" --aircraft=f15c \
      --callsign="$1" \
      --lat="$7" --lon="$8" --heading="$9" \
      --altitude="${FG_ALT:-18000}" --vc=400 --in-air \
      --telnet="$2" \
      --native-fdm=socket,out,20,,"$3",udp \
      --native-ctrls=socket,in,20,,"$4",udp \
      --multiplay=in,10,,"$5" \
      --multiplay=out,10,localhost,"$6" \
      --disable-real-weather-fetch --timeofday=noon $VIS_ARGS \
      > "/tmp/fg_$1.log" 2>&1 &
}

# CO-HEADING offset start at 18000ft: RED 6km ahead of BLUE on the same
# heading, both at 400kt. NOT a head-on merge -- confirmed live that a head-on
# start is destroyed by its own setup time: two jets closing at 800kt cover
# 6.5km in ~16s, while sequentially starting engines and clearing the FCS
# locks on two instances takes ~60s, so they merged and were already tens of
# km apart and opening before a single radar read happened (designation
# returned active_u=nil for exactly this reason).
#
# Co-speed co-heading separation is stable, so the engagement geometry still
# exists whenever setup finishes; CombatPilot then generates the maneuvering
# from there. FG_START_OFFSET_M can widen it.
launch BLUE1 5501 5504 5505 5000 5001 37.26000 -115.80000 45
sleep 5
launch RED1  5601 5604 5605 5001 5000 37.29815 -115.75204 45

echo "mem before run:"; free -m | sed -n '2p'

.venv/bin/python - <<'PY'
import sys, time
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
from f15_intercept import Telnet
ready = {5501: False, 5601: False}
for i in range(120):  # 240s cap
    for port in list(ready):
        if ready[port]:
            continue
        try:
            t = Telnet(port=port)
            fdm = t.get("/sim/signals/fdm-initialized")
            n2 = t.get("/engines/engine[0]/n2")
            t.close()
            if fdm == "true" and n2 is not None:
                ready[port] = True
                print(f"  telnet {port} READY after {i*2}s", flush=True)
        except Exception:
            pass
    if all(ready.values()):
        print("BOTH SIMS READY", flush=True); sys.exit(0)
    time.sleep(2)
print(f"SIMS NEVER BECAME READY: {ready}"); sys.exit(1)
PY

RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
set +e
timeout --signal=INT --kill-after=15 "$RUN_TIMEOUT" .venv/bin/python -u "$@"
rc=$?
set -e
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "TIMEOUT: killed after ${RUN_TIMEOUT}s"
fi
pkill -9 -f 'flightgear.*AppImage' 2>/dev/null || true
pkill -9 -f '[f]gfs' 2>/dev/null || true
exit "$rc"
