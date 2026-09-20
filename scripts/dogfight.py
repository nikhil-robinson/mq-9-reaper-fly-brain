"""F-15 vs F-15 dogfight arena. Drives BOTH instances from one process.

Launch via scripts/with_two_sims.sh -- never by hand. It restarts both
instances clean and waits for the FDM AND Nasal on each; telnet opens well
before Nasal finishes loading, and connecting in that window silently produces
a run where the engines never start.

Phases, selected by env:
  (default)   both jets fly under CombatPilot, no weapons  -- proves the dual
              NativeBridge, the port split and achieved framerate
  WEAPONS=1   radar lock, real AIM-9 launch, kill resolution, flares

Why two instances: a single instance flies one FDM, so an enemy that really
runs fox2.nas -- real missile guidance, real flare seduction -- has to be its
own process. Confirmed in Phase 0 that the pair see each other as MP contacts
and that the radar handles a multiplayer-type contact correctly.
"""
import os
import sys
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from f15_intercept import Telnet, connect_stick, g_to_elevator, startup
from fly_brain.combat import CAS_ROLL_RATE_FULL_DPS, CombatPilot
from fly_brain.geo import angle_diff_deg, bearing_deg, haversine_m
from fly_brain.native_protocol import NativeBridge

STEP_HZ = 20
SPEED_UP = int(os.environ.get("SPEED_UP", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "2400"))
WEAPONS = os.environ.get("WEAPONS", "0") == "1"
RED_AGGRESSION = float(os.environ.get("RED_AGGRESSION", "1.0"))
M_TO_NM = 1.0 / 1852.0


class Jet:
    """One FlightGear instance, its pilot, and its two sockets."""

    def __init__(self, callsign, telnet_port, fdm_port, ctrls_port, aggression=1.0):
        self.callsign = callsign
        self.telnet = Telnet(port=telnet_port)
        self.bridge = NativeBridge(fdm_port=fdm_port, ctrls_port=ctrls_port)
        self.pilot = CombatPilot(aggression=aggression, callsign=callsign)
        self.alive = True
        self.telem = {}
        self.foe_node = None
        self.locked = False
        self.shots = []        # [(step_fired, target_jet)]
        self.kills = 0
        self.min_sep_m = 1e9

    def setup(self):
        print("=== %s setup" % self.callsign, flush=True)
        startup(self.telnet)        # engines, crank release, FCS lock clear
        connect_stick(self.telnet)  # assert the stick really is connected

    def read(self):
        self.telem = self.bridge.get_telemetry()
        return self.telem

    def apply(self, cmd):
        """g-demand and roll-rate demand through the calibrated decode layer.

        Never raw surface deflections: this airframe's pitch axis is a
        g-command system and its roll axis a rate-command system.
        """
        self.bridge.send_controls(
            aileron=max(-1.0, min(1.0, cmd.roll_rate_dps / CAS_ROLL_RATE_FULL_DPS)),
            elevator=g_to_elevator(cmd.g_cmd),
            rudder=0.0,
            throttle=cmd.throttle,
            speedup=SPEED_UP,
        )

    def close(self):
        self.bridge.close()
        self.telnet.close()


def build_telem(me, foe):
    """Pilot input: own state plus what the radar would give about the foe."""
    rng_m = haversine_m(me.telem["lat_deg"], me.telem["lon_deg"],
                        foe.telem["lat_deg"], foe.telem["lon_deg"])
    return dict(
        lat=me.telem["lat_deg"], lon=me.telem["lon_deg"],
        alt_ft=me.telem["altitude_m"] / 0.3048,
        heading_deg=me.telem["heading_deg"], roll_deg=me.telem["roll_deg"],
        airspeed_kt=me.telem["airspeed_kt"],
        climb_rate_fps=me.telem["climb_rate_fps"],
        tgt_lat=foe.telem["lat_deg"], tgt_lon=foe.telem["lon_deg"],
        tgt_heading_deg=foe.telem["heading_deg"],
        tgt_speed_kt=foe.telem["airspeed_kt"],
        tgt_range_nm=rng_m * M_TO_NM,
        has_lock=me.locked,
    )


def score(jet, steps):
    return dict(callsign=jet.callsign, alive=jet.alive, kills=jet.kills,
                missiles_fired=jet.pilot.missiles_fired,
                flares_used=jet.pilot.flares_used,
                min_separation_m=round(jet.min_sep_m), steps=steps)


def main():
    blue = Jet("BLUE1", 5501, 5504, 5505, aggression=1.0)
    red = Jet("RED1", 5601, 5604, 5605, aggression=RED_AGGRESSION)
    for jet in (blue, red):
        jet.setup()

    # Each pilot defends the altitude it actually spawned at, read from the FDM
    # rather than assumed: the launcher asks for FG_ALT but the sim settles
    # somewhere near it, and a wrong target altitude is a standing g bias.
    for jet in (blue, red):
        jet.read()
        jet.pilot.target_alt_ft = jet.telem["altitude_m"] / 0.3048
        print("  %s defending %.0fft" % (jet.callsign, jet.pilot.target_alt_ft),
              flush=True)

    print("fps: BLUE=%s RED=%s (fox2 guidance degrades below ~25)" % (
        blue.telnet.get("/sim/frame-rate"), red.telnet.get("/sim/frame-rate")),
        flush=True)

    for jet in (blue, red):
        jet.read()

    step = 0
    for step in range(MAX_STEPS):
        for jet in (blue, red):
            jet.read()

        sep_now = haversine_m(blue.telem["lat_deg"], blue.telem["lon_deg"],
                              red.telem["lat_deg"], red.telem["lon_deg"])
        blue.min_sep_m = min(blue.min_sep_m, sep_now)
        red.min_sep_m = min(red.min_sep_m, sep_now)

        for me, foe in ((blue, red), (red, blue)):
            if not me.alive:
                continue
            t = build_telem(me, foe)
            # RWR-grade threat only: inbound plus bearing, never perfect truth.
            # The arena knows a shot is in the air because it commanded it, but
            # a fight decided by omniscience is not a fight.
            inbound = any(target is me for (_, target) in foe.shots)
            threat = dict(
                missile_inbound=inbound,
                missile_bearing_deg=(angle_diff_deg(
                    bearing_deg(me.telem["lat_deg"], me.telem["lon_deg"],
                                foe.telem["lat_deg"], foe.telem["lon_deg"]),
                    me.telem["heading_deg"]) if inbound else None),
            )
            cmd = me.pilot.step(t, threat)
            me.apply(cmd)

            if step % 20 == 0:
                print("  %s step %d %s rng=%.2fnm alt=%.0f spd=%.0f "
                      "roll=%+.0f g=%.2f thr=%.2f" % (
                          me.callsign, step, cmd.state, t["tgt_range_nm"],
                          t["alt_ft"], t["airspeed_kt"], t["roll_deg"],
                          cmd.g_cmd, cmd.throttle), flush=True)

        if not (blue.alive and red.alive):
            print("ENGAGEMENT OVER at step %d" % step, flush=True)
            break

        time.sleep(1.0 / (STEP_HZ * SPEED_UP))

    for jet in (blue, red):
        print("SCORE %s" % score(jet, step), flush=True)
    for jet in (blue, red):
        jet.close()


if __name__ == "__main__":
    main()
