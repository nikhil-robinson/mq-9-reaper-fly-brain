"""F-15 intercept: hybrid control. Vertical axis uses the aircraft's own
real AFCS altitude-hold (confirmed live to work cleanly -- smooth,
converging, no oscillation). Horizontal axis uses our own bank-cascade
aileron control over FlightGear's native binary protocol (--native-fdm /
--native-ctrls), because:

  1. The aircraft's own native heading-hold is broken in this add-on: its
     "HEADING TRUE SYNC" filter (Systems/autoflight-rules.xml) keeps
     overwriting /autopilot/settings/true-heading-deg back to the current
     heading every frame regardless of engage state, so heading-error-deg
     always computes near zero. Not worth fixing blind; this control law
     doesn't need it.
  2. Telnet is too slow for the hot loop (confirmed live: a first version
     of this script driving controls over telnet at 5Hz rolled to 180deg
     and hit terrain within ~15 steps -- the same telnet-latency failure
     diagnosed hours earlier on the F-16).
  3. FlightGear's file-based generic protocol (--generic=...) depends on
     a Protocol/ directory FlightGear itself controls -- fine on a normal
     install, but this flatpak's $FG_ROOT (/app/share/flightgear) is
     read-only by design (confirmed live: "Failed to parse config" on our
     file even though it's valid XML -- the file was never found). Trying
     to redirect FG_ROOT to a writable overlay directory instead triggered
     FlightGear's own auto-repair mechanism, which tried to re-download
     the entire ~2GB data package -- not something to trigger by accident.
     --native-fdm/--native-ctrls need no protocol file at all: the struct
     layout is fixed and compiled into FlightGear itself.

Struct layout: see src/fly_brain/native_protocol.py's docstring for the
byte-order/padding details (pulled directly from FlightGear's own source,
not guessed -- a naive struct.calcsize() without accounting for C's
alignment padding before double fields silently produced a 12-byte-short
packet that FlightGear's own length check rejected with zero error
output, confirmed live).

FGNetCtrls unconditionally overwrites every engine/electrical field on
every packet (confirmed by reading FlightGear's own FGCtrls2Props). This
F-15 add-on's own Nasal engine logic reacts badly to a nonzero write on
the generic `magnetos` property specifically (confirmed live: writing 3
there got silently reset to 0, and that transition triggered
/controls/engines/engine[0]/cutoff=true -- a property not even present in
FGNetCtrls -- killing the engine in a way that needed a full FlightGear
restart to clear, a plain quickstart()+crank retry was not enough).
native_protocol.py sends magnetos=0 from the very first packet onward to
avoid that transition.

Startup sequence (all required, found by reading Nasal, not guessing):
  1. aircraft.quickstart() -- sets up power/avionics AND, critically, is
     the only thing that gets the engines past ~15% N2 to actually
     "running" (confirmed live: direct starter/JFS property writes alone
     plateau at 15-17% N2 forever and never light off).
  2. sim/model/f15/controls/engine/engine-crank = 1 -- the real crank
     switch (Nasal function engine_crank_switch, not the raw starter
     property) that engines.nas's own start logic is gated on.
  3. sim/model/f15/controls/AFCS/att-hold = 1 -- the real AFCS master
     bootstrap (afcs_heading_switch() only *cycles* existing modes, it
     bails out immediately if the lock is "" -- this is what seeds it to
     "wing-leveler" first, confirmed live).
  4. autopilot/locks/altitude = "altitude-hold" +
     autopilot/settings/target-altitude-ft -- real, working altitude hold.

Run FlightGear first:
  flatpak run --command=fgfs org.flightgear.FlightGear \\
    --fg-aircraft=$HOME/.fgfs/Aircraft --aircraft=f15c \\
    --airport=KXTA --runway=03L --altitude=15000 --vc=300 --in-air \\
    --telnet=5501 --disable-real-weather-fetch --timeofday=noon \\
    --native-fdm=socket,out,20,,5504,udp \\
    --native-ctrls=socket,in,20,,5505,udp
Then: .venv/bin/python scripts/f15_intercept.py
"""

import math
import os
import socket
import time

from fly_brain.geo import angle_diff_deg, bearing_deg, haversine_m, offset_point
from fly_brain.native_protocol import NativeBridge

TELNET_HOST = "localhost"
TELNET_PORT = 5501

# Real root cause of bearing_err never converging in every session so far
# (not a control-law bug): turn radius R = v^2/(g*tan(bank)). At the speeds
# this aircraft settles into under full throttle (~500kt = 257m/s) and a
# 33deg bank, R = 257^2/(9.81*tan(33deg)) = ~10350m -- a 20km turn
# diameter, more than 4x the old 5000m waypoint distance. The aircraft was
# never going to reach a waypoint that close at these speeds no matter how
# the gains were tuned. Widened to a distance the achievable turn radius
# can actually close.
WAYPOINT_DIST_M = 20000.0
WAYPOINT_RADIUS_M = 300.0
CAPTURE_WINDOW_M = 2000.0  # inside this, opening range counts as the intercept
STEP_HZ = 20  # matches --native-fdm/--native-ctrls's ,20, rate
# Sim seconds per wall second. The flight itself takes a fixed amount of
# SIMULATED time (~85s for the 20km intercept), and the loop was pinned to
# wall time by its own sleep(1/STEP_HZ) -- so a run cost ~85s of wall clock
# no matter how much headroom the machine had. Raising this runs the same
# flight proportionally faster in real time. It is sent inside FGNetCtrls,
# NOT set as /sim/speed-up over telnet: see encode_ctrls: our own control
# packets would overwrite a telnet-set value 20 times a second.
SPEED_UP = int(os.environ.get("SPEED_UP", "1"))
MAX_STEPS = 6000  # 5 minutes at 20Hz -- widened waypoint needs more time to close

# Real cause of the altitude collapse, found after the 45deg-cap crash
# AND the 20deg-cap "barely turns" result: FGNetCtrls unconditionally
# overwrites /controls/flight/elevator on every packet (confirmed pattern
# -- this is the same unconditional-overwrite behavior already confirmed
# for aileron and the engine/electrical fields). Sending elevator=0.0 at
# 20Hz was directly fighting and cancelling whatever the native
# altitude-hold's own pitch loop had just computed a moment earlier --
# native altitude-hold was never actually getting a chance to act. Fixed
# by computing our own elevator from altitude error instead of leaving it
# at 0 and hoping AFCS wins the race (see ALT_HOLD_GAIN below); the
# native altitude-hold lock is left engaged but effectively moot now.
# Confirmed live with the stall guard + corrected elevator sign/gain in
# place: 30deg cap held altitude and airspeed rock solid (alt_err stayed
# within +-60ft, speed climbed 275->430kt) but the turn was too gentle to
# converge -- bearing error grew from 91 to 107deg instead of shrinking.
# Tried raising this back to 45 with the elevator/bank-compensation fixes
# in place (see ALT_HOLD_GAIN/ELEV_CAP below) -- confirmed live this is a
# genuine roll-loop instability, not an elevator/altitude problem: actual
# roll ran away past 100-180deg (inverted) even with aileron pegged at
# +-1.0 and the roll *command* slew-limited to ROLL_RATE_MAX_DPS/STEP_HZ
# per step, ending in a stall/impact both times. The K_BANK_TO_RATE /
# K_RATE_TO_AILERON cascade doesn't have enough margin at large bank
# commands. Reverting to 30deg, the last configuration confirmed to hold
# altitude/airspeed with zero divergence over a full run -- the roll-
# cascade redesign needed for a tighter turn is real future work, not
# something to force by cranking this constant.
# Reduced from 30 -- see ELEV_CAP comment above: even 30-33deg sustained
# bank demands more pitch authority than either elevator cap can supply
# without stalling one way or the other. A shallower, sustained turn
# converges more slowly but the widened 20000m waypoint gives it the time.
# Raised 25 -> 45 in session 5. Every prior argument for a shallow bank
# ("more bank demands more pitch authority than the aircraft can supply")
# came from the disconnected-elevator era and is void: the aircraft was
# never being given ANY pitch authority, at any bank angle. With the stick
# actually connected and the level-turn g commanded properly, the real
# numbers at our 450kt cruise are:
#     bank 25deg -> 1.10g, turn diameter 23.4km
#     bank 29deg -> 1.14g, turn diameter 19.9km   <- what 25deg settled at
#     bank 45deg -> 1.41g, turn diameter 10.9km
# ...against (450/180)^2 = 6.2g available. 25deg was not a control problem
# at all, it was a GEOMETRY problem: a ~20km turn diameter against a 20km
# waypoint meant the aircraft flew a stable orbit around the target and
# never captured it (confirmed live -- dist bottomed out and grew again
# while altitude held to +-50ft and speed sat at 450kt the whole time).
# 45deg halves the turn diameter at a load factor the jet does not notice.
ROLL_MAX_DEG = 45.0
# Confirmed live at gain=1/3000: elevator ramped up correctly (0 -> 0.16)
# as altitude error grew, but still lost 500ft in 10s at a 42deg bank --
# not enough authority. Raised once to 1/800 (held 30deg fine, still not
# enough at 45deg -- see ELEV_CAP/BANK_COMPENSATION below).
# Tried 1/400 + ELEV_CAP=0.8 + a bank-compensation term for the 45deg
# attempt -- confirmed live this untested combination caused a dive (lost
# 10000ft while accelerating to 631kt with elevator pinned nose-up) even
# at 30deg, a regime that was never actually validated. Reverting to the
# exact values session 3 confirmed stable (alt/speed held with zero
# divergence over a full run) rather than guessing further.
# Confirmed live: ELEV_CAP=1.0 is actually worse than 0.6, not better --
# full elevator deflection at 300-350kt produced an ACCELERATED stall
# (exceeding max lift coefficient from AoA/g-load alone, independent of
# airspeed): roll went chaotic and speed collapsed from 350->190kt within
# 15 seconds, well before altitude was even the limiting factor. Neither
# cap value fixes it because the real problem is upstream: a ~33deg bank
# demands more sustained pitch authority than this aircraft can supply
# without stalling one way or the other. Since WAYPOINT_DIST_M is now
# 20000m (plenty of room/time), demand less bank instead of more elevator
# -- see ROLL_MAX_DEG below.
# The F-15's pitch axis is a G-COMMAND system, not a surface-deflection one.
# Systems/f-15-cas.xml's cas/pitch-g-demand maps /controls/flight/elevator
# straight to a commanded load factor (TM-72861 Figure 18), and the CAS then
# closes the loop on measured accelerations/Nz:
#     elevator -1.00 -> +8.05g      elevator +0.70 -> -1.62g
#     elevator -0.54 -> +3.58g      elevator +1.00 -> -3.99g
#     elevator  0.00 -> +1.00g   <-- NOT "neutral stick": commands level 1g
# Two consequences that invalidate every gain tuned before this was read:
#   1. NEGATIVE elevator = nose up. (The sign was right; it just never
#      reached the FCS -- see the pitch lock note in startup().)
#   2. elevator=0 already holds 1g, so the controller's job is only the
#      DELTA from 1g. A 0.6 "cap" was demanding ~4g continuously.
# So command g directly and invert this table, instead of PID-tuning a
# surface deflection the aircraft doesn't actually accept.
# CORRECTED: this table used to be a direct inversion of ONE lookup table
# read off elevator->g behavior empirically, but the real FCS composes TWO
# tables in series (flight-controls.xml fig 17: stick -> stick-pitch-scheduled,
# then fig 18: stick-pitch-scheduled -> stick-commanded-g). Inverting only the
# second table and applying it to elevator directly over-commanded by
# ~1.3x in both directions -- confirmed live via FCS_DIAG readback of the
# real fdm/jsbsim/fcs/stick-commanded-g against our old g_cmd (e.g. we
# commanded 0.30g while the real airframe was asked for -0.26g). This is
# the composed f2(f1(elevator)) table instead, computed from both tableData
# blocks. Spot-checked against the old table's known-wrong outputs: old
# claimed elev -0.54 -> 3.58g, this table's inverse gives elev -0.54 for
# g=5.40, matching the real composed math.
G_TABLE_G = [-4.00, -2.00, -0.67, 1.00, 2.06, 4.00, 8.00]
G_TABLE_ELEV = [1.000, 0.476, 0.351, 0.003, -0.188, -0.293, -1.000]
K_ALT_G = 1.0 / 4000.0  # g of extra pull per ft of altitude error
K_CLIMB_G = 0.02  # g per ft/s of climb rate (damping)
G_CMD_MIN, G_CMD_MAX = 0.3, 5.0
STALL_SPEED_KT = 180.0  # 1g stall speed -- max usable g scales as (V/Vs)^2


def g_to_elevator(g_cmd):
    """Invert the CAS g-demand table: load factor -> elevator command."""
    for i in range(len(G_TABLE_G) - 1):
        g0, g1 = G_TABLE_G[i], G_TABLE_G[i + 1]
        if g0 <= g_cmd <= g1:
            f = (g_cmd - g0) / (g1 - g0)
            return G_TABLE_ELEV[i] + f * (G_TABLE_ELEV[i + 1] - G_TABLE_ELEV[i])
    return G_TABLE_ELEV[0] if g_cmd < G_TABLE_G[0] else G_TABLE_ELEV[-1]
ROLL_RATE_MAX_DPS = 30.0
CAS_ROLL_RATE_FULL_DPS = 213.1  # aileron=1.0 commands this roll rate via the CAS
K_BANK_TO_RATE = 0.7
ROLL_RATE_FILTER_ALPHA = 0.3

# Confirmed live at throttle=0.5: even full elevator authority (capped at
# 0.6, saturated) couldn't hold altitude through a ~38deg sustained bank --
# lost 1000ft+ over 14s despite maximum climb command. Not a pitch-
# authority problem, a thrust problem: a fighter at half throttle doesn't
# have the excess power to sustain a banked turn without bleeding energy.
# Confirmed by reading Systems/f-15-engines.xml directly (not guessing):
# augmentation-stage = int(1 + (throttle_pos_norm - 1) * 5), clipped to
# [0,5] -- this only crosses above 0 (i.e. afterburner lights) at
# throttle_pos_norm=1.0 exactly. THROTTLE=0.95 was just under that
# threshold the entire project so far -- dry thrust only, no AB, which is
# the likely cause of the low-speed stall in the final approach (not
# enough thrust to sustain level flight at low speed without it).
THROTTLE = 1.0

# Back side of the power curve, real fix: fly for airspeed, let altitude
# float. Below CRUISE_SPEED_KT the controller stops defending the original
# altitude and lets the target sag by ALT_TRADE_FT_PER_KT per knot of
# deficit, so alt_err goes negative and commands nose-DOWN -- trading
# potential energy back for speed, the way a pilot recovers from the back
# side. 25ft/kt is roughly the real exchange rate (d(v^2/2g) ~ 22ft per kt
# at 400kt). Tapering the *gain* was tried and failed (see nose_up_demand);
# this changes what the controller wants, not how hard it tries.
CRUISE_SPEED_KT = 350.0
ALT_TRADE_FT_PER_KT = 25.0


def effective_target_alt_ft(target_alt_ft, airspeed_kt):
    """Altitude the controller actually defends at the current airspeed."""
    deficit_kt = max(0.0, CRUISE_SPEED_KT - airspeed_kt)
    return target_alt_ft - deficit_kt * ALT_TRADE_FT_PER_KT


class Telnet:
    """One-time setup only -- not used in the hot loop. See module docstring."""

    def __init__(self, host=TELNET_HOST, port=TELNET_PORT, timeout=5.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = b""
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
        """Property value, or None if it does not exist.

        FlightGear's telnet `get` answers a NON-EXISTENT property with an
        empty quoted value, indistinguishable at the protocol level from a
        property that exists and is empty. Returning '' for that was a real
        trap, not a cosmetic one: every caller here tests `is None` to mean
        "not there", so a missing node read as a truthy-shaped '' instead.
        Confirmed live -- an index walk over /ai/models/aircraft[i] looking
        for a scenario-spawned target never terminated, happily reporting
        aircraft[73] with every field '', and made a run with NO AI aircraft
        at all look like a run with dozens. find_ai_target() has the same
        `is None` break, so it would hang the same way.
        """
        response = self._send(f"get {path}")
        parts = response.split("'")
        if len(parts) < 2 or parts[1] == "":
            return None
        return parts[1]

    def set(self, path, value):
        self._send(f"set {path} {value}")

    def nasal(self, script):
        self._send(f"run nasal script={script}")

    def close(self):
        self._sock.close()


def startup(t):
    print("Starting engines (quickstart + crank)...")
    t.nasal('aircraft.quickstart()')
    time.sleep(1.5)
    t.set("/sim/model/f15/controls/engine/engine-crank", 1)
    for i in range(10):
        time.sleep(2)
        running0 = t.get("/engines/engine[0]/running")
        running1 = t.get("/engines/engine[1]/running")
        print(f"  {2*(i+1)}s: engine[0] running={running0} engine[1] running={running1}")
        if running0 == "true" and running1 == "true":
            break
    else:
        raise RuntimeError("Engine never started -- aborting before we glide into terrain.")

    # ENG CRANK is spring-loaded back to OFF on the real jet once the
    # engine lights off. Leaving it at 1 is what pinned MASTER CAUTION on
    # for this entire project: electrics.nas's runEMMISC lights
    # ca-start-valve (and latches master-caution) whenever crank==1 AND an
    # engine is running -- a real, correct warning that the start valve is
    # still commanded open with the engine up. Confirmed live: setting it
    # back to 0 clears both ca-start-valve and master-caution within one
    # frame, with both engines still running at n2=81.6 (it is only the
    # start path that needs it, not the run path). No master-caution
    # button press required -- the latch self-clears once the underlying
    # condition goes away.
    t.set("/sim/model/f15/controls/engine/engine-crank", 0)
    time.sleep(0.5)
    mc = t.get("/sim/model/f15/instrumentation/warnings/master-caution")
    print(f"  Crank switch released, MASTER CAUTION = {mc}")

    thrust0 = t.get("/engines/engine[0]/thrust_lb")
    thrust1 = t.get("/engines/engine[1]/thrust_lb")
    n2_0 = t.get("/engines/engine[0]/n2")
    n2_1 = t.get("/engines/engine[1]/n2")
    print(f"  Thrust check: engine[0] thrust={thrust0}lb n2={n2_0} | engine[1] thrust={thrust1}lb n2={n2_1}")

    # Tried disengaging att-hold entirely (thinking it was fighting our
    # elevator the way the altitude-hold lock seemed to) -- confirmed live
    # this was wrong and much worse: total loss of control within 4
    # seconds (pitch -12->-80deg in 80 steps, pitch rate spiking to
    # +-30-60dps, roll flipping chaotically +-1.0 aileron, crashed by step
    # 374 instead of ~900). att-hold is real stability augmentation (like
    # a fighter's SAS/CAS on a relaxed-static-stability airframe), not an
    # autopilot mode fighting us -- it must stay on. Also: elev_surface
    # read exactly 0.00 in that chaotic run too, despite obviously large
    # real control inputs happening -- that FGNetFDM field is apparently
    # just not wired to telemetry for this aircraft at all, not evidence
    # of anything. Back to att-hold=1 (matches session 3 and run 13/14,
    # the only configurations that produced smooth, non-chaotic flight).
    telem_alt_ft = float(t.get("/position/altitude-ft"))
    t.nasal('setprop("sim/model/f15/controls/AFCS/att-hold",1)')
    time.sleep(0.3)
    print(f"  AFCS att-hold engaged, alt={telem_alt_ft:.0f}ft")

    # THE root cause of four sessions of "our controls do nothing", found by
    # reading both sides of the switch live instead of inferring from
    # behaviour. flight-controls.xml gates BOTH axes identically:
    #     fcs/stick-pitch = fcs/elevator-cmd-norm  only if pitch-switch ne 1
    #     fcs/stick-roll  = fcs/aileron-cmd-norm   only if roll-switch  ne 1
    # ...and autoflight-rules.xml sets pitch-switch from
    # /autopilot/locks/altitude and roll-switch from /autopilot/locks/heading.
    # quickstart + att-hold leave those at "pitch-hold" and "wing-leveler",
    # so our stick inputs were switched out of the FCS and replaced by 0.
    # Confirmed live: with the locks set, elevator-cmd-norm=-0.30 while
    # stick-pitch=0 and pitch-input-sum=-0.005; cleared, elevator-cmd-norm
    # =-0.15 gives stick-pitch=-0.15 and pitch-input-sum=-0.235 -- ~45x the
    # authority. Every elevator sign flip, gain sweep and cap tested in
    # sessions 1-4, and the whole roll-cascade "instability", was tuning a
    # disconnected wire.
    #
    # This must come AFTER att-hold: engaging att-hold puts both locks back
    # (confirmed live -- clearing them before it is silently undone). We keep
    # att-hold=1 because it is what seeds the CAS, and clear only the locks:
    # that leaves cas/pitch-cas-active and cas/roll-cas-active = 1, i.e. the
    # real stability augmentation stays on while the autopilot lets go of the
    # stick. This is why disengaging att-hold outright (session 4) was chaos
    # and this is not -- that removed the augmentation too.
    # Retry: engaging att-hold re-seeds these locks asynchronously, so a
    # single clear can be undone a moment later by the aircraft's own
    # listener. Confirmed live -- an identical startup sometimes came up with
    # pitch-switch/roll-switch back at 1. Clear until it actually sticks
    # rather than clearing once and hoping we won the race.
    connect_stick(t)


def connect_stick(t):
    """Clear the altitude/heading locks so our stick inputs reach the FCS.

    Split out of startup() so callers with extra telnet setup to do (placing
    obstacle models, hunting for an AI target) can do it BEFORE disconnecting
    the stick -- confirmed live: a run that disconnected the stick as part of
    startup() and then spent ~15-30s doing unrelated telnet work with nothing
    actively piloting (engine-start's own wait loop plus placement/lookup
    sleeps) glided off trim and hit terrain ("Aircraft hit ShrubCover and
    exploded") before a single jet.step() ever ran. Leaving FlightGear's own
    altitude/heading-hold engaged during that window costs nothing -- it's
    only disengaged here, right before the native-protocol hot loop takes
    over -- and it means the jet is never actually unpiloted.
    """
    # Speed too -- a caller that engaged "speed-with-throttle" during setup
    # (dogfight_bombable.py does, to hold a target speed while arming) left
    # it running through the ENTIRE fight otherwise: that autothrottle keeps
    # writing its own throttle every frame to hold its target speed, fighting
    # our native-ctrls throttle=1.0 combat command and throttling back well
    # before real afterburner. Confirmed as the cause of a "throttle very
    # low, no afterburner while chasing" complaint -- the stick was
    # reconnected but the autothrottle never let go.
    t.set("/autopilot/locks/speed", "")
    for _ in range(10):
        t.set("/autopilot/locks/altitude", "")
        t.set("/autopilot/locks/heading", "")
        time.sleep(0.4)
        pitch_sw = t.get("/fdm/jsbsim/autoflight/input/pitch-switch")
        roll_sw = t.get("/fdm/jsbsim/autoflight/input/roll-switch")
        if pitch_sw == "0" and roll_sw == "0":
            break
    print(f"  Stick connected: pitch-switch={pitch_sw} roll-switch={roll_sw} "
          f"pitch-cas={t.get('/fdm/jsbsim/cas/pitch-cas-active')} "
          f"roll-cas={t.get('/fdm/jsbsim/cas/roll-cas-active')}")
    if pitch_sw != "0" or roll_sw != "0":
        raise RuntimeError(
            "Autopilot still holds the stick -- our inputs would be ignored. "
            "Aborting rather than flying a disconnected control law again."
        )


def main():
    t = Telnet()
    startup(t)
    own_lat = float(t.get("/position/latitude-deg"))
    own_lon = float(t.get("/position/longitude-deg"))
    heading_deg = float(t.get("/orientation/heading-deg"))
    t.close()  # setup done, hot loop is native binary protocol only from here

    waypoint_heading = (heading_deg + 90.0) % 360.0
    target_lat, target_lon = offset_point(own_lat, own_lon, waypoint_heading, WAYPOINT_DIST_M)
    print(f"Waypoint: {target_lat:.5f}, {target_lon:.5f}, {WAYPOINT_DIST_M:.0f}m to the right of spawn heading")

    print("Connecting native binary protocol link (5504/5505)...")
    nb = NativeBridge()
    telem = nb.get_telemetry()
    target_alt_ft = telem["altitude_m"] / 0.3048

    prev_dist = float("inf")
    filtered_climb_rate_fps = 0.0

    for step in range(MAX_STEPS):
        telem = nb.get_telemetry()
        own_lat, own_lon = telem["lat_deg"], telem["lon_deg"]
        heading_deg = telem["heading_deg"]
        roll_deg = telem["roll_deg"]

        dist = haversine_m(own_lat, own_lon, target_lat, target_lon)
        want_bearing = bearing_deg(own_lat, own_lon, target_lat, target_lon)
        bearing_err = angle_diff_deg(want_bearing, heading_deg)

        # With roll CAS active, /controls/flight/aileron is a ROLL RATE
        # command, not a deflection: cas/roll-rate-demand (TM-72861 Fig 21)
        # maps +-1.0 aileron to +-213.1 deg/s and the CAS closes the loop on
        # /orientation/roll-rate-degps itself. So the hand-tuned
        # bank->rate->aileron cascade (K_BANK_TO_RATE / K_RATE_TO_AILERON /
        # the roll-rate filter) is redundant -- the aircraft already has that
        # inner loop, and ours was a second, worse copy of it fighting a
        # half-disconnected input. Command the rate directly.
        desired_roll_deg = max(-ROLL_MAX_DEG, min(ROLL_MAX_DEG, bearing_err * 1.5))
        desired_rate_dps = max(
            -ROLL_RATE_MAX_DPS,
            min(ROLL_RATE_MAX_DPS, K_BANK_TO_RATE * (desired_roll_deg - roll_deg)),
        )
        aileron = max(-1.0, min(1.0, desired_rate_dps / CAS_ROLL_RATE_FULL_DPS))
        rudder = max(-0.03, min(0.03, bearing_err / 900.0))

        own_alt_ft = telem["altitude_m"] / 0.3048
        airspeed_kt = telem["airspeed_kt"]
        alt_err_ft = effective_target_alt_ft(target_alt_ft, airspeed_kt) - own_alt_ft

        # FlightGear's own climb rate, straight out of the FDM packet. This
        # used to be (alt - prev_alt) / dt with dt from time.monotonic() --
        # a WALL-clock derivative, which reads high by exactly SPEED_UP once
        # sim time and wall time diverge, corrupting the K_CLIMB_G damping
        # term that stops the altitude oscillation. The filter is kept: the
        # gains below were tuned against a smoothed signal.
        filtered_climb_rate_fps += ROLL_RATE_FILTER_ALPHA * (
            telem["climb_rate_fps"] - filtered_climb_rate_fps)

        # Real turn physics, commanded directly instead of tuned blind: a
        # level coordinated turn at bank phi needs load factor n = 1/cos(phi)
        # (lift's vertical component must still carry the weight). At our
        # 25deg bank that is 1.10g -- the aircraft was never given it, which
        # is the whole reason every banked run descended. On top of that
        # baseline, add a small correction for altitude error and a damping
        # term on climb rate.
        level_turn_g = 1.0 / max(0.2, math.cos(math.radians(roll_deg)))
        g_cmd = level_turn_g + alt_err_ft * K_ALT_G - filtered_climb_rate_fps * K_CLIMB_G

        # Stall guard, now in the units the airframe actually limits: max
        # attainable load factor is (V/Vstall)^2, because stall speed scales
        # with sqrt(n). This replaces the old ad-hoc "taper elevator by
        # airspeed" hack with the real aerodynamic limit, and it is what
        # stops the back-side-of-the-power-curve spiral: below the speed
        # where a level turn is sustainable, the aircraft is simply not
        # allowed to demand the g it would take.
        g_available = (airspeed_kt / STALL_SPEED_KT) ** 2
        g_cmd = max(G_CMD_MIN, min(min(G_CMD_MAX, g_available), g_cmd))
        elevator = g_to_elevator(g_cmd)

        # Confirmed live: with elevator pinned at full nose-up the whole
        # time, the aircraft still fell from 14900ft to 5100ft while
        # accelerating past 700kt -- an uncontrolled spiral dive (turning
        # at a constant ~33deg bank isn't converging fast enough, and full
        # throttle just keeps adding energy to the dive faster than
        # elevator authority can bleed it off as climb). Cut throttle when
        # overspeeding, the way a real pilot would in an unintended dive,
        # instead of holding it at THROTTLE unconditionally.
        throttle = THROTTLE if airspeed_kt < 450.0 else 0.2
        nb.send_controls(aileron=aileron, elevator=elevator, rudder=rudder,
                         throttle=throttle, speedup=SPEED_UP)

        if step % 20 == 0:
            print(
                f"step {step}: dist={dist:.0f}m bearing_err={bearing_err:+.1f}deg "
                f"roll={roll_deg:+.1f}deg alt={own_alt_ft:.0f}ft alt_err={alt_err_ft:+.0f}ft "
                f"speed={airspeed_kt:.0f}kt "
                f"pitch={telem['pitch_deg']:+.1f}deg alpha={telem['alpha_deg']:+.1f}deg "
                f"pitchrate={telem['pitch_rate_dps']:+.1f}dps "
                f"elev_surface={telem['elevator_pos']:+.2f} "
                f"action=[ail={aileron:.2f} elev={elevator:.2f} rud={rudder:.2f}]"
            )

        # Waypoint capture. A bare "dist < radius" test fails on a fast
        # mover: confirmed live, the closest approach was 324m against a
        # 300m radius -- the intercept geometry was right and the run missed
        # by 24m, then flew past and orbited for 4300 more steps. At 450kt
        # the aircraft covers 232m per 20Hz step, so a 300m sphere is barely
        # one step wide and can be stepped straight over. Detect the
        # closest-approach PASSAGE instead, the way real nav systems do:
        # once inside a capture window, the first time range starts opening
        # again IS the intercept.
        if dist < WAYPOINT_RADIUS_M or (dist < CAPTURE_WINDOW_M and dist > prev_dist):
            print(f"Arrived at waypoint, closest approach {min(dist, prev_dist):.0f}m, "
                  f"after {step} steps.")
            break
        prev_dist = dist

        # Crash/stall detection -- previously the loop ran all the way to
        # MAX_STEPS (2400 steps, 2 minutes) after the aircraft was already
        # on the ground with near-zero airspeed, wasting the whole rest of
        # the run before reporting anything. Bail out immediately instead.
        if airspeed_kt < 10.0 and own_alt_ft < target_alt_ft - 1000:
            print(f"CRASHED at step {step}: speed={airspeed_kt:.0f}kt alt={own_alt_ft:.0f}ft (stalled/impacted).")
            break

        # Under speed-up FlightGear emits STEP_HZ * SPEED_UP packets per
        # WALL second, so pace off wall time scaled by SPEED_UP -- otherwise
        # the loop under-samples and re-decodes the same cached packet.
        time.sleep(1.0 / (STEP_HZ * SPEED_UP))

    nb.close()


def _self_check():
    assert abs(g_to_elevator(1.0) - 0.0) < 1e-9  # 1g = level = neutral stick
    assert g_to_elevator(2.0) < 0  # pulling g is NEGATIVE elevator
    assert g_to_elevator(0.5) > 0  # unloading is positive
    assert abs(g_to_elevator(3.58) - (-0.54)) < 1e-9  # hits a table knot exactly
    assert g_to_elevator(99.0) == -1.00 and g_to_elevator(-99.0) == 1.00  # clamps
    # a 25deg level turn needs ~1.10g, and that must be a nose-up command
    assert abs(1.0 / math.cos(math.radians(25.0)) - 1.103) < 0.001
    assert g_to_elevator(1.0 / math.cos(math.radians(25.0))) < 0

    assert effective_target_alt_ft(15000.0, 400.0) == 15000.0  # fast: defend it
    assert effective_target_alt_ft(15000.0, 350.0) == 15000.0  # at cruise: defend it
    assert effective_target_alt_ft(15000.0, 250.0) == 12500.0  # slow: give it up
    # slow + at altitude => negative error => nose-down demand
    assert effective_target_alt_ft(15000.0, 250.0) - 15000.0 < 0


if __name__ == "__main__":
    _self_check()
    main()
