# F-15 vs F-15 Dogfight — Design

Date: 2026-09-20
Status: approved design, not yet implemented

## Goal

A real dogfight: two F-15s, each with a real FDM and real weapons, fighting
each other. Our jet flies a scripted combat AI; the enemy flies the same
controller and genuinely shoots back. Missiles are real AIM-9s with real
guidance, flares genuinely fool them, hits genuinely kill, and a killed jet
explodes and falls.

Decided with the user:

- **Our pilot:** scripted combat AI. Build the arena first; the connectome
  comes later behind the same interface.
- **Kill model:** full mutual (both sides can die), using FlightGear's own
  impact reporting where it exists and supplying the consequence it lacks.
- **Scale:** 1v1 proven end to end first, then raise the enemy count.
- **Enemy:** a second FlightGear instance over multiplayer, so its missiles
  are real rather than simulated.

### Scope boundary

**This spec covers 1v1 only.** A second RED jet is a follow-on that reuses
the same `CombatPilot` and a third instance (or a kinematic wingman); it gets
its own plan once 1v1 is proven. Nothing here should be built "ready for N"
beyond the fact that `CombatPilot` is already instantiated per jet — threat
prioritisation across multiple attackers is explicitly deferred.

**The connectome is out of scope.** It is priority C of the build plan. The
only commitment this design makes is that swapping `LIFBrain` in later is a
policy change behind `CombatPilot.step()`, not a plumbing change.

Also out of scope: ground attack, radar-guided (fox3) missiles, chaff,
multiplayer with anyone other than our own second instance, and damage
modelling more granular than alive/dead.

## Why a second instance, and not a cheaper enemy

A stock FlightGear scenario AI aircraft cannot maneuver (it flies a
flightplan) and cannot run `fox2.nas`. So with a single instance, the
enemy's missiles, their guidance, and their susceptibility to our flares
would all be our own Python approximations — a mockup of exactly the part
that is supposed to feel real, in a project whose premise is "no mockups".

Two instances linked by multiplayer means each jet sees the other as an **MP
contact**, which makes all of the following work as designed on both sides,
with no new code:

- `awg_9` radar scan, designation, `Target.new()`, seeker lock
- real AIM-9 launch and guidance from `fox2.nas`
- flare seduction via `Tgt.getFlareNode()` and `flare-resistance`
- `fox2.nas:40` states plainly that missiles hit MP aircraft considerably
  better than they hit scenario/AI objects, because of how AI objects are
  updated

`--multiplay={in|out},hz,address,port` is present in this build and supports
direct peer-to-peer, so **no `fgms` server is needed**: each instance sends
`out` to the other's `in` port on localhost.

## Architecture

```
                  ┌─────────────── dogfight.py (one process) ───────────────┐
                  │  arena loop: spawn → merge → engage → resolve → score   │
                  └────┬──────────────────────────────────────────┬─────────┘
          FDM 5504 ▲   │ ctrls 5505                    FDM 5604 ▲ │ ctrls 5605
          telnet 5501  ▼                              telnet 5601 ▼
        ┌─────────────────────────┐   MP 5000↔5001   ┌─────────────────────────┐
        │  BLUE  (ours, f15c)     │◄────────────────►│  RED  (enemy, f15c)     │
        │  callsign BLUE1         │   peer-to-peer   │  callsign RED1          │
        │  real FDM + fox2.nas    │   --multiplay    │  real FDM + fox2.nas    │
        └─────────────────────────┘                  └─────────────────────────┘
```

One Python process drives both jets: two `NativeBridge` instances (20Hz UDP
hot loop each) plus one `Telnet` each for one-time setup. No new transport is
introduced — this is the split the project already uses and the module
docstring already justifies.

`scripts/with_sim.sh` is generalised to take a role and a port offset rather
than being copy-pasted. Everything fixed in session 7 carries over unchanged:
`speedup` in `FGNetCtrls`, FDM `climb_rate`, the FCS lock clear and its retry,
`payload/armament/models`, and the `AIM-9` case symlink.

## Components

### `src/fly_brain/combat.py` — `CombatPilot`

Pure logic, no FlightGear dependency, so it is testable without a sim.
Instantiated **twice** — once per jet — with different aggression/skill
parameters. Both sides run the same code: one controller to debug, and an
enemy exactly as capable as us.

Interface, deliberately the one the connectome must eventually drive:

```
CombatPilot.step(telem, threat) -> (g_cmd, roll_rate_dps, throttle, fire, flares)
```

`g_cmd` and `roll_rate_dps` then go through the existing calibrated decode
layer (the CAS g-demand table inversion, and `/213.1` for roll rate). Those
constants are the motor-neuron-to-muscle-gain layer and are not re-tuned here.

State machine:

| state | entry condition | behaviour |
|---|---|---|
| `SEARCH` | start of engagement | fly toward the merge, radar scanning |
| `INTERCEPT` | contact at range | lead-pursuit closure, manage energy |
| `ENGAGE` | inside 0.5–5nm | pursuit curve, seek lock, fire when in envelope |
| `EVADE` | missile inbound | break turn into the threat, pop flares |
| `RESET` | overshoot, or no shot available | extend, regain energy, re-merge |

Three capabilities the existing intercept law does not have:

1. **Lead pursuit.** The current law points *at* the target, which is correct
   for a fixed waypoint and wrong for a dogfight: against a maneuvering jet
   it degenerates into a lag chase that never closes. Aim at the predicted
   position from the target's velocity vector.
2. **Energy / corner-speed management.** Maximum turn rate occurs at corner
   speed (~400–450kt for an F-15); below it there is not enough lift to pull
   g, above it the turn radius blows up. The existing stall guard
   `n_max = (V/Vs)²` already supplies the limit; this adds a throttle policy
   to stay near corner.
3. **Evasion geometry.** Break into the missile to force it to bleed energy,
   drive it toward the 3/9 line, and release flares. `flare-resistance` in
   `fox2.nas` makes this genuinely probabilistic, with a 50% extra chance of
   seduction at front aspect.

**Threat input is deliberately limited.** The arena knows RED fired, because
it commanded the shot — but the pilot is passed only what an RWR would give
(missile inbound, plus bearing). The fight must not be decided by
omniscience.

### `scripts/dogfight.py` — the arena

Owns: launching both instances, per-jet setup, the engagement loop, hit
resolution, kill consequences, scoring and telemetry.

### Weapons and firing discipline

Two weapons, because real dogfights end in guns:

- **AIM-9** — chain already proven in session 7. Envelope: 0.2–9nm, 27.5°
  seeker FOV, target faster than 15kt, lock required.
- **Gun (M61A1)** — for the endgame inside ~0.5nm where the AIM-9 cannot
  arm. Fired via `fdm/jsbsim/fcs/guntrigger`; `controls/armament/trigger`
  does not work on this airframe and is already recorded as a landmine.

Both sides obey the same firing rules: AIM-9 only with a lock inside the
envelope; guns inside ~0.6nm with the pipper on target. Without explicit
discipline an AI empties its rails at zero range and nothing ever hits.

### Countermeasures

Flares are already genuinely modelled, so this is mostly wiring rather than
new physics: `fox2.nas` reads the *target's* flare counter through
`getFlareNode()` and rolls against `flare-resistance`. Releasing a flare is
therefore an increment of a published counter property, and the enemy's real
missile genuinely may or may not be seduced.

### Damage, kill and explosion

```
fox2 detonates → impact report appears in /ai/models (warhead wt + miss distance)
                          ↓
        arena reads the report → lethality check against miss distance
                          ↓
   KILL → explosion model at the point + engines cut + controls frozen
          → the jet falls ballistic and strikes terrain
```

FlightGear supplies the *hit data* (its own impact reporting, which
`fox2.nas:58` documents as landing in `AI/models` and carrying warhead
weight). We supply the *consequence*, which FlightGear has no model for.

The same code path kills RED or BLUE, so "they can shoot me down" costs
nothing extra beyond a loss condition.

Explosion visuals reuse `Models/Stores/aim-9/explosion-N.xml` and
`motor-smoke-N.xml`, which resolve correctly now that the `AIM-9` case
symlink exists, scaled up for an airframe kill. A killed jet goes ballistic
rather than vanishing.

## Scoring and telemetry

Deterministic start geometry, as with the obstacle course: both jets spawned
at known positions and headings for a repeatable merge. Without that,
"did this change help?" cannot be answered.

Per engagement:

| metric | what it tells us |
|---|---|
| kills / deaths | the outcome |
| time-to-kill | how decisively |
| missiles expended per kill | firing discipline |
| flares used / missiles defeated | whether evasion actually works |
| min range, max sustained g | whether it is dogfighting or orbiting |

This is also the baseline the connectome is measured against later, the same
role the intercept run plays today.

## Testing

Layered, because every fault cleared in session 7 was only findable in
isolation:

- **Pure-function self-checks**, no sim required: lead-pursuit solution,
  corner-speed throttle policy, lethality vs miss distance, launch-envelope
  predicate. One `assert`-based in-file check each, matching the repo's
  existing style.
- **Phase 0 — MP link probe.** Two instances: do they see each other as
  contacts? This gates everything; no combat code is written before it is
  confirmed.
- **Phase 1 — both jets fly**, no weapons. Confirms the dual `NativeBridge`,
  the port split, and achieved framerate.
- **Phase 2 — one-sided.** BLUE engages a passive RED, reusing the proven
  lock/fire chain but now against an MP contact rather than an AI object.
- **Phase 3 — mutual.** RED shoots back; flares and evasion active.

**Framerate guard.** `fox2.nas:39` warns that missile guidance degrades below
~25fps, and we are now running two instances. Achieved framerate is measured
and recorded with every result: a missile that misses at 12fps tells us
nothing about the guidance. Both instances launch with reduced visuals.

## Unknowns to verify before building on them

Each of these is a quick property probe, and each is assumed by the design
above. Session 6's `FG_HOME/AI` dead end is the precedent for checking first.

1. **MP contact visibility** — two instances seeing each other as contacts
   over localhost peer-to-peer (Phase 0 gates on this).
2. **Which `sim/multiplay/generic/int[N]` carries the flare counter** across
   the MP link for this F-15.
3. **Where the impact report lands** — the shooter's property tree, the
   victim's, or both.
4. **Achieved framerate** with two instances, against the ~25fps guidance
   threshold.

## Risks

| risk | mitigation |
|---|---|
| Two instances too heavy; framerate below the guidance threshold | reduced visuals both instances; measure and record fps; `SPEED_UP` is available but note it cuts wall-clock, not render cost |
| MP peer-to-peer does not produce usable contacts | Phase 0 gates the whole design; fall back to a kinematic enemy with our own missile model, accepting it is an approximation |
| Impact report insufficient to judge lethality | fall back to computing miss distance ourselves from both jets' positions, which the arena already has |
| Dogfight never converges (both sides evade forever) | scoring exposes it directly (no kills, low max-g); tune aggression asymmetrically |

## Phase 0 findings (measured 2026-09-20)

Gate: **PASSED.** The two-instance MP design is viable.

| unknown | result |
|---|---|
| MP contact visibility | **YES**, both directions, repeatable: `blue_sees=['RED1']`, `red_sees=['BLUE1']`, under `/ai/models/multiplayer[N]` |
| achieved framerate | **31-57 fps**, above fox2.nas:39's ~25fps guidance threshold even with two instances |
| flare counter property | **deferred to Task 6** — `getFlareNode()` resolves off the contact object, so it reads `nil-contact` until a lock exists |
| impact report location | **deferred to Task 6** — needs a real shot, which needs a lock |

### Two corrections to the design's assumptions

**1. The binding constraint is MEMORY, not framerate.** The spec's top risk was
two instances starving the renderer below the guidance threshold. Measured fps
was 31-57, so that risk is retired. What actually bit: 7.3GB total RAM, 5.6GB
used with ~3.9GB of swap consumed, and the driving python process was killed
silently mid-run with no traceback. `with_two_sims.sh` now trims FOOTPRINT
(scenery paging, cloud layers, sound, LOD) rather than resolution, because
there is fps to spare and no memory to spare.

**2. A head-on merge start defeats itself.** Two jets closing at 800kt cover
6.5km in ~16s, while sequentially starting engines and clearing the FCS locks
on two instances takes ~60s. The jets merged and were already far apart and
opening before the first radar read, which is what produced a designation
failure (`active_u=nil`) that looked like a radar fault. The start is now
CO-HEADING with a 6km offset, which is geometrically stable, and the
maneuvering comes from CombatPilot rather than from the spawn.

### The radar handles MP contacts correctly

Worth recording explicitly, because it retires a whole class of suspicion:
with the MP contact in view the radar reported `tgts=1 active_cs=RED1
first_cs=RED1 first_type=multiplayer airbone=1`. Callsigns match exactly with
no MP padding, and `airbone` is satisfied by `type == "multiplayer"` without
needing the `radar` child that a scenario AI object needs.

The only unmet condition was azimuth (`dev=-164deg` against a +-60deg field,
so `display=0`) — because nothing was pointing either jet at the other. That
is CombatPilot's job, not a property fix, so lock/fire verification moves to
Task 6 where a pilot is actually flying.
