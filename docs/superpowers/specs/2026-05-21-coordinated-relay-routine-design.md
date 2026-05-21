# Coordinated Five-Rod Relay Routine — Design

**Date:** 2026-05-21
**Status:** Approved for planning

## Summary

A choreographed routine that moves the puck through all five rods in sequence —
Left D → Right D → Center → Left Wing → Right Wing → **SHOT** — with every rod
touching the puck once. Each receiving rod follows the puck's actual detected
position before passing (robust to imperfect passes). Rods coordinate via
vision: the next rod acts only once the camera confirms the puck has reached it.

## Goals

- Move the puck rod-to-rod across all five rods and finish with a shot.
- Each receiving rod positions itself dynamically to the puck's real location
  (Approach B — puck-following), so a slightly off pass is still recoverable.
- Coordinate rods with vision gates: a rod acts only when the puck is confirmed
  in its zone.
- Reuse the existing playbook structure and calibration conventions.

## Non-Goals

- No physics simulation or optimal-action search (that is `engine/planner.py`'s
  job for gameplay; this routine is a fixed choreography).
- No calibrated values delivered by this work — placeholder `# TODO: calibrate`
  values only, consistent with the rest of `robot/playbook.py`.
- No open-loop / no-vision fallback mode.

## Architecture & Files

| File | Change |
| ---- | ------ |
| `robot/playbook.py` | Add the `RELAY` definition (relay leg data). |
| `robot/routine.py` | **New.** Relay orchestrator: connection, vision polling, coordinate math, leg loop. |
| `routine.py` | **New, top-level.** Thin CLI entry point, styled like `run_play.py`. |
| `robot/const.py` | Add relay tuning constants. |
| `robot/vision.py` | Refactor: extract `detect_puck(machine, ...)` taking an existing connection. |

One shared `RobotClient` connection is opened for the whole relay (as in
`move.py`), not one connection per command — the relay polls vision and drives
motors many times in quick succession.

### Vision refactor

`get_puck_game_coordinates()` currently opens and closes its own connection.
Extract the detection body into:

```python
async def detect_puck(machine, camera_name="C270"):
    """Return the puck's game-space (x, y) using an existing connection,
    or (None, None) if no puck is detected."""
```

The `camera_name` default stays `"C270"` so the legacy detection path is
unchanged; the relay passes `RELAY_CAMERA` (`"dynamic-crop"`) explicitly at
each call site.

`get_puck_game_coordinates()` keeps its current behavior by opening a
connection and delegating to `detect_puck()`. The relay calls `detect_puck()`
repeatedly on its shared connection.

## The Relay Data Model

`RELAY` is an ordered list of legs, one per rod, in relay order. Each leg:

```python
{
    "player": PlayerID.LEFT_D,
    "receive_r": 90,                                          # neutral catch angle (TODO: calibrate)
    "pass_step": {"r": 300, "rpm": 400, "direction": "cw"},   # flick to next rod (TODO: calibrate)
}
```

- The final leg's `pass_step` is the **shot** instead of a pass.
- `t` (translation) is intentionally **not** stored — it is computed live from
  the puck position (Approach B).
- All `r` / `rpm` / `direction` values start as `# TODO: calibrate` placeholders.

Relay order: `LEFT_D, RIGHT_D, CENTER, LEFT_WING, RIGHT_WING`.

## Orchestration Flow

Per leg, in order:

1. **Detect** — read the puck's current game position via `detect_puck()`
   (`vision-1` service, `dynamic-crop` camera).
2. **Position** — compute the receiving rod's translation:
   `t = clamp((puck_y − min_y_rod) / (max_y_rod − min_y_rod), 0, 1)`.
   Four rods (`center`, `right_wing`, `right_d`, `left_d`) already have
   `min_y_* / max_y_*` bands in `engine/constants.py`; **left wing has none and
   must be added** (`min_y_left_wing`, `max_y_left_wing`, plus `left_wing_x` for
   the gate). With those, positioning is plain normalization — no mapping table.
   Send `{"t": t, "r": receive_r}`.
3. **Pass** — send the leg's `pass_step` (or the shot, on the last leg).
4. **Gate** (every leg except the last) — poll vision every
   `RELAY_VISION_POLL_INTERVAL_S` until the puck's **x** is within
   `RELAY_GATE_TOLERANCE_PX` of the next rod's x. Then begin the next leg.

### Coordinate model — pixels, not mm

- `vision-1` returns camera pixels; `scale_puck_coords()` maps those to **game
  pixels** (the engine's 450 × 837.5 px space). No mm anywhere in this path.
- Rod zones in `engine/constants.py` are also in game pixels.
- The vision gate is a pure game-pixels comparison.
- mm exists only inside the hockey-player module: the `t` field is a normalized
  `[0,1]` value the module maps onto `[min_translation_mm, max_translation_mm]`.
  Python never computes mm — it sends normalized `t`.

### Geometric assumption

Rods sit at distinct **x**-positions; players slide along **y**. Therefore the
puck travels in **x** between rods (the gate axis) and rods follow the puck's
**y** (the positioning axis). This is the one assumption to confirm on
hardware — if reversed, it is a single x↔y swap in `routine.py`.

## Error Handling & Cleanup

- **No puck detected at start**, or puck not on Left D's rod (puck_x not within
  tolerance of `left_d_x`) → print a clear message, abort, home all rods.
- **Gate timeout** (puck never reaches the next rod within
  `RELAY_GATE_TIMEOUT_S`) → abort, report which leg failed, home all rods.
- **DoCommand error** → abort, home all rods.
- Cleanup homes **all five rods** once, at the very end (on success or abort),
  via `{"t": 0, "r": 0}` per rod. Rods are **not** homed mid-relay — a homing
  rod could swat the puck in transit.

## CLI

`routine.py` is a thin entry point:

```
python3 routine.py            # run the relay
python3 routine.py --dry-run  # print each leg's planned commands and gates, no hardware
```

`--dry-run` prints the resolved plan (per-leg poses, pass steps, gate targets)
without connecting — lets the logic be verified before risking the puck.

## Tuning Constants (`robot/const.py`)

| Constant | Meaning |
| -------- | ------- |
| `RELAY_CAMERA` | Camera component name for vision (`"dynamic-crop"`). |
| `RELAY_GATE_TOLERANCE_PX` | How close (game px) the puck's x must be to a rod to count as "arrived". |
| `RELAY_GATE_TIMEOUT_S` | Max wait for the puck to reach the next rod before aborting. |
| `RELAY_VISION_POLL_INTERVAL_S` | Delay between vision polls while gating. |

## Testing

- **Unit tests** for pure functions: `puck_y_to_t()` (endpoints, clamping) and
  the gate predicate (puck_x within tolerance of a rod's x).
- **`--dry-run`** exercises the orchestration logic end-to-end with no hardware.
- **Hardware verification** is manual: run `--dry-run`, then a live run,
  watching for missed gates and mis-aimed passes.

## Risks

- **Zig-zag path.** The relay order crosses the table in x
  (325 → 137 → 200 → 278 → 50), so the puck passes other rods' x-positions in
  transit. Pass-angle calibration will be the hard part; expect tuning.
- **`min_y_* / max_y_*` vs. module `t` range.** The puck_y→t map assumes the
  module's `t=0/1` endpoints align with the constants' y-bands. If not, a
  per-rod calibration offset is needed. The puck-following design tolerates
  small errors but not a wrong scale.
- **Vision gate latency.** Each gate adds a poll-interval-scale stutter; the
  relay will not look perfectly fluid. Accepted trade-off for robustness.
