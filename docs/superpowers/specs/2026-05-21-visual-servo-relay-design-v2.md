# Visual-Servo Relay Routine — Design v2

**Date:** 2026-05-21
**Status:** Proposed
**Supersedes:** `2026-05-21-coordinated-relay-routine-design.md`

## Why v2

The v1 relay assumed a static `puck_y → t` coordinate map. Hardware testing
disproved every assumption behind it:

- The puck class is `"orange"`, not `"green"` (fixed, committed).
- The camera→game geometry in `scale_puck_coords` is mirrored relative to the
  physical rods.
- The puck's position relative to a rod is a function of that rod's `(t, r)` —
  no static per-rod coordinate can describe it.

The relay must instead use **live visual feedback**. v2 replaces the coordinate
model with a closed-loop visual servo.

## Available hardware interfaces

- **`hockey-player` module** (one generic component per rod). `DoCommand`
  supports move (`{"t","r","rpm"/"power","direction"}`, blocking, returns
  `{"t_final","r_final"}`) and query (`{"cmd":"get_position"}` →
  `{"t","r","t_moving","r_moving"}`). `t` ∈ [0,1] is precise and repeatable;
  the module has no notion of camera/table space.
- **`vision-1`** on camera `dynamic-crop`: detects the puck, class `"orange"`,
  bounding boxes in camera pixels.
- **`Players--detector`** on camera `dynamic-crop`: detects `"Robot_Player"`,
  `"Opponent_Player"`, `"Robot_Goalie"` boxes in camera pixels.
- The `dynamic-crop` camera is **flaky** — it intermittently fails with
  `expected exactly 4 detections, got 5`. Every vision call needs retry.

## Coordinate frame

Everything operates in **camera pixels**. Both the puck and the players are
detected through the same `dynamic-crop` camera, so puck and player positions
are directly comparable. No table/mm conversion. `engine/constants.py`
geometry and `scale_puck_coords` are **not used** by the relay.

## Goals

- Relay the puck through all five rods and finish with a shot, starting from
  **wherever the puck is placed**.
- Each rod catches the puck by closed-loop visual servo — no static calibration.
- Survive the flaky camera.

## Non-Goals

- No physics simulation or planner integration.
- No calibrated pass values delivered by this work — pass angles stay as
  placeholders to be tuned on hardware.
- No table/world coordinate system.

## System design

### 1. Robust vision layer

A retry wrapper around every detection call: on `GRPCError` or empty result,
retry up to N times with a short sleep. Two functions:

- `detect_puck()` → puck box centre in camera pixels, or `None`.
- `detect_robot_players()` → list of `Robot_Player` box centres.

Both run on a shared `RobotClient` connection.

### 2. Rod identification (startup, once per run)

`Players--detector` returns several `Robot_Player` boxes with no rod labels.
To map each rod to its player box:

- For each rod: capture player boxes, command a small `r` wiggle (rotate and
  return), capture again. The box that moved is that rod's player.
- Result: `rod → current player-box centre`. Thereafter the rod's player is
  tracked frame-to-frame as the `Robot_Player` box nearest its last known
  centre (only one rod moves at a time, so this is unambiguous).

### 3. Servo-axis probe (startup, once per rod)

To learn which camera direction `t` moves the player:

- Command the rod to `t=0.2`, then `t=0.8` (gentle, rpm mode). Record the
  player-box centre at each.
- The delta is the rod's `t`-axis unit vector in camera pixels, plus a rough
  gain (pixels per unit `t`).

### 4. The catch — visual servo

For a rod to catch the puck:

1. Detect the puck `P` and this rod's player `Q` (camera pixels).
2. Project `P − Q` onto the rod's `t`-axis → signed lateral error `e` (pixels).
3. If `|e| ≤ catch_tol_px` → aligned, catch complete.
4. Else: read current `t` from the module, command
   `t_new = clamp(t + k·e, 0, 1)` where `k` derives from the probe gain.
5. Re-detect, repeat. Cap at `max_servo_iters`; if not converged → abort.

The rod holds a fixed "catch" rotation `r` (stick down) during the servo.

### 5. The pass

Once aligned, the rod swings `r` (a fast rotation: the leg's `pass_step`,
`{"r","rpm","direction"}`) to strike the puck toward the next rod. Pass values
are per-leg placeholders, tuned on hardware.

### 6. The relay loop

- Detect the puck. Determine the **start rod**: the rod whose player box is
  nearest the puck along the non-`t` (longitudinal) axis.
- Run the fixed leg order from the start rod onward
  (Left D → Right D → Center → Left Wing → Right Wing → shot). Each leg:
  catch (servo) → pass. The last leg's pass is the shot.
- A leg that cannot converge its servo (puck never found/reached) aborts the
  relay.
- On exit — success or abort — all five rods are homed (`t=0, r=0`).

## Files

- `robot/vision.py` — add retry wrapper; add `detect_robot_players()`;
  `detect_puck()` returns the puck box centre with retry.
- `robot/routine.py` — substantially rewritten: rod identification, servo-axis
  probe, the servo `catch`, the relay loop. The v1 coordinate functions
  (`puck_y_to_t`, `puck_reached_rod`, `_ROD_X`, `_ROD_Y_BAND`) are removed.
- `robot/playbook.py` — `RELAY` keeps `player` and `pass_step`; `receive_r`
  becomes the fixed catch angle.
- `robot/const.py` — relay constants become servo constants: `catch_tol_px`,
  `max_servo_iters`, servo gain, retry count, poll interval.
- `routine.py` — CLI unchanged in spirit (`--dry-run` prints the leg order and
  configured constants).
- `engine/constants.py` left-wing additions from v1 are now unused by the
  relay; left in place (harmless, used elsewhere).

## Testing

- **Unit:** the pure servo math — error projection onto the `t`-axis, the
  proportional step with clamping, the start-rod selection — with synthetic
  pixel inputs. `--dry-run` for the leg order.
- **Hardware:** the servo loop, rod identification, and probe are inherently
  hardware-validated, run incrementally on the rig.

## Risks

- **Player-box identification drift** — if two rods' players overlap in the
  image, nearest-box tracking can mis-assign. Mitigated by moving one rod at a
  time and re-confirming with the wiggle if a box is ambiguous.
- **Servo gain** — too high oscillates, too low is slow. Starts conservative,
  tuned on hardware.
- **Flaky camera** — retry handles transient failures; a persistent camera
  fault still aborts the relay (correctly).
- **Pass aim** — the pass swing is still open-loop and needs hardware tuning.
