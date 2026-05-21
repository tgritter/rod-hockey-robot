# One-Rod Puck Control — Design (learn from reality)

**Date:** 2026-05-21
**Status:** Proposed
**Relates to:** supersedes the puck-handling intent of
`2026-05-21-visual-servo-relay-design-v2.md` for a single rod. The five-rod
relay is deferred until this foundation is solid.

## Why this design

Earlier relay attempts failed because puck control was hardcoded — fixed
sweeps, fixed rotations, static coordinate maps. Holding rotation constant
while translating rolls the puck off the stick blade; real control needs
translation and rotation coordinated, and that coordination depends on the
real, per-state geometry and contact physics of the rig.

This design builds **one rod** that controls the puck by **learning the rig's
real behaviour from data** — driving the rod and watching the puck through
`vision-1`, fitting a model to those observations. No simulation (it may not
match reality), no hardcoded geometry.

## Goals

- A single rod can **catch the puck and carry it, under control, to a
  commanded target position**, with translation and rotation coordinated so
  the puck stays on the blade.
- The control model is **learned from real `(t, r, puck)` data**, inspectable
  and debuggable.
- A stable foundation: predictable, interpretable, no black-box policy.

## Non-Goals

- No forceful directed *pass* / shot — a deliberate follow-on once controlled
  carry is solid.
- No multi-rod relay.
- No physics simulator, no reinforcement learning.
- No table/world coordinate system — everything stays in camera pixels and the
  module's `(t, r)`.

## Interfaces

- **`hockey-player` module** (one generic component per rod). `DoCommand`
  move `{"t","r","speed_mm_per_sec",...}` (blocking, returns
  `{"t_final","r_final"}`) and `{"cmd":"get_position"}` → `{"t","r",...}`.
  `t` ∈ [0,1], `r` ∈ [0,360], both precise and repeatable.
- **`vision-1`** via `detect_puck_px` — puck `(x, y)` in camera pixels, with
  retry through the flaky `dynamic-crop` camera. Reliable (~0.3 s/frame).

The system works entirely in `(t, r, puck_x_px, puck_y_px)` space. The model
never assumes table geometry; it learns whatever relationship the data shows.

Development rod: `left-defense` (puck control already proven there). All code
is rod-agnostic — the rod is a parameter.

## Architecture & files

| File | Responsibility |
| ---- | -------------- |
| `robot/rod_model.py` | The learned model: load the dataset, locally-weighted regression, predict the puck's response to a move and invert it for control. Pure — no hardware. |
| `robot/rod_session.py` | Hardware: autonomous data collection and the closed-loop controller, sharing one `RobotClient` connection. |
| `rod.py` | Top-level CLI. Subcommands `collect` and `control --target X Y`. |
| `data/rod_<name>.jsonl` | The dataset — one real sample per line. The training artifact, committed. |
| `robot/const.py` | Collection/control constants (see below). |
| `robot/vision.py` | Reused unchanged — `detect_puck_px`. |

## 1. Data collection — training on reality

The puck is placed on the rod once (by a human). The rod then drives **itself**
through many small moves, recording one sample per move:

```json
{"t": 0.30, "r": 0.0, "puck_x": 471.5, "puck_y": 250.0,
 "dt": 0.05, "dr": 20.0, "puck_x2": 466.0, "puck_y2": 248.5}
```

— the rod's `(t, r)` read from the module before the move, the puck before
(`puck_x/y`) and after (`puck_x2/y2`) from `vision-1`, and the move
`(dt, dr)`.

Collection procedure:

- Iterate over a coarse grid of base states `(t, r)`. At each base state, issue
  several **small** probe moves spanning `±Δt` and `±Δr` so the local response
  is sampled in every direction.
- Every move is gentle (low `speed_mm_per_sec`) so the puck is nudged, not
  launched.
- After each move, detect the puck. Record the sample regardless of whether the
  puck moved — no-motion samples bound the no-contact region.
- If the puck is not detected, or has left the rod's reachable span, pause and
  print a clear request for the human to reposition it; resume on input.
- Append every sample to `data/rod_<name>.jsonl` as it is taken, so a crash
  loses nothing.

This is the entire "training": the dataset is a direct record of how the rig
behaves.

## 2. The model (`robot/rod_model.py`)

The model answers one query: **at state `(t, r, puck_x, puck_y)`, what does a
move `(Δt, Δr)` do to the puck, and inversely, what move achieves a desired
`Δpuck`?**

- **Locally-weighted linear regression.** For a query state, take the `K`
  nearest dataset samples (distance in normalised `(t, r, puck_x, puck_y)`
  space), weight them by proximity, and least-squares fit a local linear map
  `Δpuck ≈ J · (Δt, Δr)`, where `J` is a 2×2 Jacobian. No global functional
  form, no assumed geometry — `J` is whatever the local data says.
- **Contact / controllability.** A query state is "in contact / controllable"
  when its neighbouring samples show puck motion well above the vision noise
  floor and the local fit residual is small. Otherwise the model reports
  "no reliable control here."
- **Inversion for control.** Given a desired `Δpuck`, return
  `(Δt, Δr) = J⁺ · Δpuck` (least-squares pseudo-inverse), clamped to the
  configured maximum step. The translation-rotation coordination is inherent
  in `J`: achieving a given puck motion generally requires a specific `Δt`
  *and* `Δr` together.

Public API (all pure functions / a small class):

- `load_dataset(path) -> list[Sample]`
- `RodModel(samples)` — holds the dataset.
- `RodModel.predict(t, r, puck, d_t, d_r) -> Δpuck`
- `RodModel.solve(t, r, puck, d_puck_desired) -> (Δt, Δr, controllable: bool)`

`numpy` (already a dependency) is used for the least-squares fits.

## 3. The controller (`robot/rod_session.py`)

Closed-loop carry of the puck to a target pixel position:

1. Detect the puck `P` (`vision-1`). The rod's `(t, r)` is tracked from the
   last move's `t_final`/`r_final` (seeded once from `get_position`).
2. If `P` is already within `target_tol_px` of the target → done.
3. Desired step `Δpuck_desired` = vector toward the target, clamped to
   `max_puck_step_px`.
4. `(Δt, Δr, controllable) = model.solve(t, r, P, Δpuck_desired)`.
5. If not `controllable` → run a small **re-acquire**: a few gentle probe moves
   to regain contact (and which themselves become new samples).
6. Command the gentle move `(t+Δt, r+Δr)`; read back `t_final`/`r_final`.
7. Detect the puck again. If the actual `Δpuck` disagrees with the model's
   prediction beyond `surprise_tol_px`, append the `(state, move, Δpuck)` as a
   new sample to the dataset — the model improves online.
8. Repeat from 1, up to `max_control_iters`.

On exit (target reached, iteration cap, or unrecoverable loss of the puck) the
rod is homed (`t=0, r=0`) and the outcome is reported.

## 4. CLI (`rod.py`)

```
python rod.py collect [--rod left-defense] [--samples N]
python rod.py control --target X Y [--rod left-defense]
```

- `collect` runs the data-collection routine and writes the dataset.
- `control` loads the dataset, fits the model, and runs the controller to
  carry the puck to `(X, Y)` in camera pixels.

## 5. Constants (`robot/const.py`)

| Constant | Meaning |
| -------- | ------- |
| `ROD_COLLECT_DT`, `ROD_COLLECT_DR` | Probe move magnitudes during collection. |
| `ROD_COLLECT_GRID_T`, `ROD_COLLECT_GRID_R` | Base-state grid resolution. |
| `ROD_MOVE_SPEED_MM_S` | Gentle move speed for collection and control. |
| `ROD_MODEL_K` | Neighbour count for locally-weighted regression. |
| `ROD_CONTACT_MOVE_PX` | Puck motion above which a sample counts as contact. |
| `ROD_MAX_PUCK_STEP_PX` | Max desired puck step per control iteration. |
| `ROD_MAX_STEP_T`, `ROD_MAX_STEP_R` | Clamp on a single commanded move. |
| `ROD_TARGET_TOL_PX` | Puck-to-target distance that counts as arrived. |
| `ROD_SURPRISE_TOL_PX` | Prediction error above which a sample is added online. |
| `ROD_MAX_CONTROL_ITERS` | Iteration cap for a carry. |

## Testing

- **Unit tests** (`tests/test_rod_model.py`) with synthetic datasets: the
  locally-weighted regression recovers a known linear `J`; `solve` inverts it;
  the contact/controllability flag is correct for in-contact vs no-motion
  sample clouds; step clamping holds.
- **Hardware**: collection and the control loop are validated incrementally on
  the rig — first that collection produces a sane dataset, then that the
  controller carries the puck to a target.

## Risks

- **Puck leaves reach during collection** — collection pauses and asks for a
  reposition; exploration moves stay small to limit drift.
- **Flaky `dynamic-crop` camera** — handled by `detect_puck_px`'s existing
  retry/timeout.
- **Noisy, non-repeatable contact dynamics** — locally-weighted regression
  averages neighbouring samples; collect enough data and keep moves small.
- **Low-friction momentum** — all moves (collection and control) are slow and
  small so the puck is nudged, not launched.
- **Sparse data regions** — if control reaches a state with too few neighbours,
  the model reports "no reliable control"; the controller re-acquires and the
  new samples fill the gap.
