# Autonomous One-Player Puck Control — Design

**Date:** 2026-05-21
**Status:** Proposed
**Supersedes:** all prior relay / one-rod specs in this directory. This is a
square-zero rebuild.

## Why a rebuild

Hours of hardware work established what is real and what is not. Every prior
attempt failed for one of two reasons: it **hardcoded** the action→puck
relationship (the camera geometry turned out mirrored and unreliable), or it
**collected data with a human in the loop** (a person had to keep repositioning
the puck). This design removes both: the action→puck model is *learned from
reality*, and the data is gathered by a *self-sustaining autonomous loop* that
never needs a human.

## Hardware facts this design is built on

- `vision-1` gives the puck's position in **camera pixels** — fast (~0.3 s) and
  reliable, with retry through the flaky `dynamic-crop` camera.
- The `hockey-player` module gives precise `(t, r)` control and `get_position`;
  `t ∈ [0, 1]`, `r ∈ [0, 360]`.
- Contact is **one-directional**: the player moves the puck only by translating
  *into* it (a push). A continuous push reliably **carries** the puck along the
  rod — this is the one robust primitive everything is built on.
- Camera→table geometry is mirrored and unreliable. The system works **entirely
  in camera pixels** and module `(t, r)` — never table geometry.
- Hardware cycles are slow (~3 s per move+detect), so the approach must be
  sample-efficient: model-based control, not reinforcement learning.

## Goal (v1)

One player, fully autonomous: wherever the puck is within the player's reach,
the robot detects it, drives the player to it, moves it toward target
positions, and recovers it on its own if contact slips. **Zero human
involvement. No hardcoded geometry.**

## Non-Goals (v1)

- Multi-player handoffs around the rink — v2.
- Forceful passing / shooting (the fast rotation strike) — v2.
- Any table/world coordinate system, physics simulator, or reinforcement
  learning.

## Architecture & files

A clean rebuild. The prior one-rod / relay modules (`robot/rod_model.py`,
`robot/rod_session.py`, `robot/routine.py`, `rod.py`, `routine.py`) carried the
broken assumptions and are **removed**.

| File | Responsibility |
| ---- | -------------- |
| `robot/vision.py` | Kept — `detect_puck_px`: puck position in camera pixels, retry through the flaky camera. |
| `robot/puck_model.py` | The learned model: dataset I/O, locally-weighted regression of `(player state, move) → puck response`, a confidence score, and inversion for control. Pure — no hardware. |
| `robot/autonomy.py` | The autonomous loop: perceive, decide explore/exploit, act, record, re-acquire. |
| `robot/const.py` | Tuning constants (`AUTO_*`). |
| `auto.py` | Top-level CLI — start the autonomous loop on a player. |
| `data/<player>.jsonl` | The dataset — one real sample per line, persists across runs so learning accumulates. |

Developed and run on `left-defense`; all code is player-agnostic.

## 1. The learned model (`robot/puck_model.py`)

The model answers: **at state `(player_t, player_r, puck_x, puck_y)`, what does
a move `(Δt, Δr)` do to the puck — and how confident is that answer?**

- **Sample** — one recorded interaction:
  `(t, r, puck_x, puck_y, dt, dr, puck_x2, puck_y2)`.
- **Locally-weighted regression** — for a query state, take the `K` nearest
  samples (distance in normalised state space), weight by proximity, and
  least-squares fit a local 2×2 Jacobian `J` with `Δpuck ≈ J·(Δt, Δr)`. No
  assumed geometry — `J` is whatever the data says.
- **Confidence** — derived from how much data sits near the query (nearest-
  neighbour distances) and the local fit residual. Low confidence = the model
  does not know this region. This is what lets the loop know what it doesn't
  know.
- **`predict(state, move) → Δpuck`** and
  **`solve(state, desired Δpuck) → (Δt, Δr, confidence)`** (least-squares
  inversion of `J`, clamped to a max step).
- **Dataset I/O** — `load_dataset`, `append_sample`; the dataset is appended to
  live by the loop.

`numpy` (already a dependency) does the fits. The model is pure and
unit-testable with synthetic datasets.

## 2. The autonomous loop (`robot/autonomy.py`)

One loop, unattended, runs indefinitely. It owns a single shared `RobotClient`
connection. Each cycle (~3 s):

1. **Perceive** — puck pixel position via `detect_puck_px`; player `(t, r)`
   tracked from the module's reported `t_final`/`r_final` (seeded from
   `get_position`).
2. **Goal** — the loop maintains a current target puck position and generates
   its own: when the puck reaches the current target (within
   `AUTO_TARGET_TOL_PX`), it picks the next one from a self-generated sequence
   of reachable pixel points. No human supplies goals or pucks.
3. **Decide** — explore or exploit (§3).
4. **Act** — command one gentle move (`AUTO_MOVE_SPEED_MM_S`), clamped to
   `[0,1]` in `t` and wrapped in `r`.
5. **Observe & record** — detect the puck again; append the
   `(state, move, puck before, puck after)` sample to `data/<player>.jsonl`;
   the model picks it up on its next query (the dataset *is* the model's
   memory).
6. **Re-acquire** — if the move did not move the puck (`< AUTO_CONTACT_MOVE_PX`,
   contact lost), the loop drives the player toward the puck: it sweeps `t`
   gently until the puck moves again. Each sweep step is also a recorded
   sample. This is the "recover if it slips" behaviour.

The loop never needs a human because the robot always knows where the puck is
and can always push it back into play. It stops only on an unrecoverable
condition (puck not found anywhere after a full search → it reports and keeps
searching) or operator interrupt. On exit it homes the player.

## 3. Explore vs exploit

Every cycle, the loop asks the model for `solve(state, desired)` where
`desired` is a clamped step toward the current goal. The returned confidence
decides:

- **Confident** (`≥ AUTO_CONFIDENCE_THRESHOLD`) → **exploit**: command the
  solved move — the model's best estimate of how to advance the puck toward the
  goal.
- **Uncertain** → **explore**: command an informative move into an
  under-sampled part of the action space, biased toward a *push* (the robust
  primitive) with random variation in `(Δt, Δr)`.

Both branches just move the player; they differ only in precision. Early in a
run the loop is mostly uncertain → it explores and the dataset grows fast. As
data accumulates the model becomes confident → the loop exploits and hits
targets. A failed exploit (puck moved unexpectedly) is simply a new sample in a
region the model was wrong about — so control failures *are* the exploration
that fixes them. This is the self-correcting mechanism the earlier attempts
lacked.

## 4. Data flow

```
vision-1 ─┐
          ├─► autonomy loop ─► model.solve ─► move ─► observe
module  ──┘        ▲                                    │
                   └──────── append sample ◄────────────┘
                              (data/<player>.jsonl)
```

The dataset is the single source of truth — the model is re-fit from it (cheap,
`numpy` least-squares over `K` neighbours) and it persists across runs.

## 5. CLI (`auto.py`)

```
python auto.py [--player left-defense-hockey-player]
```

Starts the autonomous loop on the given player and runs until interrupted.
There are no other modes — the system is the loop.

## 6. Constants (`robot/const.py`)

| Constant | Meaning |
| -------- | ------- |
| `AUTO_MOVE_SPEED_MM_S` | Gentle move speed — carry the puck, never launch it. |
| `AUTO_MAX_STEP_T`, `AUTO_MAX_STEP_R` | Clamp on one commanded move. |
| `AUTO_CONTACT_MOVE_PX` | Puck motion above which a move counts as "made contact". |
| `AUTO_TARGET_TOL_PX` | Puck-to-target distance counting as "arrived". |
| `AUTO_MODEL_K` | Neighbour count for locally-weighted regression. |
| `AUTO_CONFIDENCE_THRESHOLD` | Confidence above which the loop exploits rather than explores. |
| `AUTO_EXPLORE_DT`, `AUTO_EXPLORE_DR` | Magnitude of exploratory moves. |
| `AUTO_VISION_RETRIES`, `AUTO_VISION_CALL_TIMEOUT_S` | Robust-detection retry budget. |

## Testing

- **Unit tests** (`tests/test_puck_model.py`): the model on synthetic datasets
  — locally-weighted regression recovers a known Jacobian, `solve` inverts it,
  the confidence score is high in dense regions and low in sparse ones, step
  clamping holds. The explore/exploit decision (a pure function of confidence)
  is unit-tested.
- **Hardware**: the loop is validated by *running it* — start `auto.py`, watch
  it autonomously acquire, move, and recover the puck. Because it is
  autonomous, validation needs no human babysitting.

## Risks

- **Confidence calibration** — if the threshold is too low the loop explores
  forever; too high and it exploits a bad model. Starts conservative, tuned by
  watching a run.
- **Flaky camera** — handled by `detect_puck_px`'s retry/timeout; a persistent
  failure stalls the loop, which is correct (it reports and waits).
- **Puck knocked off the table / out of all reach** — the loop searches; if the
  puck is genuinely gone it reports and keeps searching rather than crashing.
- **Slow hardware** — accepted: the loop runs unattended, so wall-clock time is
  not a babysitting cost, and model-based control needs hundreds of samples,
  not the millions reinforcement learning would.
- **One-directional contact** — the model learns it; the controller plans moves
  the model says actually work. The loop is not told the geometry.
