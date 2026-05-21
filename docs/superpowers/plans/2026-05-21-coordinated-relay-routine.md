# Coordinated Five-Rod Relay Routine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a routine that relays the puck through all five rods (Left D → Right D → Center → Left Wing → Right Wing → shot), with each rod following the puck's real position and vision confirming the puck has arrived before the next rod acts.

**Architecture:** A new `robot/routine.py` orchestrator opens one shared Viam connection, then runs the relay leg-by-leg: detect the puck, position the receiving rod by normalizing the puck's y to a `[0,1]` translation, fire the pass, then poll `vision-1` until the puck reaches the next rod. Relay data lives in `robot/playbook.py`; a thin top-level `routine.py` provides the CLI.

**Tech Stack:** Python 3.13, viam-sdk, asyncio, pytest. Run commands via `uv`.

> **Commit convention:** Every commit message in this plan must end with the trailer:
> `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
> Work happens on the existing `coordinated-relay-routine` branch.

---

## File Structure

| File | Responsibility |
| ---- | -------------- |
| `engine/constants.py` | *Modify.* Add left-wing rod x and y-band constants. |
| `robot/const.py` | *Modify.* Add relay tuning constants. |
| `robot/playbook.py` | *Modify.* Add the `RELAY` leg data. |
| `robot/vision.py` | *Modify.* Extract `detect_puck(machine, ...)` for reuse on a shared connection. |
| `robot/routine.py` | *Create.* Relay orchestrator: coordinate math, vision gating, leg loop. |
| `routine.py` | *Create.* Thin CLI entry point. |
| `tests/test_routine.py` | *Create.* Unit tests for the pure functions and the vision-gate poll. |
| `pyproject.toml` | *Modify.* Add `pytest` dev dependency and pytest config. |

---

## Task 1: Project setup — pytest and left-wing constants

**Files:**
- Modify: `pyproject.toml`
- Modify: `engine/constants.py:100-104` (after the Left D block)

- [ ] **Step 1: Add pytest as a dev dependency**

Run: `uv add --dev pytest`
Expected: `pyproject.toml` gains a `[dependency-groups]` (or `[tool.uv]` dev) entry for `pytest`; `uv.lock` updates.

- [ ] **Step 2: Add pytest config to `pyproject.toml`**

Append this block to `pyproject.toml` so `import robot` / `import engine` resolve when running tests from the repo root:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Add left-wing rod constants to `engine/constants.py`**

In `engine/constants.py`, in the "Player home ranges" section, immediately after the Left D block (the line `left_d_x = 13 * SCALE   # 325 px`), add:

```python

# Left Wing  (no calibrated band existed — placeholders, TODO: calibrate)
min_y_left_wing = 21.5 * SCALE   # 537.5 px — TODO: calibrate
max_y_left_wing = 31   * SCALE   # 775 px   — TODO: calibrate
left_wing_x     = 11   * SCALE   # 275 px   — TODO: calibrate
```

- [ ] **Step 4: Verify the constants import**

Run: `uv run python -c "from engine.constants import left_wing_x, min_y_left_wing, max_y_left_wing; print(left_wing_x, min_y_left_wing, max_y_left_wing)"`
Expected: `275.0 537.5 775.0`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock engine/constants.py
git commit -m "Add pytest setup and left-wing rod constants"
```

---

## Task 2: `puck_y_to_t` — map puck y to a normalized translation

Creates `robot/routine.py` with its imports, the `_ROD_Y_BAND` lookup, and the first pure function.

**Files:**
- Create: `robot/routine.py`
- Create: `tests/test_routine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_routine.py` with:

```python
"""Unit tests for the relay routine's pure functions."""

from engine.constants import (
    min_y_center, max_y_center,
    min_y_left_d, max_y_left_d,
)
from engine.constants import PlayerID
from robot.routine import puck_y_to_t


def test_puck_y_to_t_endpoints():
    assert puck_y_to_t(PlayerID.CENTER, min_y_center) == 0.0
    assert puck_y_to_t(PlayerID.CENTER, max_y_center) == 1.0


def test_puck_y_to_t_midpoint():
    mid = (min_y_center + max_y_center) / 2
    assert puck_y_to_t(PlayerID.CENTER, mid) == 0.5


def test_puck_y_to_t_clamps_below_min():
    assert puck_y_to_t(PlayerID.CENTER, min_y_center - 100) == 0.0


def test_puck_y_to_t_clamps_above_max():
    assert puck_y_to_t(PlayerID.CENTER, max_y_center + 100) == 1.0


def test_puck_y_to_t_uses_per_rod_band():
    # Left D has a different band than Center
    assert puck_y_to_t(PlayerID.LEFT_D, min_y_left_d) == 0.0
    assert puck_y_to_t(PlayerID.LEFT_D, max_y_left_d) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_routine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robot.routine'`

- [ ] **Step 3: Create `robot/routine.py` with the minimal implementation**

Create `robot/routine.py` with:

```python
"""Coordinated five-rod relay routine.

Relays the puck through every rod in order:
  Left D -> Right D -> Center -> Left Wing -> Right Wing -> SHOT

Each receiving rod follows the puck's detected position; vision confirms the
puck has reached the next rod before that rod acts. See
docs/superpowers/specs/2026-05-21-coordinated-relay-routine-design.md.
"""

from engine.constants import (
    PlayerID,
    center_x, min_y_center, max_y_center,
    right_wing_x, min_y_right_wing, max_y_right_wing,
    right_d_x, min_y_right_d, max_y_right_d,
    left_d_x, min_y_left_d, max_y_left_d,
    left_wing_x, min_y_left_wing, max_y_left_wing,
)

# Per-rod translation band: t=0 -> min_y, t=1 -> max_y (game pixels).
_ROD_Y_BAND = {
    PlayerID.LEFT_D:     (min_y_left_d, max_y_left_d),
    PlayerID.RIGHT_D:    (min_y_right_d, max_y_right_d),
    PlayerID.CENTER:     (min_y_center, max_y_center),
    PlayerID.LEFT_WING:  (min_y_left_wing, max_y_left_wing),
    PlayerID.RIGHT_WING: (min_y_right_wing, max_y_right_wing),
}


def puck_y_to_t(player_id: PlayerID, puck_y: float) -> float:
    """Map a puck game-y coordinate to a normalized [0, 1] translation for a rod."""
    min_y, max_y = _ROD_Y_BAND[player_id]
    t = (puck_y - min_y) / (max_y - min_y)
    return max(0.0, min(1.0, t))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_routine.py -v`
Expected: PASS — 5 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/routine.py tests/test_routine.py
git commit -m "Add puck_y_to_t translation mapping for relay routine"
```

---

## Task 3: `puck_reached_rod` — the vision-gate predicate

**Files:**
- Modify: `robot/routine.py`
- Modify: `tests/test_routine.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_routine.py`:

```python
from robot.routine import puck_reached_rod


def test_puck_reached_rod_inside_tolerance():
    assert puck_reached_rod(puck_x=205.0, rod_x=200.0, tol=30.0) is True


def test_puck_reached_rod_outside_tolerance():
    assert puck_reached_rod(puck_x=260.0, rod_x=200.0, tol=30.0) is False


def test_puck_reached_rod_exactly_at_tolerance():
    assert puck_reached_rod(puck_x=230.0, rod_x=200.0, tol=30.0) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_routine.py -v`
Expected: FAIL — `ImportError: cannot import name 'puck_reached_rod'`

- [ ] **Step 3: Add the implementation**

In `robot/routine.py`, after `puck_y_to_t`, add the `_ROD_X` lookup and the function:

```python


# Per-rod x position (game pixels) — the gate axis for puck-arrival checks.
_ROD_X = {
    PlayerID.LEFT_D:     left_d_x,
    PlayerID.RIGHT_D:    right_d_x,
    PlayerID.CENTER:     center_x,
    PlayerID.LEFT_WING:  left_wing_x,
    PlayerID.RIGHT_WING: right_wing_x,
}


def puck_reached_rod(puck_x: float, rod_x: float, tol: float) -> bool:
    """True if the puck's game-x is within `tol` pixels of a rod's x position."""
    return abs(puck_x - rod_x) <= tol
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_routine.py -v`
Expected: PASS — 8 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/routine.py tests/test_routine.py
git commit -m "Add puck_reached_rod vision-gate predicate"
```

---

## Task 4: Relay tuning constants

**Files:**
- Modify: `robot/const.py` (append at end of file)

- [ ] **Step 1: Add the constants**

Append to `robot/const.py`:

```python


# ============================================================
#  Relay routine
# ============================================================

# Camera component used for puck detection during the relay.
RELAY_CAMERA = "dynamic-crop"

# How close (game pixels) the puck's x must be to a rod to count as "arrived".
RELAY_GATE_TOLERANCE_PX = 30.0   # TODO: calibrate

# Max time to wait for the puck to reach the next rod before aborting (seconds).
RELAY_GATE_TIMEOUT_S = 8.0

# Delay between vision polls while waiting at a gate (seconds).
RELAY_VISION_POLL_INTERVAL_S = 0.3
```

- [ ] **Step 2: Verify the constants import**

Run: `uv run python -c "from robot.const import RELAY_CAMERA, RELAY_GATE_TOLERANCE_PX, RELAY_GATE_TIMEOUT_S, RELAY_VISION_POLL_INTERVAL_S; print(RELAY_CAMERA, RELAY_GATE_TOLERANCE_PX, RELAY_GATE_TIMEOUT_S, RELAY_VISION_POLL_INTERVAL_S)"`
Expected: `dynamic-crop 30.0 8.0 0.3`

- [ ] **Step 3: Commit**

```bash
git add robot/const.py
git commit -m "Add relay routine tuning constants"
```

---

## Task 5: `RELAY` leg data

**Files:**
- Modify: `robot/playbook.py` (append after the Left Wing playbook section)

- [ ] **Step 1: Add the RELAY definition**

In `robot/playbook.py`, append after the `_LEFT_WING_PLAYBOOK` block (before the `# ── Public API` section):

```python


# ── Coordinated relay routine ──────────────────────────────────────────────────
#
# Ordered legs for the five-rod relay: Left D -> Right D -> Center ->
# Left Wing -> Right Wing -> SHOT. Each leg's `t` (translation) is computed at
# runtime from the puck position, so only `r` / `rpm` / `direction` are stored.
# The last leg's `pass_step` is the shot.

RELAY = [
    {
        "player": PlayerID.LEFT_D,
        "receive_r": 90,                                              # TODO: calibrate
        "pass_step": {"r": 300, "rpm": 220, "direction": "cw"},       # TODO: calibrate
    },
    {
        "player": PlayerID.RIGHT_D,
        "receive_r": 90,                                              # TODO: calibrate
        "pass_step": {"r": 60, "rpm": 220, "direction": "ccw"},       # TODO: calibrate
    },
    {
        "player": PlayerID.CENTER,
        "receive_r": 90,                                              # TODO: calibrate
        "pass_step": {"r": 60, "rpm": 300, "direction": "ccw"},       # TODO: calibrate
    },
    {
        "player": PlayerID.LEFT_WING,
        "receive_r": 90,                                              # TODO: calibrate
        "pass_step": {"r": 300, "rpm": 300, "direction": "cw"},       # TODO: calibrate
    },
    {
        "player": PlayerID.RIGHT_WING,
        "receive_r": 90,                                              # TODO: calibrate
        "pass_step": {"r": 0, "rpm": 400, "direction": "ccw"},        # TODO: calibrate -- SHOT
    },
]
```

- [ ] **Step 2: Verify the data imports**

Run: `uv run python -c "from robot.playbook import RELAY; print(len(RELAY), [leg['player'].name for leg in RELAY])"`
Expected: `5 ['LEFT_D', 'RIGHT_D', 'CENTER', 'LEFT_WING', 'RIGHT_WING']`

- [ ] **Step 3: Commit**

```bash
git add robot/playbook.py
git commit -m "Add RELAY leg data for the relay routine"
```

---

## Task 6: `format_relay_plan` — the `--dry-run` description

**Files:**
- Modify: `robot/routine.py`
- Modify: `tests/test_routine.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_routine.py`:

```python
from robot.routine import format_relay_plan


def test_format_relay_plan_lists_all_five_legs():
    text = format_relay_plan()
    assert "Leg 1:" in text
    assert "Leg 5:" in text
    assert "LEFT_D" in text
    assert "RIGHT_WING" in text


def test_format_relay_plan_marks_the_shot():
    text = format_relay_plan()
    assert "SHOT" in text


def test_format_relay_plan_describes_gates():
    text = format_relay_plan()
    # Four gates (one between each pair of legs), none after the last leg.
    assert text.count("gate:") == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_routine.py -v`
Expected: FAIL — `ImportError: cannot import name 'format_relay_plan'`

- [ ] **Step 3: Add the implementation**

In `robot/routine.py`, add this import near the top (with the other imports):

```python
from robot.const import (
    RELAY_CAMERA,
    RELAY_GATE_TOLERANCE_PX,
    RELAY_GATE_TIMEOUT_S,
    RELAY_VISION_POLL_INTERVAL_S,
)
from robot.playbook import RELAY
```

Then add the function after `puck_reached_rod`:

```python


def format_relay_plan() -> str:
    """Return a human-readable description of the relay plan (no hardware)."""
    lines = ["Relay plan (Left D -> Right D -> Center -> Left Wing -> Right Wing):"]
    last = len(RELAY) - 1
    for i, leg in enumerate(RELAY):
        player = leg["player"]
        action = "SHOT" if i == last else "pass"
        lines.append(
            f"  Leg {i + 1}: {player.name:11s} "
            f"receive r={leg['receive_r']} (t follows puck_y), "
            f"{action} {leg['pass_step']}"
        )
        if i != last:
            nxt = RELAY[i + 1]["player"]
            lines.append(
                f"          gate: wait until puck_x within "
                f"{RELAY_GATE_TOLERANCE_PX:.0f}px of {nxt.name} "
                f"(x={_ROD_X[nxt]:.0f})"
            )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_routine.py -v`
Expected: PASS — 11 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/routine.py tests/test_routine.py
git commit -m "Add format_relay_plan for relay --dry-run output"
```

---

## Task 7: Extract `detect_puck` from `robot/vision.py`

Lets the relay poll the camera repeatedly on one shared connection.

**Files:**
- Modify: `robot/vision.py:70-109`

- [ ] **Step 1: Replace `get_puck_game_coordinates` with a thin wrapper plus `detect_puck`**

In `robot/vision.py`, replace the entire `get_puck_game_coordinates` function (currently lines 70-109) with:

```python
async def detect_puck(machine, camera_name="C270"):
    """Detect the puck using an existing robot connection.

    Fetches puck detections from vision-1 and corner detections from vision-2
    to derive dynamic camera bounds, falling back to hardcoded bounds if corners
    are not found. Returns (game_x, game_y) in game pixels, or (None, None) if
    no puck is detected.
    """
    vision1 = VisionClient.from_robot(machine, "vision-1")
    vision2 = VisionClient.from_robot(machine, "vision-2")

    puck_detections, corner_detections = await asyncio.gather(
        vision1.get_detections_from_camera(camera_name),
        vision2.get_detections_from_camera(camera_name),
    )

    pink = [d for d in puck_detections if d.class_name == _PUCK_CLASS]
    if not pink:
        return None, None

    # Pick the median detection to reduce noise
    pink.sort(key=lambda d: d.y_min)
    puck = pink[len(pink) // 2]
    camera_x, camera_y = get_center(puck)
    print(f"Camera puck: x={camera_x:.1f}, y={camera_y:.1f}")

    bounds = _field_bounds_from_corners(corner_detections)
    if bounds:
        cam_x_min, cam_x_max, cam_y_min, cam_y_max = bounds
    else:
        cam_x_min, cam_x_max = CAMERA_X_MIN, CAMERA_X_MAX
        cam_y_min, cam_y_max = CAMERA_Y_MIN, CAMERA_Y_MAX

    return scale_puck_coords(camera_x, camera_y, cam_x_min, cam_x_max, cam_y_min, cam_y_max)


async def get_puck_game_coordinates():
    """Connect to the robot, detect the puck, and return its game-space (x, y).

    Returns (game_x, game_y) in pixels, or (None, None) if no puck is detected.
    """
    machine = await _connect()
    try:
        return await detect_puck(machine)
    finally:
        await machine.close()
```

- [ ] **Step 2: Verify imports still resolve and `main.py` is unaffected**

Run: `uv run python -c "from robot.vision import detect_puck, get_puck_game_coordinates; import main; print('ok')"`
Expected: `ok` (no ImportError; `main.py` imports `get_puck_game_coordinates` which still exists).

- [ ] **Step 3: Run the full test suite (regression check)**

Run: `uv run pytest -v`
Expected: PASS — 11 passed (no test imports `vision`, but confirm nothing broke).

- [ ] **Step 4: Commit**

```bash
git add robot/vision.py
git commit -m "Extract detect_puck for reuse on a shared connection"
```

---

## Task 8: Vision-gate poll — `wait_for_puck_at_rod` and `home_all`

**Files:**
- Modify: `robot/routine.py`
- Modify: `tests/test_routine.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_routine.py`:

```python
import asyncio

import robot.routine as routine_mod


def _fake_detect(return_value):
    """Build an async stand-in for detect_puck that always returns one value."""
    async def _detect(machine, camera_name=None):
        return return_value
    return _detect


def test_wait_for_puck_at_rod_times_out(monkeypatch):
    monkeypatch.setattr(routine_mod, "detect_puck", _fake_detect((None, None)))
    result = asyncio.run(
        routine_mod.wait_for_puck_at_rod(
            machine=None, rod_x=200.0, timeout=0.2, interval=0.05
        )
    )
    assert result is False


def test_wait_for_puck_at_rod_detects_arrival(monkeypatch):
    monkeypatch.setattr(routine_mod, "detect_puck", _fake_detect((205.0, 400.0)))
    result = asyncio.run(
        routine_mod.wait_for_puck_at_rod(
            machine=None, rod_x=200.0, tol=30.0, timeout=1.0, interval=0.05
        )
    )
    assert result is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_routine.py -v`
Expected: FAIL — `AttributeError: module 'robot.routine' has no attribute 'wait_for_puck_at_rod'`

- [ ] **Step 3: Add the implementation**

In `robot/routine.py`, add these imports near the top (with the other imports):

```python
import asyncio

from viam.components.generic import Generic

from robot.execution import _PLAYER_TO_COMPONENT
from robot.vision import detect_puck
```

Then add both functions at the end of `robot/routine.py`:

```python


async def wait_for_puck_at_rod(machine, rod_x,
                               tol=RELAY_GATE_TOLERANCE_PX,
                               timeout=RELAY_GATE_TIMEOUT_S,
                               interval=RELAY_VISION_POLL_INTERVAL_S) -> bool:
    """Poll vision until the puck's x is within `tol` of `rod_x`.

    Returns True once the puck arrives, or False if `timeout` seconds elapse
    first. `machine` is an open RobotClient connection.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        puck_x, _ = await detect_puck(machine, RELAY_CAMERA)
        if puck_x is not None and puck_reached_rod(puck_x, rod_x, tol):
            return True
        await asyncio.sleep(interval)
    return False


async def home_all(components: dict) -> None:
    """Return every rod in `components` to home pose (t=0, r=0) concurrently."""
    print("Homing all rods.")
    await asyncio.gather(*[c.do_command({"t": 0, "r": 0}) for c in components.values()])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_routine.py -v`
Expected: PASS — 13 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/routine.py tests/test_routine.py
git commit -m "Add vision-gate poll and home_all for relay routine"
```

---

## Task 9: `run_relay` orchestrator and the `routine.py` CLI

`run_relay` is integration glue over already-tested functions and a live Viam connection, so it has no unit test; it is verified via `--dry-run` and a manual hardware run.

**Files:**
- Modify: `robot/routine.py`
- Create: `routine.py`

- [ ] **Step 1: Add `run_relay` to `robot/routine.py`**

Append to `robot/routine.py`:

```python


async def run_relay(machine) -> None:
    """Run the full five-rod relay on an open RobotClient connection.

    Detects the puck, then walks each leg: position the receiving rod to the
    puck's y, fire the pass, and (except on the last leg) wait for vision to
    confirm the puck reached the next rod. All rods are homed on exit, whether
    the relay finishes or aborts.
    """
    components = {
        pid: Generic.from_robot(robot=machine, name=_PLAYER_TO_COMPONENT[pid])
        for pid in _ROD_X
    }
    try:
        puck_x, puck_y = await detect_puck(machine, RELAY_CAMERA)
        if puck_x is None:
            print("No puck detected — aborting relay.")
            return

        first = RELAY[0]["player"]
        if not puck_reached_rod(puck_x, _ROD_X[first], RELAY_GATE_TOLERANCE_PX):
            print(f"Puck not on {first.name}'s rod (puck_x={puck_x:.0f}, "
                  f"expected ~{_ROD_X[first]:.0f}) — place the puck there to start.")
            return

        last = len(RELAY) - 1
        for i, leg in enumerate(RELAY):
            player = leg["player"]
            comp = components[player]

            # Legs after the first: re-detect the puck the gate just confirmed.
            if i > 0:
                puck_x, puck_y = await detect_puck(machine, RELAY_CAMERA)
                if puck_x is None:
                    print(f"Lost the puck before {player.name}'s leg — aborting.")
                    return

            t = puck_y_to_t(player, puck_y)
            print(f"Leg {i + 1}: {player.name} — receive t={t:.2f}, r={leg['receive_r']}")
            await comp.do_command({"t": t, "r": leg["receive_r"]})
            await comp.do_command(leg["pass_step"])

            if i != last:
                nxt = RELAY[i + 1]["player"]
                print(f"  waiting for puck to reach {nxt.name}...")
                arrived = await wait_for_puck_at_rod(machine, _ROD_X[nxt])
                if not arrived:
                    print(f"Puck never reached {nxt.name} — aborting relay.")
                    return

        print("Relay complete.")
    finally:
        await home_all(components)
```

- [ ] **Step 2: Create the `routine.py` CLI**

Create `routine.py` (top level) with:

```python
"""Run the coordinated five-rod relay routine.

Usage:
  python3 routine.py            # run the relay on the robot
  python3 routine.py --dry-run  # print the relay plan, no hardware
"""

import argparse
import asyncio

from viam.robot.client import RobotClient

from robot.const import ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID
from robot.routine import format_relay_plan, run_relay


async def _main():
    parser = argparse.ArgumentParser(description="Five-rod relay routine")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the relay plan without connecting to hardware")
    args = parser.parse_args()

    if args.dry_run:
        print(format_relay_plan())
        return

    opts = RobotClient.Options.with_api_key(
        api_key=ROBOT_API_KEY, api_key_id=ROBOT_API_KEY_ID,
    )
    machine = await RobotClient.at_address(ROBOT_ADDRESS, opts)
    try:
        await run_relay(machine)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(_main())
```

- [ ] **Step 3: Verify `--dry-run` works end-to-end**

Run: `uv run python routine.py --dry-run`
Expected: prints the relay plan — a "Relay plan" header, five `Leg N:` lines (Leg 1 LEFT_D … Leg 5 RIGHT_WING marked `SHOT`), and four `gate:` lines. No network connection attempted.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS — 13 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/routine.py routine.py
git commit -m "Add run_relay orchestrator and routine.py CLI"
```

---

## Task 10: Documentation

**Files:**
- Modify: `README.md` (in the "Usage" section)

- [ ] **Step 1: Document the routine in `README.md`**

In `README.md`, after the "Drive all players to a pose (smoke test)" subsection, add:

```markdown
### Run the coordinated relay routine

Relays the puck through all five rods and finishes with a shot. Each rod
follows the puck's detected position; vision confirms the puck arrived before
the next rod acts.

```bash
python routine.py            # run the relay on the robot
python routine.py --dry-run  # print the relay plan, no hardware
```

Place the puck on the Left D rod before starting. Tuning lives in
`robot/const.py` (`RELAY_*`) and `robot/playbook.py` (`RELAY`).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document the relay routine in README"
```

---

## Manual Hardware Verification (after all tasks)

These are not automated — run them on the physical robot once the plan is complete:

1. `uv run python routine.py --dry-run` — confirm the plan reads correctly.
2. Place the puck on the Left D rod. Run `uv run python routine.py`.
3. Watch for: each rod positioning to the puck, passes landing near the next rod,
   gates resolving without timing out, and all rods homing at the end.
4. Calibrate the `# TODO: calibrate` values in `robot/playbook.py` (`RELAY`
   pass angles/rpm), `robot/const.py` (`RELAY_GATE_TOLERANCE_PX`), and
   `engine/constants.py` (left-wing band) against observed behavior.

---

## Self-Review Notes

- **Spec coverage:** architecture/files (Tasks 1-10), relay data model (Task 5),
  orchestration flow (Tasks 8-9), pixels-not-mm coordinate model (Task 2's
  `puck_y_to_t` + Task 7's `detect_puck`), geometric x/y assumption (encoded in
  `_ROD_X`/`_ROD_Y_BAND`, Tasks 2-3), error handling & cleanup (Task 9's
  abort paths + `home_all`), CLI & `--dry-run` (Tasks 6, 9), tuning constants
  (Task 4), testing (unit tests Tasks 2/3/6/8 + manual section). All spec
  sections map to a task.
- **Spec correction applied:** the spec assumed all rods had `min_y_*/max_y_*`
  bands; left wing did not, so Task 1 adds `min_y_left_wing`, `max_y_left_wing`,
  and `left_wing_x`.
- **Type consistency:** `puck_y_to_t`, `puck_reached_rod`, `format_relay_plan`,
  `wait_for_puck_at_rod`, `home_all`, `run_relay`, `detect_puck`, and the
  `RELAY` leg keys (`player`, `receive_r`, `pass_step`) are used consistently
  across all tasks.
