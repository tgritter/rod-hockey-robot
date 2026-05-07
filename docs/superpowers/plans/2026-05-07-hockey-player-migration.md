# Hockey-player Module Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `robot/playbook.py` and `robot/execution.py` from raw `Motor.go_for(rpm, revolutions)` calls with relative tick-tuple sequences to the `nfranczak:generic:hockey-player` Viam Generic component with absolute `(t, r)` dict-shaped sequences.

**Architecture:** `playbook.py` keeps every public name and the side-selection logic in `get_instructions`; only the contents of each sequence change shape from `(motor, ticks, rpm)` tuples to `DoCommand`-shaped dicts (placeholders for on-hardware calibration). `execution.py` opens a single `RobotClient` connection to the primary part, looks up one `Generic` component per player via a local `PlayerID → component-name` map, and forwards each step dict directly to `do_command`. No reset-to-home.

**Tech Stack:** Python 3.13, `viam-sdk` (already a dependency), Viam `Generic` component.

**Repo facts to know:**
- No first-party test suite; verification is via `python -c` import/shape checks plus hardware smoke tests through `main.py`.
- `main.py` imports these names from `robot.playbook` (must remain importable with same shapes):
  `get_instructions`, `get_rw_sequence`, `_CENTER_PLAYBOOK`, `_RIGHT_D_PLAYBOOK`, `_LEFT_D_PLAYBOOK`, `_LEFT_WING_PLAYBOOK`.
- `engine.constants.PlayerID` is an `IntEnum` with members `CENTER`, `RIGHT_WING`, `LEFT_WING`, `RIGHT_D`, `LEFT_D` (values 0–4) and a stale `get_prefix()` we will not call from the new code.
- The three Viam parts are joined as remotes of the primary so bare component names like `center-hockey-player` resolve from a single connection.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `robot/playbook.py` | Modify (full rewrite of sequences + docstring; logic preserved) | Define per-player sequence dicts and side-selection in `get_instructions` / `get_rw_sequence`. |
| `robot/execution.py` | Modify (full rewrite) | Connect once to the primary part, look up the right `Generic` component for the player, dispatch each step dict via `do_command`. |

`engine/constants.py`, `main.py`, `robot/const.py`, `robot/vision.py` are out of scope and must not be edited. `robot/const.py:TICKS_PER_ROTATION` becomes unused (left in place).

---

## Task 1: Rewrite `robot/playbook.py`

**Files:**
- Modify: `robot/playbook.py` (full file replacement)

This task replaces every step tuple `(motor, ticks, rpm)` with a placeholder dict `{"t": 0.0, "r": 0.0}` and updates the module docstring to reflect the new format. Every named sequence constant (`CENTER_LEFT`, `RIGHT_WING_LEFT`, `LEFT_D_LEFT`, etc.) and every dict (`_CENTER_PLAYBOOK`, `_RIGHT_D_PLAYBOOK`, `_LEFT_D_PLAYBOOK`, `_LEFT_WING_PLAYBOOK`, `_RIGHT_WING_POSITIONS`, `_RIGHT_WING_ACTIONS`) is preserved by name. `get_rw_sequence` and `get_instructions` are preserved with the same signatures and side-selection logic.

- [ ] **Step 1: Read the current file to confirm the names and selection logic before rewriting**

Run:
```bash
sed -n '1,50p' robot/playbook.py
```
Expected: docstring referencing `(motor, revs, rpm)` format, plus `from engine.constants import PlayerID, center_x, left_d_x, LEFT_WING_SEG_B_X_MID, right_d_x, right_wing_x`.

- [ ] **Step 2: Replace `robot/playbook.py` in full with the content below**

```python
"""Calibrated instruction playbooks for each player.

Each step is a dict matching the hockey-player module's DoCommand payload:
  {"t": 0.5, "r": 90, "rpm": 30, "speed_mm_per_sec": 100}

Fields (all optional; omit to skip an axis or use config defaults):
  t                 : translation target, normalized over [min_translation_mm,
                      max_translation_mm]. Range [0, 1].
  r                 : rotation target in degrees. Range [0, 360].
  rpm               : rotation speed (defaults from component config).
  speed_mm_per_sec  : translation speed (defaults from component config).

All values below are placeholders -- calibrate on hardware.
"""

from engine.constants import (
    PlayerID,
    center_x,
    left_d_x,
    LEFT_WING_SEG_B_X_MID,
    right_d_x,
    right_wing_x,
)


# ── Center player playbook ─────────────────────────────────────────────────────
#
# X-axis: right (puck_x < center_x, closer to 0) vs left (puck_x >= center_x).

CENTER_LEFT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- center, puck on left
]

CENTER_RIGHT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- center, puck on right
]

_CENTER_PLAYBOOK = {
    "left":  CENTER_LEFT,
    "right": CENTER_RIGHT,
}


# ── Right wing playbook ────────────────────────────────────────────────────────
#
# Two-phase: position sequence + action sequence, concatenated at runtime.
# e.g. RIGHT_WING_LEFT + RIGHT_WING_SHOT.

# Position sequences -- move puck to sweet spot
RIGHT_WING_LEFT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- right wing position, puck on left
]

RIGHT_WING_RIGHT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- right wing position, puck on right
]

RIGHT_WING_BOTTOM_LEFT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- right wing position, puck bottom-left
]

RIGHT_WING_BOTTOM_RIGHT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- right wing position, puck bottom-right
]

# Action sequences -- execute the play
RIGHT_WING_SHOT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- right wing shot
]

RIGHT_WING_PASS = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- right wing pass
]

RIGHT_WING_BOTTOM_SHOT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- right wing bottom shot
]

RIGHT_WING_BOTTOM_PASS = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- right wing bottom pass
]

_RIGHT_WING_POSITIONS = {
    "left":         RIGHT_WING_LEFT,
    "right":        RIGHT_WING_RIGHT,
    "bottom_left":  RIGHT_WING_BOTTOM_LEFT,
    "bottom_right": RIGHT_WING_BOTTOM_RIGHT,
}

_RIGHT_WING_ACTIONS = {
    "shot":        RIGHT_WING_SHOT,
    "pass":        RIGHT_WING_PASS,
    "bottom_shot": RIGHT_WING_BOTTOM_SHOT,
    "bottom_pass": RIGHT_WING_BOTTOM_PASS,
}


def get_rw_sequence(side: str, action: str) -> list:
    """Combine a position and action sequence for the right wing."""
    return _RIGHT_WING_POSITIONS[side] + _RIGHT_WING_ACTIONS[action]


# ── Right defenseman playbook ──────────────────────────────────────────────────

RIGHT_D_LEFT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- right D, puck on left
]

RIGHT_D_RIGHT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- right D, puck on right
]

_RIGHT_D_PLAYBOOK = {
    "left":  RIGHT_D_LEFT,
    "right": RIGHT_D_RIGHT,
}


# ── Left defenseman playbook ───────────────────────────────────────────────────

LEFT_D_LEFT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- left D, puck on left (pass to center)
]

LEFT_D_RIGHT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- left D, puck on right (pass to center)
]

_LEFT_D_PLAYBOOK = {
    "left":  LEFT_D_LEFT,
    "right": LEFT_D_RIGHT,
}


# ── Left wing playbook ─────────────────────────────────────────────────────────
#
# Zones are 2D (x + y), but playbook uses simple left/right for now.

LEFT_WING_LEFT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- left wing, puck on left
]

LEFT_WING_RIGHT = [
    {"t": 0.0, "r": 0.0},  # TODO: calibrate -- left wing, puck on right
]

_LEFT_WING_PLAYBOOK = {
    "left":  LEFT_WING_LEFT,
    "right": LEFT_WING_RIGHT,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def _center_side(puck_x: float) -> str:
    return "right" if puck_x < center_x else "left"  # x closer to 0 = right


def get_instructions(puck_x: float, puck_y: float, player_id: PlayerID = PlayerID.CENTER):  # noqa: ARG001
    """Return the calibrated instruction sequence for the given player and puck position."""
    if player_id == PlayerID.CENTER:
        side = _center_side(puck_x)
        print(f"Center side: {side}  (puck_x={puck_x:.0f})")
        return _CENTER_PLAYBOOK[side]

    if player_id == PlayerID.RIGHT_WING:
        side = "left" if puck_x < right_wing_x else "right"
        action = "shot"  # default; caller can override via get_rw_sequence directly
        print(f"Right wing side: {side}  (puck_x={puck_x:.0f})")
        return get_rw_sequence(side, action)

    if player_id == PlayerID.RIGHT_D:
        side = "right" if puck_x < right_d_x else "left"
        print(f"Right D side: {side}  (puck_x={puck_x:.0f})")
        return _RIGHT_D_PLAYBOOK[side]

    if player_id == PlayerID.LEFT_D:
        side = "right" if puck_x < left_d_x else "left"
        print(f"Left D side: {side}  (puck_x={puck_x:.0f})")
        return _LEFT_D_PLAYBOOK[side]

    if player_id == PlayerID.LEFT_WING:
        side = "right" if puck_x < LEFT_WING_SEG_B_X_MID else "left"
        print(f"Left Wing side: {side}  (puck_x={puck_x:.0f})")
        return _LEFT_WING_PLAYBOOK[side]

    return None
```

- [ ] **Step 3: Verify the file imports cleanly**

Run:
```bash
cd /home/nick/rod-hockey-robot && uv run python -c "import robot.playbook; print('ok')"
```
Expected output: `ok`.

If `ModuleNotFoundError` or `SyntaxError` — re-read the file, fix, retry.

- [ ] **Step 4: Verify every name `main.py` imports is present and has the expected shape**

Run:
```bash
cd /home/nick/rod-hockey-robot && uv run python -c "
from robot.playbook import (
    get_instructions, get_rw_sequence,
    _CENTER_PLAYBOOK, _RIGHT_D_PLAYBOOK, _LEFT_D_PLAYBOOK, _LEFT_WING_PLAYBOOK,
)
for name, pb in [('CENTER', _CENTER_PLAYBOOK),
                 ('RIGHT_D', _RIGHT_D_PLAYBOOK),
                 ('LEFT_D',  _LEFT_D_PLAYBOOK),
                 ('LEFT_WING', _LEFT_WING_PLAYBOOK)]:
    assert set(pb.keys()) == {'left', 'right'}, (name, pb.keys())
    for side, seq in pb.items():
        assert isinstance(seq, list) and len(seq) >= 1, (name, side)
        assert isinstance(seq[0], dict), (name, side, type(seq[0]))
print('shapes ok')

rw = get_rw_sequence('bottom_left', 'shot')
assert isinstance(rw, list) and isinstance(rw[0], dict), rw
print('rw ok')
"
```
Expected output:
```
shapes ok
rw ok
```

- [ ] **Step 5: Verify `get_instructions` selection still routes correctly**

Run:
```bash
cd /home/nick/rod-hockey-robot && uv run python -c "
from robot.playbook import get_instructions, _CENTER_PLAYBOOK, _LEFT_D_PLAYBOOK
from engine.constants import PlayerID, center_x, left_d_x

# center: puck_x < center_x -> right; >= center_x -> left
assert get_instructions(center_x - 1, 0, PlayerID.CENTER) is _CENTER_PLAYBOOK['right']
assert get_instructions(center_x + 1, 0, PlayerID.CENTER) is _CENTER_PLAYBOOK['left']

# left D: puck_x < left_d_x -> right; >= left_d_x -> left
assert get_instructions(left_d_x - 1, 0, PlayerID.LEFT_D) is _LEFT_D_PLAYBOOK['right']
assert get_instructions(left_d_x + 1, 0, PlayerID.LEFT_D) is _LEFT_D_PLAYBOOK['left']

print('selection ok')
"
```
Expected output (after the per-player `print` lines):
```
selection ok
```

- [ ] **Step 6: Commit**

```bash
git add robot/playbook.py
git commit -m "$(cat <<'EOF'
Convert playbook sequences to DoCommand dict format

Replace (motor, ticks, rpm) tuples with placeholder dicts shaped for the
nfranczak:generic:hockey-player module's DoCommand payload. All public
names and selection logic preserved; values are TODO placeholders to be
calibrated on hardware.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Rewrite `robot/execution.py`

**Files:**
- Modify: `robot/execution.py` (full file replacement)

Switch from two `Motor` components per player to one `Generic` hockey-player component per player. Each step dict is forwarded to `do_command` directly. No reset-to-home. The `PlayerID → component-name` map lives locally because `engine/constants.py` is out of scope.

- [ ] **Step 1: Read the current file to confirm imports and `execute_sequence` signature**

Run:
```bash
sed -n '1,50p' robot/execution.py
```
Expected: imports `RobotClient`, `Motor`, `TICKS_PER_ROTATION`, defines `execute_sequence(sequence, player_id=PlayerID.CENTER)`.

- [ ] **Step 2: Replace `robot/execution.py` in full with the content below**

```python
from viam.robot.client import RobotClient
from viam.components.generic import Generic

from .const import ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID
from engine.constants import PlayerID


_PLAYER_TO_COMPONENT = {
    PlayerID.CENTER:     "center-hockey-player",
    PlayerID.RIGHT_WING: "right-wing-hockey-player",
    PlayerID.LEFT_WING:  "left-wing-hockey-player",
    PlayerID.RIGHT_D:    "right-defense-hockey-player",
    PlayerID.LEFT_D:     "left-defense-hockey-player",
}


async def execute_sequence(sequence, player_id=PlayerID.CENTER):
    """Send each step in `sequence` to the player's hockey-player component via DoCommand.

    Each step is a dict matching the DoCommand payload (t, r, rpm,
    speed_mm_per_sec -- all optional). No automatic reset-to-home.
    """
    if not sequence:
        print("Empty sequence.")
        return

    component_name = _PLAYER_TO_COMPONENT[player_id]
    print(f"Executing sequence ({len(sequence)} steps, player={player_id.name}, component={component_name})")

    opts = RobotClient.Options.with_api_key(
        api_key=ROBOT_API_KEY, api_key_id=ROBOT_API_KEY_ID,
    )
    robot = await RobotClient.at_address(ROBOT_ADDRESS, opts)
    try:
        player = Generic.from_robot(robot=robot, name=component_name)
        for step in sequence:
            await player.do_command(step)
    finally:
        await robot.close()

    print("Done.")
```

- [ ] **Step 3: Verify the file imports cleanly**

Run:
```bash
cd /home/nick/rod-hockey-robot && uv run python -c "import robot.execution; print('ok')"
```
Expected output: `ok`.

If `ImportError: cannot import name 'Generic'` — confirm `viam-sdk>=0.71.0` is installed (`uv pip show viam-sdk`); the `Generic` component class lives at `viam.components.generic.Generic`. If a syntax/name error appears, fix and retry.

- [ ] **Step 4: Verify the `PlayerID → component-name` map covers every member**

Run:
```bash
cd /home/nick/rod-hockey-robot && uv run python -c "
from robot.execution import _PLAYER_TO_COMPONENT
from engine.constants import PlayerID
expected = {
    PlayerID.CENTER:     'center-hockey-player',
    PlayerID.RIGHT_WING: 'right-wing-hockey-player',
    PlayerID.LEFT_WING:  'left-wing-hockey-player',
    PlayerID.RIGHT_D:    'right-defense-hockey-player',
    PlayerID.LEFT_D:     'left-defense-hockey-player',
}
assert _PLAYER_TO_COMPONENT == expected, _PLAYER_TO_COMPONENT
assert set(_PLAYER_TO_COMPONENT.keys()) == set(PlayerID), 'missing player'
print('map ok')
"
```
Expected output: `map ok`.

- [ ] **Step 5: Verify `main.py` still imports and parses (it imports `execute_sequence`)**

Run:
```bash
cd /home/nick/rod-hockey-robot && uv run python -c "import main; print('main ok')"
```
Expected output: `main ok`.

If this fails, the rewrite changed something `main.py` depends on. Re-read `main.py` lines 36–38 to confirm what it imports from `robot.execution` and `robot.playbook`, and adjust.

- [ ] **Step 6: Commit**

```bash
git add robot/execution.py
git commit -m "$(cat <<'EOF'
Drive players via hockey-player Generic component

Replace per-axis Motor.go_for calls with a single Generic.do_command per
step, dispatched to the corresponding nfranczak:generic:hockey-player
component for each PlayerID. Drop ticks-to-revs conversion and the
reset-to-home pass. Local PlayerID -> component-name map keeps engine/
out of scope.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Hardware smoke tests

**Files:** none (manual verification only)

Run each manual override against the hardware to confirm the new code path successfully reaches each `*-hockey-player` component. Because every step is `{"t": 0.0, "r": 0.0}`, the rod should drive to (or stay at) home — the goal here is **not** to see coordinated motion but to see each component respond and the call return without error. Once these all pass, on-hardware calibration of real `(t, r)` values is follow-on work outside this plan.

**Pre-flight (out-of-scope but blocking for right-wing):** Fix the `right-wing-gantry` motor reference in `rig1-2270-2` cloud config (`right-defense-movement` → `right-wing-movement`). Without this, right-wing translation drives the wrong gantry.

- [ ] **Step 1: Smoke test left defense**

Run: `python main.py --ld-left`
Expected: `Manual override: player=LEFT_D` → `Executing sequence (1 steps, player=LEFT_D, component=left-defense-hockey-player)` → `Done.`
Hardware: rod drives to / stays at `t=0, r=0`.
If `ResourceNotFoundError` for `left-defense-hockey-player` — confirm the primary part is rig1-2270-main and `ROBOT_ADDRESS` in `.env` points at it.

- [ ] **Step 2: Smoke test left wing**

Run: `python main.py --lw-left`
Expected: `component=left-wing-hockey-player`, `Done.`

- [ ] **Step 3: Smoke test center**

Run: `python main.py --center-left`
Expected: `component=center-hockey-player`, `Done.`
If `ResourceNotFoundError` — the `rig1-5072-3` part is not exposing `center-hockey-player` as a remote of `rig1-2270-main`. Confirm the remote is configured and reachable.

- [ ] **Step 4: Smoke test right defense**

Run: `python main.py --rd-left`
Expected: `component=right-defense-hockey-player`, `Done.`

- [ ] **Step 5: Smoke test right wing (only after the gantry-motor config bug is fixed)**

Run: `python main.py --rw-shot --left`
Expected: `component=right-wing-hockey-player`, `Done.`

- [ ] **Step 6: Mark migration complete**

No commit (smoke tests don't change files). Hand off to calibration: replace each `{"t": 0.0, "r": 0.0}` placeholder in `robot/playbook.py` with real values per-player, on hardware.

---

## Self-review notes

- Spec coverage: every section in the design spec is covered — playbook step format (Task 1 Step 2), public API preservation (Task 1 Steps 4–5), execution rewrite (Task 2 Step 2), `PlayerID → component-name` map (Task 2 Step 4), no reset-to-home (Task 2 Step 2), per-player smoke tests (Task 3).
- Placeholder scan: every code step shows the actual code; verification steps show the actual command and expected output. No "implement later," no "add error handling."
- Type consistency: `_PLAYER_TO_COMPONENT` keys/values match between Task 2 Step 2 and Step 4. `do_command` is the one-arg form (`step` dict). `get_rw_sequence(side, action)` signature unchanged from current.
- The `right-wing-gantry` motor reference bug is called out as a pre-flight in Task 3 and as out-of-scope in the design spec; not fixed here because viam config edits are out of scope.
