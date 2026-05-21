# Autonomous Puck Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-running loop that autonomously learns how a player's moves affect the puck and uses that learned, confidence-aware model to move the puck to targets — no human, no hardcoded geometry.

**Architecture:** A pure model module fits locally-weighted regression of `(player state, move) → puck response` from real samples and reports a confidence per query. A hardware loop module perceives the puck, decides explore-vs-exploit from that confidence, moves the player, records the outcome, and re-acquires the puck if contact slips. A thin CLI starts the loop.

**Tech Stack:** Python 3.13, numpy, viam-sdk, asyncio, pytest. Run via `uv`.

> **Commit convention:** every commit message ends with the trailer
> `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
> Work continues on the existing `coordinated-relay-routine` branch.

---

## File Structure

| File | Responsibility |
| ---- | -------------- |
| `robot/const.py` | *Modify.* Add `AUTO_*` constants. |
| `robot/puck_model.py` | *Create.* Dataset I/O, the locally-weighted model with confidence, control helpers. Pure — no hardware. |
| `robot/autonomy.py` | *Create.* The autonomous loop. |
| `auto.py` | *Create, top-level.* CLI to start the loop. |
| `tests/test_puck_model.py` | *Create.* Unit tests for the pure model. |
| *(removed)* | `robot/routine.py`, `robot/rod_model.py`, `robot/rod_session.py`, `rod.py`, `routine.py`, `tests/test_routine.py`, `tests/test_rod_model.py` — superseded. |

`robot/vision.py` is reused unchanged (`detect_puck_px`). `data/<player>.jsonl` is created at runtime by the loop.

---

## Task 1: Remove superseded files; add `AUTO_*` constants

**Files:**
- Delete: `robot/routine.py`, `robot/rod_model.py`, `robot/rod_session.py`, `rod.py`, `routine.py`, `tests/test_routine.py`, `tests/test_rod_model.py`
- Modify: `robot/const.py` (append at end)

- [ ] **Step 1: Remove the superseded files**

```bash
git rm robot/routine.py robot/rod_model.py robot/rod_session.py rod.py routine.py tests/test_routine.py tests/test_rod_model.py
```

- [ ] **Step 2: Confirm nothing else imports them**

Run: `uv run python -c "import main, run_play, simulate, move, robot.vision, robot.playbook, robot.execution; print('ok')"`
Expected: `ok` (the remaining modules do not depend on the removed ones).

- [ ] **Step 3: Add the constants**

Append to `robot/const.py`:

```python


# ============================================================
#  Autonomous puck control
# ============================================================

AUTO_MOVE_SPEED_MM_S      = 30.0   # gentle move speed — carry, never launch
AUTO_MAX_STEP_T           = 0.10   # clamp on one commanded translation move
AUTO_MAX_STEP_R           = 30.0   # clamp on one commanded rotation move (deg)
AUTO_CONTACT_MOVE_PX      = 8.0    # puck motion above which a move "made contact"
AUTO_TARGET_TOL_PX        = 22.0   # puck-to-target distance counting as arrived
AUTO_MAX_PUCK_STEP_PX     = 25.0   # max desired puck step per cycle
AUTO_MODEL_K              = 8      # neighbours for locally-weighted regression
AUTO_CONFIDENCE_SCALE     = 1.0    # neighbour distance at which confidence = 0.5
AUTO_CONFIDENCE_THRESHOLD = 0.5    # confidence at/above which the loop exploits
AUTO_EXPLORE_DT           = 0.07   # magnitude of exploratory translation moves
AUTO_EXPLORE_DR           = 20.0   # magnitude of exploratory rotation moves
```

- [ ] **Step 4: Verify the constants import**

Run: `uv run python -c "from robot.const import AUTO_MODEL_K, AUTO_CONFIDENCE_SCALE, AUTO_CONFIDENCE_THRESHOLD, AUTO_MAX_STEP_T, AUTO_MAX_STEP_R; print(AUTO_MODEL_K, AUTO_CONFIDENCE_SCALE, AUTO_CONFIDENCE_THRESHOLD, AUTO_MAX_STEP_T, AUTO_MAX_STEP_R)"`
Expected: `8 1.0 0.5 0.1 30.0`

- [ ] **Step 5: Commit**

```bash
git add robot/const.py
git commit -m "Remove superseded relay/rod modules; add autonomous-control constants"
```

---

## Task 2: `puck_model.py` — `Sample` and dataset I/O

**Files:**
- Create: `robot/puck_model.py`
- Create: `tests/test_puck_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_puck_model.py` with:

```python
"""Unit tests for the autonomous puck-control model."""

import os

from robot.puck_model import Sample, dataset_path, load_dataset, append_sample


def test_dataset_path_uses_data_dir_and_player():
    assert dataset_path("left-defense") == os.path.join("data", "left-defense.jsonl")


def test_load_dataset_missing_file_is_empty():
    assert load_dataset("data/does-not-exist-xyz.jsonl") == []


def test_append_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "ds.jsonl")
    append_sample(path, t=0.2, r=30.0, puck=(100.0, 200.0),
                  dt=0.05, dr=10.0, puck2=(108.0, 205.0))
    append_sample(path, t=0.3, r=40.0, puck=(108.0, 205.0),
                  dt=-0.05, dr=-10.0, puck2=(101.0, 199.0))
    samples = load_dataset(path)
    assert len(samples) == 2
    assert samples[0] == Sample(0.2, 30.0, 100.0, 200.0, 0.05, 10.0, 108.0, 205.0)
    assert samples[1].puck_x2 == 101.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_puck_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robot.puck_model'`

- [ ] **Step 3: Create `robot/puck_model.py`**

Create `robot/puck_model.py` with:

```python
"""Learned model of how a player's moves affect the puck.

Locally-weighted regression over real (player state, move, puck response)
samples, with a confidence score so the autonomous loop knows where the model
is reliable. See
docs/superpowers/specs/2026-05-21-autonomous-puck-control-design.md.
"""

import json
import os
from typing import NamedTuple

import numpy as np

from robot.const import (
    AUTO_MODEL_K, AUTO_MAX_STEP_T, AUTO_MAX_STEP_R, AUTO_CONFIDENCE_SCALE,
)


class Sample(NamedTuple):
    """One interaction: player (t, r) and puck before, the move, puck after."""
    t: float
    r: float
    puck_x: float
    puck_y: float
    dt: float
    dr: float
    puck_x2: float
    puck_y2: float


def dataset_path(player: str) -> str:
    """Path of the dataset file for a player."""
    return os.path.join("data", f"{player}.jsonl")


def load_dataset(path: str) -> list:
    """Load a .jsonl dataset into a list of Samples. Empty list if no file."""
    if not os.path.exists(path):
        return []
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(Sample(d["t"], d["r"], d["puck_x"], d["puck_y"],
                                  d["dt"], d["dr"], d["puck_x2"], d["puck_y2"]))
    return samples


def append_sample(path, t, r, puck, dt, dr, puck2) -> None:
    """Append one sample to a .jsonl dataset, creating the directory if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec = {"t": t, "r": r, "puck_x": puck[0], "puck_y": puck[1],
           "dt": dt, "dr": dr, "puck_x2": puck2[0], "puck_y2": puck2[1]}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
```

Note: the `numpy` and `AUTO_*` imports are used by later tasks — leave them.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_puck_model.py -v`
Expected: PASS — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/puck_model.py tests/test_puck_model.py
git commit -m "Add puck-model dataset I/O"
```

---

## Task 3: `PuckModel` + `predict`

**Files:**
- Modify: `robot/puck_model.py`
- Modify: `tests/test_puck_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_puck_model.py`:

```python
import numpy as np

from robot.puck_model import PuckModel


def _linear_samples(jacobian, n=40, seed=0):
    """Build n Samples whose puck response is exactly jacobian @ (dt, dr)."""
    rng = np.random.default_rng(seed)
    j = np.array(jacobian, float)
    out = []
    for _ in range(n):
        t = rng.uniform(0, 1)
        r = rng.uniform(0, 360)
        px = rng.uniform(0, 540)
        py = rng.uniform(0, 300)
        dt = rng.uniform(-0.1, 0.1)
        dr = rng.uniform(-30, 30)
        dp = j @ np.array([dt, dr])
        out.append(Sample(t, r, px, py, dt, dr, px + dp[0], py + dp[1]))
    return out


def test_predict_recovers_a_global_linear_jacobian():
    j = [[2.0, 0.1], [0.3, -1.5]]
    model = PuckModel(_linear_samples(j))
    dx, dy = model.predict((0.5, 180.0, 270.0, 150.0), d_t=0.05, d_r=10.0)
    expected = np.array(j) @ np.array([0.05, 10.0])
    assert abs(dx - expected[0]) < 1e-3
    assert abs(dy - expected[1]) < 1e-3


def test_predict_with_empty_model_is_zero():
    assert PuckModel([]).predict((0.5, 180.0, 270.0, 150.0), 0.05, 10.0) == (0.0, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_puck_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'PuckModel'`

- [ ] **Step 3: Add the implementation**

In `robot/puck_model.py`, append at the end of the file:

```python


class PuckModel:
    """Locally-weighted model of (player state, move) -> puck displacement.

    State is (t, r, puck_x, puck_y). For a query state a 2x2 Jacobian J with
    Δpuck ≈ J·(Δt, Δr) is least-squares fit from the nearest samples. An empty
    dataset is valid — predict/solve return zeros and confidence returns 0, so
    the loop simply explores until it has data.
    """

    def __init__(self, samples):
        self.samples = list(samples)
        if self.samples:
            self._states = np.array(
                [[s.t, s.r, s.puck_x, s.puck_y] for s in self.samples], float)
            self._scale = np.maximum(self._states.std(axis=0), 1e-6)
            self._moves = np.array([[s.dt, s.dr] for s in self.samples], float)
            self._responses = np.array(
                [[s.puck_x2 - s.puck_x, s.puck_y2 - s.puck_y]
                 for s in self.samples], float)

    def _neighbours(self, state):
        """(indices, weights, distances) of the nearest samples to a state."""
        q = np.array(state, float)
        d = np.linalg.norm((self._states - q) / self._scale, axis=1)
        k = min(AUTO_MODEL_K, len(self.samples))
        idx = np.argsort(d)[:k]
        nd = d[idx]
        bw = max(nd.mean(), 1e-6)
        w = np.exp(-(nd / bw) ** 2)
        return idx, w, nd

    def _local_jacobian(self, state):
        """Weighted least-squares 2x2 Jacobian near a query state."""
        idx, w, _ = self._neighbours(state)
        a = self._moves[idx]
        y = self._responses[idx]
        sw = np.sqrt(w)[:, None]
        jt, *_ = np.linalg.lstsq(a * sw, y * sw, rcond=None)
        return jt.T

    def predict(self, state, d_t, d_r):
        """Predicted puck displacement (dx, dy) for a move; (0, 0) with no data."""
        if not self.samples:
            return 0.0, 0.0
        j = self._local_jacobian(state)
        dp = j @ np.array([d_t, d_r], float)
        return float(dp[0]), float(dp[1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_puck_model.py -v`
Expected: PASS — 5 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/puck_model.py tests/test_puck_model.py
git commit -m "Add PuckModel with locally-weighted predict"
```

---

## Task 4: `PuckModel.confidence`

**Files:**
- Modify: `robot/puck_model.py`
- Modify: `tests/test_puck_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_puck_model.py`:

```python
def test_confidence_is_one_when_samples_sit_at_the_query():
    state = (0.5, 180.0, 270.0, 150.0)
    samples = [Sample(*state, 0.05, 10.0, 271.0, 151.0) for _ in range(10)]
    assert PuckModel(samples).confidence(state) == 1.0


def test_confidence_is_near_zero_far_from_the_data():
    state = (0.5, 180.0, 270.0, 150.0)
    samples = [Sample(*state, 0.05, 10.0, 271.0, 151.0) for _ in range(10)]
    far = (0.95, 350.0, 520.0, 290.0)
    assert PuckModel(samples).confidence(far) < 0.1


def test_confidence_with_empty_model_is_zero():
    assert PuckModel([]).confidence((0.5, 180.0, 270.0, 150.0)) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_puck_model.py -v`
Expected: FAIL — `AttributeError: 'PuckModel' object has no attribute 'confidence'`

- [ ] **Step 3: Add the implementation**

In `robot/puck_model.py`, add a `confidence` method to `PuckModel`, directly after `predict` (same indentation as `predict`):

```python

    def confidence(self, state):
        """Confidence in [0, 1] that the model knows this state.

        1.0 when samples sit exactly at the query, falling toward 0 as the
        nearest samples get farther away. 0.0 with no data.
        """
        if not self.samples:
            return 0.0
        _, _, nd = self._neighbours(state)
        mean_d = float(nd.mean())
        return AUTO_CONFIDENCE_SCALE / (AUTO_CONFIDENCE_SCALE + mean_d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_puck_model.py -v`
Expected: PASS — 8 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/puck_model.py tests/test_puck_model.py
git commit -m "Add PuckModel.confidence"
```

---

## Task 5: `PuckModel.solve`

**Files:**
- Modify: `robot/puck_model.py`
- Modify: `tests/test_puck_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_puck_model.py`. It reuses `_linear_samples` from Task 3:

```python
def test_solve_inverts_the_jacobian():
    j = [[2.0, 0.1], [0.3, -1.5]]
    model = PuckModel(_linear_samples(j))
    state = (0.5, 180.0, 270.0, 150.0)
    d_t, d_r, conf = model.solve(state, d_puck_desired=(0.2, -0.3))
    # Small desired step -> solved move stays within clamps -> reproduces it.
    dx, dy = model.predict(state, d_t, d_r)
    assert abs(dx - 0.2) < 1e-2
    assert abs(dy - (-0.3)) < 1e-2
    assert 0.0 <= conf <= 1.0


def test_solve_clamps_the_move():
    j = [[0.05, 0.0], [0.0, 0.05]]
    model = PuckModel(_linear_samples(j))
    d_t, d_r, _ = model.solve((0.5, 180.0, 270.0, 150.0),
                              d_puck_desired=(500.0, 500.0))
    assert abs(d_t) <= 0.10 + 1e-9
    assert abs(d_r) <= 30.0 + 1e-9


def test_solve_with_empty_model_is_zero():
    assert PuckModel([]).solve((0.5, 180.0, 270.0, 150.0), (10.0, 10.0)) == (0.0, 0.0, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_puck_model.py -v`
Expected: FAIL — `AttributeError: 'PuckModel' object has no attribute 'solve'`

- [ ] **Step 3: Add the implementation**

In `robot/puck_model.py`, add a `solve` method to `PuckModel`, directly after `confidence` (same indentation):

```python

    def solve(self, state, d_puck_desired):
        """Return (d_t, d_r, confidence) to achieve a desired puck displacement.

        d_t / d_r are clamped to AUTO_MAX_STEP_T / AUTO_MAX_STEP_R. With no data,
        returns a zero move and confidence 0.
        """
        if not self.samples:
            return 0.0, 0.0, 0.0
        j = self._local_jacobian(state)
        move = np.linalg.pinv(j) @ np.array(d_puck_desired, float)
        d_t = float(np.clip(move[0], -AUTO_MAX_STEP_T, AUTO_MAX_STEP_T))
        d_r = float(np.clip(move[1], -AUTO_MAX_STEP_R, AUTO_MAX_STEP_R))
        return d_t, d_r, self.confidence(state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_puck_model.py -v`
Expected: PASS — 11 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/puck_model.py tests/test_puck_model.py
git commit -m "Add PuckModel.solve for confidence-aware control"
```

---

## Task 6: Control helpers — `puck_step_toward` and `should_exploit`

**Files:**
- Modify: `robot/puck_model.py`
- Modify: `tests/test_puck_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_puck_model.py`:

```python
import math

from robot.puck_model import puck_step_toward, should_exploit


def test_puck_step_toward_within_range_returns_full_vector():
    assert puck_step_toward((100.0, 100.0), (110.0, 100.0), max_step=25.0) == (10.0, 0.0)


def test_puck_step_toward_clamps_to_max_step():
    step = puck_step_toward((0.0, 0.0), (300.0, 400.0), max_step=25.0)
    assert math.isclose(math.hypot(*step), 25.0, rel_tol=1e-9)


def test_puck_step_toward_at_target_is_zero():
    assert puck_step_toward((50.0, 50.0), (50.0, 50.0), max_step=25.0) == (0.0, 0.0)


def test_should_exploit_compares_confidence_to_threshold():
    assert should_exploit(0.8, threshold=0.5) is True
    assert should_exploit(0.5, threshold=0.5) is True
    assert should_exploit(0.3, threshold=0.5) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_puck_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'puck_step_toward'`

- [ ] **Step 3: Add the implementation**

In `robot/puck_model.py`, add at the end of the file (module-level functions, after the `PuckModel` class):

```python


def puck_step_toward(puck, target, max_step):
    """Return a displacement vector from puck toward target, capped at max_step."""
    dx = target[0] - puck[0]
    dy = target[1] - puck[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= max_step or dist == 0.0:
        return (dx, dy)
    return (dx / dist * max_step, dy / dist * max_step)


def should_exploit(confidence, threshold):
    """True when the model is confident enough to exploit rather than explore."""
    return confidence >= threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_puck_model.py -v`
Expected: PASS — 15 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/puck_model.py tests/test_puck_model.py
git commit -m "Add puck_step_toward and should_exploit control helpers"
```

---

## Task 7: The autonomous loop — `robot/autonomy.py`

`autonomy.py` drives real hardware, so it has no unit test; it is verified to import and is validated by running it.

**Files:**
- Create: `robot/autonomy.py`

- [ ] **Step 1: Create `robot/autonomy.py`**

Create `robot/autonomy.py` with:

```python
"""The autonomous puck-control loop.

Runs unattended: perceive the puck, decide explore-vs-exploit from the model's
confidence, move the player, record the outcome, and re-acquire the puck if
contact slips. See
docs/superpowers/specs/2026-05-21-autonomous-puck-control-design.md.
"""

import random

from viam.components.generic import Generic

from robot.const import (
    AUTO_MOVE_SPEED_MM_S, AUTO_CONTACT_MOVE_PX, AUTO_TARGET_TOL_PX,
    AUTO_MAX_PUCK_STEP_PX, AUTO_CONFIDENCE_THRESHOLD,
    AUTO_EXPLORE_DT, AUTO_EXPLORE_DR,
)
from robot.puck_model import (
    PuckModel, dataset_path, load_dataset, append_sample,
    puck_step_toward, should_exploit,
)
from robot.vision import detect_puck_px


def _dist(a, b):
    """Euclidean distance between two (x, y) points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _pick_target(model, puck, rng):
    """Pick a new target puck position from places the puck has already been.

    Returns a past puck-after position that is not too close to the current
    puck, or None if the dataset has nothing usable yet.
    """
    if not model.samples:
        return None
    for _ in range(10):
        s = rng.choice(model.samples)
        cand = (s.puck_x2, s.puck_y2)
        if _dist(cand, puck) > AUTO_TARGET_TOL_PX * 2:
            return cand
    return None


async def _reacquire(machine, comp, path, t, r):
    """Sweep translation until a step moves the puck; record each step.

    Returns the rod's (t, r) after sweeping. Used to regain contact when a move
    failed to move the puck.
    """
    direction = 1.0
    for _ in range(int(2.0 / AUTO_EXPLORE_DT) + 2):
        puck = await detect_puck_px(machine)
        if not (0.0 <= t + direction * AUTO_EXPLORE_DT <= 1.0):
            direction = -direction
        t2 = min(1.0, max(0.0, t + direction * AUTO_EXPLORE_DT))
        resp = await comp.do_command({"t": t2, "r": r,
                                      "speed_mm_per_sec": AUTO_MOVE_SPEED_MM_S})
        puck2 = await detect_puck_px(machine)
        base_t = t
        t, r = resp.get("t_final", t2), resp.get("r_final", r)
        if puck is not None and puck2 is not None:
            append_sample(path, t=base_t, r=r, puck=puck,
                          dt=t2 - base_t, dr=0.0, puck2=puck2)
            if _dist(puck, puck2) > AUTO_CONTACT_MOVE_PX:
                return t, r
    return t, r


async def run(machine, player, max_cycles=None):
    """Run the autonomous puck-control loop on a player.

    Each cycle: perceive the puck, choose/advance a self-generated target,
    ask the model for the move toward it, exploit that move if the model is
    confident or explore otherwise, execute, record the outcome, and
    re-acquire the puck if contact was lost. Runs until `max_cycles` (or
    forever if None). Homes the player on exit.
    """
    comp = Generic.from_robot(robot=machine, name=player)
    path = dataset_path(player)
    pos = await comp.do_command({"cmd": "get_position"})
    t, r = pos["t"], pos["r"]
    rng = random.Random()
    target = None
    cycle = 0
    try:
        while max_cycles is None or cycle < max_cycles:
            cycle += 1
            puck = await detect_puck_px(machine)
            if puck is None:
                print(f"cycle {cycle}: puck not visible — re-acquiring")
                t, r = await _reacquire(machine, comp, path, t, r)
                continue

            model = PuckModel(load_dataset(path))
            if target is not None and _dist(puck, target) <= AUTO_TARGET_TOL_PX:
                print(f"  reached target {target}")
                target = None
            if target is None:
                target = _pick_target(model, puck, rng)

            state = (t, r, puck[0], puck[1])
            if target is not None:
                desired = puck_step_toward(puck, target, AUTO_MAX_PUCK_STEP_PX)
                d_t, d_r, conf = model.solve(state, desired)
                exploit = should_exploit(conf, AUTO_CONFIDENCE_THRESHOLD)
            else:
                conf, exploit = 0.0, False
            if exploit:
                mode = "exploit"
            else:
                mode = "explore"
                d_t = rng.uniform(-AUTO_EXPLORE_DT, AUTO_EXPLORE_DT)
                d_r = rng.uniform(-AUTO_EXPLORE_DR, AUTO_EXPLORE_DR)

            t2 = min(1.0, max(0.0, t + d_t))
            r2 = (r + d_r) % 360.0
            resp = await comp.do_command({"t": t2, "r": r2,
                                          "speed_mm_per_sec": AUTO_MOVE_SPEED_MM_S})
            puck2 = await detect_puck_px(machine)
            new_t, new_r = resp.get("t_final", t2), resp.get("r_final", r2)
            if puck2 is not None:
                append_sample(path, t=t, r=r, puck=puck,
                              dt=t2 - t, dr=d_r, puck2=puck2)
                moved = _dist(puck, puck2)
                print(f"cycle {cycle} [{mode}] conf={conf:.2f} "
                      f"target={target} moved={moved:.0f}px")
                if moved < AUTO_CONTACT_MOVE_PX:
                    new_t, new_r = await _reacquire(machine, comp, path,
                                                    new_t, new_r)
            t, r = new_t, new_r
    finally:
        print("Homing player.")
        await comp.do_command({"t": 0.0, "r": 0.0})
```

- [ ] **Step 2: Verify the module imports**

Run: `uv run python -c "from robot.autonomy import run, _dist, _pick_target; print('ok', _dist((0,0),(3,4)))"`
Expected: `ok 5.0`

- [ ] **Step 3: Run the test suite (regression check)**

Run: `uv run pytest -q`
Expected: PASS — 15 passed (no test imports `autonomy`; this confirms nothing broke).

- [ ] **Step 4: Commit**

```bash
git add robot/autonomy.py
git commit -m "Add the autonomous puck-control loop"
```

---

## Task 8: CLI — `auto.py`

**Files:**
- Create: `auto.py`

- [ ] **Step 1: Create the CLI**

Create `auto.py` (repo root) with:

```python
"""Run the autonomous puck-control loop.

Usage:
  python auto.py [--player left-defense-hockey-player] [--cycles N]
"""

import argparse
import asyncio

from viam.robot.client import RobotClient

from robot.const import ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID
from robot.autonomy import run


async def _main():
    parser = argparse.ArgumentParser(description="Autonomous puck control")
    parser.add_argument("--player", default="left-defense-hockey-player")
    parser.add_argument("--cycles", type=int, default=None,
                        help="stop after N cycles (default: run until interrupted)")
    args = parser.parse_args()

    opts = RobotClient.Options.with_api_key(
        api_key=ROBOT_API_KEY, api_key_id=ROBOT_API_KEY_ID,
    )
    machine = await RobotClient.at_address(ROBOT_ADDRESS, opts)
    try:
        await run(machine, args.player, max_cycles=args.cycles)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(_main())
```

- [ ] **Step 2: Verify the CLI parses**

Run: `uv run python auto.py --help`
Expected: usage text showing `--player` and `--cycles`. No connection attempted (argparse exits before `_connect`).

- [ ] **Step 3: Run the test suite**

Run: `uv run pytest -q`
Expected: PASS — 15 passed.

- [ ] **Step 4: Commit**

```bash
git add auto.py
git commit -m "Add auto.py CLI for the autonomous loop"
```

---

## Task 9: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the autonomous loop in `README.md`**

In `README.md`, find the "Usage" section. After the "Drive all players to a pose (smoke test)" subsection (the `### ...` heading, its text, and its closing ``` fence), insert:

````markdown

### Autonomous puck control

Runs a self-learning loop on one player: it detects the puck, learns how its
moves affect it from real data, and moves the puck to self-chosen targets —
no human, no hardcoded geometry.

```bash
python auto.py                       # run until interrupted
python auto.py --cycles 30           # run a bounded session
python auto.py --player center-hockey-player
```

The loop writes `data/<player>.jsonl` and re-learns from it each cycle, so
learning accumulates across runs. Tuning lives in `robot/const.py` (`AUTO_*`).
See `docs/superpowers/specs/2026-05-21-autonomous-puck-control-design.md`.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document the autonomous puck-control loop"
```

---

## Manual Hardware Validation (after all tasks)

Not automated — run on the rig once the plan is complete:

1. Place the puck anywhere within `left-defense`'s reach. Run
   `uv run python auto.py --cycles 40`.
2. Confirm the loop: detects the puck each cycle, prints `explore`/`exploit`
   with a rising `conf` as `data/left-defense-hockey-player.jsonl` grows, moves
   the puck, and re-acquires it when a move misses.
3. Let it run longer (`auto.py` with no `--cycles`). Confirm the puck gets
   driven toward targets and `conf` climbs as the dataset fills.
4. Tune `AUTO_*` in `robot/const.py` if the loop explores forever
   (lower `AUTO_CONFIDENCE_THRESHOLD`) or exploits a weak model
   (raise it / raise `AUTO_CONFIDENCE_SCALE`).

---

## Self-Review Notes

- **Spec coverage:** files & architecture (Tasks 1-9); the learned model with
  confidence (Tasks 3-5); dataset I/O (Task 2); explore/exploit decision
  (Task 6 `should_exploit` + Task 7 loop); the autonomous loop with
  re-acquire and self-generated targets (Task 7); CLI (Task 8); constants
  (Task 1); testing — pure model unit-tested, loop validated by running
  (Tasks 2-6 tests + manual section). All spec sections map to a task.
- **Deviation from spec:** the spec's constants table lists
  `AUTO_VISION_RETRIES` / `AUTO_VISION_CALL_TIMEOUT_S`; the implementation
  instead reuses `robot/vision.py`'s existing (working) retry constants
  unchanged rather than renaming them, so those two `AUTO_*` names are not
  added. Vision behaviour is identical.
- **Type consistency:** `Sample` field order (`t, r, puck_x, puck_y, dt, dr,
  puck_x2, puck_y2`) is consistent across `load_dataset`, `append_sample`, the
  tests, and `autonomy.py`. `PuckModel.predict/.confidence/.solve` take `state`
  as a 4-tuple `(t, r, puck_x, puck_y)` at every call site. `puck_step_toward`,
  `should_exploit`, `_pick_target`, `_reacquire`, and `run` signatures match
  their callers.
