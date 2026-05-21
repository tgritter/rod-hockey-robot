# One-Rod Puck Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single rod that learns, from real `(t, r, puck)` data, to catch the puck and carry it under control to a commanded target position.

**Architecture:** A pure model module fits a locally-weighted linear map from a rod move `(Δt, Δr)` to the puck's pixel displacement, using a dataset of real moves. A hardware module collects that dataset by driving the rod through probe moves, and runs a closed-loop controller that queries the model to carry the puck to a target. A thin CLI exposes `collect` and `control`.

**Tech Stack:** Python 3.13, numpy, viam-sdk, asyncio, pytest. Run via `uv`.

> **Commit convention:** every commit message ends with the trailer
> `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
> Work continues on the existing `coordinated-relay-routine` branch.

---

## File Structure

| File | Responsibility |
| ---- | -------------- |
| `robot/const.py` | *Modify.* Add `ROD_*` collection/model/control constants. |
| `robot/rod_model.py` | *Create.* Dataset I/O and the learned model (locally-weighted regression). Pure — no hardware. |
| `robot/rod_session.py` | *Create.* Hardware: autonomous data collection and the closed-loop controller. |
| `rod.py` | *Create, top-level.* CLI with `collect` and `control` subcommands. |
| `tests/test_rod_model.py` | *Create.* Unit tests for the pure model. |
| `robot/vision.py` | Reused unchanged — `detect_puck_px`. |

`data/rod_<rod>.jsonl` is created at runtime by the collector; it is not a plan artifact.

---

## Task 1: Constants

**Files:**
- Modify: `robot/const.py` (append at end)

- [ ] **Step 1: Add the constants**

Append to `robot/const.py`:

```python


# ============================================================
#  One-rod puck control
# ============================================================

# Data collection — probe move magnitudes and the base-state grid.
ROD_COLLECT_DT      = 0.06
ROD_COLLECT_DR      = 25.0
ROD_COLLECT_GRID_T  = 5
ROD_COLLECT_GRID_R  = 5
ROD_MOVE_SPEED_MM_S = 30.0     # gentle — nudge the puck, never launch it

# Model — locally-weighted regression.
ROD_MODEL_K         = 8        # neighbours per local fit
ROD_CONTACT_MOVE_PX = 8.0      # puck motion above which a sample is "contact"

# Controller.
ROD_MAX_PUCK_STEP_PX  = 25.0   # max desired puck step per control iteration
ROD_MAX_STEP_T        = 0.12   # clamp on one commanded translation move
ROD_MAX_STEP_R        = 45.0   # clamp on one commanded rotation move (degrees)
ROD_TARGET_TOL_PX     = 20.0   # puck-to-target distance counting as arrived
ROD_MAX_CONTROL_ITERS = 40
```

- [ ] **Step 2: Verify the constants import**

Run: `uv run python -c "from robot.const import ROD_MODEL_K, ROD_CONTACT_MOVE_PX, ROD_MAX_STEP_T, ROD_MAX_STEP_R, ROD_MAX_CONTROL_ITERS; print(ROD_MODEL_K, ROD_CONTACT_MOVE_PX, ROD_MAX_STEP_T, ROD_MAX_STEP_R, ROD_MAX_CONTROL_ITERS)"`
Expected: `8 8.0 0.12 45.0 40`

- [ ] **Step 3: Commit**

```bash
git add robot/const.py
git commit -m "Add one-rod puck-control constants"
```

---

## Task 2: Dataset I/O — `Sample`, `dataset_path`, `load_dataset`, `append_sample`

**Files:**
- Create: `robot/rod_model.py`
- Create: `tests/test_rod_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rod_model.py` with:

```python
"""Unit tests for the one-rod puck-control model."""

import os

from robot.rod_model import Sample, dataset_path, load_dataset, append_sample


def test_dataset_path_uses_data_dir_and_rod_name():
    assert dataset_path("left-defense") == os.path.join("data", "rod_left-defense.jsonl")


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


def test_load_dataset_skips_blank_lines(tmp_path):
    path = str(tmp_path / "ds.jsonl")
    append_sample(path, t=0.0, r=0.0, puck=(1.0, 2.0), dt=0.0, dr=0.0, puck2=(1.0, 2.0))
    with open(path, "a") as f:
        f.write("\n")
    assert len(load_dataset(path)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rod_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robot.rod_model'`

- [ ] **Step 3: Create `robot/rod_model.py` with the dataset I/O**

Create `robot/rod_model.py` with:

```python
"""Learned model of one rod's puck-control behaviour.

From real (t, r, puck) samples, fits a local linear map from a rod move
(Δt, Δr) to the puck's pixel displacement. See
docs/superpowers/specs/2026-05-21-one-rod-puck-control-design.md.
"""

import json
import os
from typing import NamedTuple

import numpy as np

from robot.const import (
    ROD_MODEL_K,
    ROD_CONTACT_MOVE_PX,
    ROD_MAX_STEP_T,
    ROD_MAX_STEP_R,
)


class Sample(NamedTuple):
    """One recorded move: rod (t, r) and puck before, the move, puck after."""
    t: float
    r: float
    puck_x: float
    puck_y: float
    dt: float
    dr: float
    puck_x2: float
    puck_y2: float


def dataset_path(rod_name: str) -> str:
    """Path of the dataset file for a given rod."""
    return os.path.join("data", f"rod_{rod_name}.jsonl")


def load_dataset(path: str) -> list:
    """Load a .jsonl dataset into a list of Samples."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(Sample(
                d["t"], d["r"], d["puck_x"], d["puck_y"],
                d["dt"], d["dr"], d["puck_x2"], d["puck_y2"],
            ))
    return samples


def append_sample(path: str, t, r, puck, dt, dr, puck2) -> None:
    """Append one sample to a .jsonl dataset, creating the directory if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec = {"t": t, "r": r, "puck_x": puck[0], "puck_y": puck[1],
           "dt": dt, "dr": dr, "puck_x2": puck2[0], "puck_y2": puck2[1]}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rod_model.py -v`
Expected: PASS — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/rod_model.py tests/test_rod_model.py
git commit -m "Add rod-model dataset I/O"
```

---

## Task 3: `puck_step_toward`

**Files:**
- Modify: `robot/rod_model.py`
- Modify: `tests/test_rod_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rod_model.py`:

```python
import math

from robot.rod_model import puck_step_toward


def test_puck_step_toward_within_range_returns_full_vector():
    assert puck_step_toward((100.0, 100.0), (110.0, 100.0), max_step=25.0) == (10.0, 0.0)


def test_puck_step_toward_clamps_to_max_step():
    step = puck_step_toward((0.0, 0.0), (300.0, 400.0), max_step=25.0)
    assert math.isclose(math.hypot(*step), 25.0, rel_tol=1e-9)


def test_puck_step_toward_at_target_is_zero():
    assert puck_step_toward((50.0, 50.0), (50.0, 50.0), max_step=25.0) == (0.0, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rod_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'puck_step_toward'`

- [ ] **Step 3: Add the implementation**

In `robot/rod_model.py`, add after `append_sample`:

```python


def puck_step_toward(puck, target, max_step):
    """Return a displacement vector from puck toward target, capped at max_step."""
    dx = target[0] - puck[0]
    dy = target[1] - puck[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= max_step or dist == 0.0:
        return (dx, dy)
    return (dx / dist * max_step, dy / dist * max_step)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rod_model.py -v`
Expected: PASS — 6 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/rod_model.py tests/test_rod_model.py
git commit -m "Add puck_step_toward helper"
```

---

## Task 4: `RodModel` + `predict`

**Files:**
- Modify: `robot/rod_model.py`
- Modify: `tests/test_rod_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rod_model.py`:

```python
import numpy as np

from robot.rod_model import RodModel


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
    model = RodModel(_linear_samples(j))
    dx, dy = model.predict(0.5, 180.0, (270.0, 150.0), d_t=0.05, d_r=10.0)
    expected = np.array(j) @ np.array([0.05, 10.0])
    assert abs(dx - expected[0]) < 1e-3
    assert abs(dy - expected[1]) < 1e-3


def test_rodmodel_rejects_empty_dataset():
    try:
        RodModel([])
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rod_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'RodModel'`

- [ ] **Step 3: Add the implementation**

In `robot/rod_model.py`, add at the end of the file:

```python


class RodModel:
    """Locally-weighted linear model of rod (Δt, Δr) -> puck displacement.

    Fit from a dataset of real Samples. For any query state, a 2x2 Jacobian J
    with Δpuck ≈ J·(Δt, Δr) is least-squares fit from the nearest samples.
    """

    def __init__(self, samples):
        if not samples:
            raise ValueError("RodModel needs at least one sample")
        self.samples = list(samples)
        # Query state per sample: (t, r, puck_x, puck_y).
        self._states = np.array(
            [[s.t, s.r, s.puck_x, s.puck_y] for s in self.samples], float)
        # Per-dimension scale so neighbour distances are comparable.
        self._scale = np.maximum(self._states.std(axis=0), 1e-6)
        # Commanded moves and observed puck responses.
        self._moves = np.array([[s.dt, s.dr] for s in self.samples], float)
        self._responses = np.array(
            [[s.puck_x2 - s.puck_x, s.puck_y2 - s.puck_y] for s in self.samples],
            float)

    def _neighbours(self, t, r, puck):
        """Return (indices, weights) of the nearest samples to a query state."""
        q = np.array([t, r, puck[0], puck[1]], float)
        d = np.linalg.norm((self._states - q) / self._scale, axis=1)
        k = min(ROD_MODEL_K, len(self.samples))
        idx = np.argsort(d)[:k]
        bandwidth = max(d[idx].mean(), 1e-6)
        weights = np.exp(-(d[idx] / bandwidth) ** 2)
        return idx, weights

    def _local_jacobian(self, t, r, puck):
        """Weighted least-squares 2x2 Jacobian near a query state, with the
        neighbour indices used to fit it."""
        idx, weights = self._neighbours(t, r, puck)
        a = self._moves[idx]
        y = self._responses[idx]
        sw = np.sqrt(weights)[:, None]
        jt, *_ = np.linalg.lstsq(a * sw, y * sw, rcond=None)
        return jt.T, idx

    def predict(self, t, r, puck, d_t, d_r):
        """Predict the puck displacement (dx, dy) for a move (d_t, d_r)."""
        j, _ = self._local_jacobian(t, r, puck)
        dp = j @ np.array([d_t, d_r], float)
        return float(dp[0]), float(dp[1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rod_model.py -v`
Expected: PASS — 8 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/rod_model.py tests/test_rod_model.py
git commit -m "Add RodModel with locally-weighted predict"
```

---

## Task 5: `RodModel.solve`

**Files:**
- Modify: `robot/rod_model.py`
- Modify: `tests/test_rod_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rod_model.py`. It reuses `_linear_samples` defined in Task 4:

```python
def test_solve_inverts_the_jacobian():
    j = [[2.0, 0.1], [0.3, -1.5]]
    model = RodModel(_linear_samples(j))
    d_t, d_r, controllable = model.solve(0.5, 180.0, (270.0, 150.0),
                                         d_puck_desired=(4.0, -6.0))
    # Applying the solved move should reproduce the desired puck displacement.
    dx, dy = model.predict(0.5, 180.0, (270.0, 150.0), d_t, d_r)
    assert abs(dx - 4.0) < 1e-2
    assert abs(dy - (-6.0)) < 1e-2
    assert controllable is True


def test_solve_clamps_the_move():
    j = [[0.05, 0.0], [0.0, 0.05]]   # tiny gains -> huge move requested
    model = RodModel(_linear_samples(j))
    d_t, d_r, _ = model.solve(0.5, 180.0, (270.0, 150.0),
                              d_puck_desired=(500.0, 500.0))
    assert abs(d_t) <= 0.12 + 1e-9
    assert abs(d_r) <= 45.0 + 1e-9


def test_solve_reports_not_controllable_when_puck_never_moves():
    # Samples where the puck barely responds to any move.
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(40):
        t = rng.uniform(0, 1); r = rng.uniform(0, 360)
        px = rng.uniform(0, 540); py = rng.uniform(0, 300)
        dt = rng.uniform(-0.1, 0.1); dr = rng.uniform(-30, 30)
        samples.append(Sample(t, r, px, py, dt, dr, px, py))   # no puck motion
    model = RodModel(samples)
    _, _, controllable = model.solve(0.5, 180.0, (270.0, 150.0),
                                     d_puck_desired=(10.0, 10.0))
    assert controllable is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rod_model.py -v`
Expected: FAIL — `AttributeError: 'RodModel' object has no attribute 'solve'`

- [ ] **Step 3: Add the implementation**

In `robot/rod_model.py`, add a `solve` method to `RodModel`, directly after `predict` (same indentation as `predict`):

```python

    def solve(self, t, r, puck, d_puck_desired):
        """Return (d_t, d_r, controllable) to achieve a desired puck displacement.

        d_t / d_r are clamped to ROD_MAX_STEP_T / ROD_MAX_STEP_R. `controllable`
        is False when the local neighbourhood shows no reliable contact (most
        neighbours barely moved the puck).
        """
        j, idx = self._local_jacobian(t, r, puck)
        move = np.linalg.pinv(j) @ np.array(d_puck_desired, float)
        d_t = float(np.clip(move[0], -ROD_MAX_STEP_T, ROD_MAX_STEP_T))
        d_r = float(np.clip(move[1], -ROD_MAX_STEP_R, ROD_MAX_STEP_R))
        magnitudes = np.linalg.norm(self._responses[idx], axis=1)
        controllable = bool(np.median(magnitudes) > ROD_CONTACT_MOVE_PX)
        return d_t, d_r, controllable
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rod_model.py -v`
Expected: PASS — 11 passed.

- [ ] **Step 5: Commit**

```bash
git add robot/rod_model.py tests/test_rod_model.py
git commit -m "Add RodModel.solve for model-based control"
```

---

## Task 6: Data collection — `collect_dataset`

`collect_dataset` drives real hardware, so it has no unit test; it is verified to import and is hardware-validated on the rig.

**Files:**
- Create: `robot/rod_session.py`

- [ ] **Step 1: Create `robot/rod_session.py` with the collector**

Create `robot/rod_session.py` with:

```python
"""Hardware session for one-rod puck control: data collection and control.

See docs/superpowers/specs/2026-05-21-one-rod-puck-control-design.md.
"""

import asyncio

from viam.components.generic import Generic

from robot.const import (
    ROD_COLLECT_DT, ROD_COLLECT_DR, ROD_COLLECT_GRID_T, ROD_COLLECT_GRID_R,
    ROD_MOVE_SPEED_MM_S, ROD_MAX_PUCK_STEP_PX, ROD_TARGET_TOL_PX,
    ROD_MAX_CONTROL_ITERS,
)
from robot.rod_model import dataset_path, append_sample, puck_step_toward
from robot.vision import detect_puck_px


def _dist(a, b):
    """Euclidean distance between two (x, y) points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


async def _puck_or_pause(machine):
    """Detect the puck; if it is not visible, ask the operator to reposition it."""
    while True:
        p = await detect_puck_px(machine)
        if p is not None:
            return p
        input("  puck not visible — reposition it on the rod and press Enter...")


async def collect_dataset(machine, rod_name):
    """Drive the rod through a grid of probe moves, recording real samples.

    For each base state on a (t, r) grid, the rod issues four small probe moves
    (+/- Δt, +/- Δr), returning to the base state between probes. Every probe is
    recorded as a sample in data/rod_<rod>.jsonl. The puck is observed before
    and after each probe via vision-1. Slow, gentle moves throughout.
    """
    comp = Generic.from_robot(robot=machine, name=rod_name)
    path = dataset_path(rod_name)
    base_ts = [i / (ROD_COLLECT_GRID_T - 1) for i in range(ROD_COLLECT_GRID_T)]
    base_rs = [i * 300.0 / (ROD_COLLECT_GRID_R - 1) for i in range(ROD_COLLECT_GRID_R)]
    probes = [(ROD_COLLECT_DT, 0.0), (-ROD_COLLECT_DT, 0.0),
              (0.0, ROD_COLLECT_DR), (0.0, -ROD_COLLECT_DR)]
    n = 0
    try:
        for bt in base_ts:
            for br in base_rs:
                await comp.do_command({"t": bt, "r": br,
                                       "speed_mm_per_sec": ROD_MOVE_SPEED_MM_S})
                for d_t, d_r in probes:
                    t2 = min(1.0, max(0.0, bt + d_t))
                    r2 = (br + d_r) % 360.0
                    puck = await _puck_or_pause(machine)
                    await comp.do_command({"t": t2, "r": r2,
                                           "speed_mm_per_sec": ROD_MOVE_SPEED_MM_S})
                    puck2 = await detect_puck_px(machine)
                    if puck2 is not None:
                        append_sample(path, t=bt, r=br, puck=puck,
                                      dt=t2 - bt, dr=d_r, puck2=puck2)
                        n += 1
                        print(f"sample {n}: base=({bt:.2f},{br:.0f}) "
                              f"move=({t2 - bt:+.2f},{d_r:+.0f}) "
                              f"puck {puck} -> {puck2}")
                    await comp.do_command({"t": bt, "r": br,
                                           "speed_mm_per_sec": ROD_MOVE_SPEED_MM_S})
        print(f"Collected {n} samples -> {path}")
    finally:
        await comp.do_command({"t": 0.0, "r": 0.0})
```

- [ ] **Step 2: Verify the module imports**

Run: `uv run python -c "from robot.rod_session import collect_dataset, _dist; print('ok', _dist((0,0),(3,4)))"`
Expected: `ok 5.0`

- [ ] **Step 3: Run the test suite (regression check)**

Run: `uv run pytest -q`
Expected: PASS — all tests pass (no test imports `rod_session`; this confirms nothing broke).

- [ ] **Step 4: Commit**

```bash
git add robot/rod_session.py
git commit -m "Add autonomous data collection for one-rod control"
```

---

## Task 7: The controller — `carry_puck`

`carry_puck` drives real hardware, so it has no unit test; it is verified to import and is hardware-validated on the rig. It reuses `_dist` and `_puck_or_pause` from Task 6.

**Files:**
- Modify: `robot/rod_session.py`

- [ ] **Step 1: Add the `ROD_CONTACT_MOVE_PX` import**

In `robot/rod_session.py`, add `ROD_CONTACT_MOVE_PX` to the existing
`from robot.const import (...)` block (it is needed by `_reacquire` below).
The block becomes:

```python
from robot.const import (
    ROD_COLLECT_DT, ROD_COLLECT_DR, ROD_COLLECT_GRID_T, ROD_COLLECT_GRID_R,
    ROD_MOVE_SPEED_MM_S, ROD_MAX_PUCK_STEP_PX, ROD_TARGET_TOL_PX,
    ROD_MAX_CONTROL_ITERS, ROD_CONTACT_MOVE_PX,
)
```

- [ ] **Step 2: Add `carry_puck` to `robot/rod_session.py`**

Append to `robot/rod_session.py`:

```python


async def _reacquire(machine, comp, rod_name, t, r):
    """Probe four small moves to regain puck contact after control is lost.

    Returns (t, r, regained). Each probe is recorded as a sample; contact is
    regained when a probe moves the puck more than ROD_CONTACT_MOVE_PX.
    """
    path = dataset_path(rod_name)
    probes = [(ROD_COLLECT_DT, 0.0), (-ROD_COLLECT_DT, 0.0),
              (0.0, ROD_COLLECT_DR), (0.0, -ROD_COLLECT_DR)]
    for d_t, d_r in probes:
        puck = await detect_puck_px(machine)
        if puck is None:
            continue
        t2 = min(1.0, max(0.0, t + d_t))
        r2 = (r + d_r) % 360.0
        resp = await comp.do_command({"t": t2, "r": r2,
                                      "speed_mm_per_sec": ROD_MOVE_SPEED_MM_S})
        puck2 = await detect_puck_px(machine)
        base_t, base_r = t, r
        t, r = resp.get("t_final", t2), resp.get("r_final", r2)
        if puck2 is not None:
            append_sample(path, t=base_t, r=base_r, puck=puck,
                          dt=t2 - base_t, dr=d_r, puck2=puck2)
            if _dist(puck, puck2) > ROD_CONTACT_MOVE_PX:
                return t, r, True
    return t, r, False


async def carry_puck(machine, rod_name, model, target):
    """Closed-loop carry the puck to a target camera-pixel position.

    Each iteration: observe the puck, ask the model for the move that nudges it
    toward the target, command that gentle move, and record the real outcome as
    a new sample. Returns True if the puck reaches the target. Homes on exit.
    """
    comp = Generic.from_robot(robot=machine, name=rod_name)
    path = dataset_path(rod_name)
    pos = await comp.do_command({"cmd": "get_position"})
    t, r = pos["t"], pos["r"]
    try:
        for i in range(ROD_MAX_CONTROL_ITERS):
            puck = await detect_puck_px(machine)
            if puck is None:
                print("Lost sight of the puck — aborting.")
                return False
            err = _dist(puck, target)
            print(f"iter {i}: puck={puck} target={target} err={err:.0f}px "
                  f"t={t:.2f} r={r:.0f}")
            if err <= ROD_TARGET_TOL_PX:
                print("Puck delivered to target.")
                return True
            d_puck = puck_step_toward(puck, target, ROD_MAX_PUCK_STEP_PX)
            d_t, d_r, controllable = model.solve(t, r, puck, d_puck)
            if not controllable:
                print("  no reliable control here — re-acquiring the puck")
                t, r, regained = await _reacquire(machine, comp, rod_name, t, r)
                if not regained:
                    print("Could not re-acquire the puck — aborting.")
                    return False
                continue
            t2 = min(1.0, max(0.0, t + d_t))
            r2 = (r + d_r) % 360.0
            resp = await comp.do_command({"t": t2, "r": r2,
                                          "speed_mm_per_sec": ROD_MOVE_SPEED_MM_S})
            puck2 = await detect_puck_px(machine)
            if puck2 is not None:
                append_sample(path, t=t, r=r, puck=puck,
                              dt=t2 - t, dr=d_r, puck2=puck2)
            t, r = resp.get("t_final", t2), resp.get("r_final", r2)
        print("Hit the iteration cap without reaching the target.")
        return False
    finally:
        await comp.do_command({"t": 0.0, "r": 0.0})
```

- [ ] **Step 3: Verify the module imports**

Run: `uv run python -c "from robot.rod_session import collect_dataset, carry_puck; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Run the test suite (regression check)**

Run: `uv run pytest -q`
Expected: PASS — all tests pass.

- [ ] **Step 5: Commit**

```bash
git add robot/rod_session.py
git commit -m "Add closed-loop carry_puck controller"
```

---

## Task 8: CLI — `rod.py`

**Files:**
- Create: `rod.py`

- [ ] **Step 1: Create the CLI**

Create `rod.py` (repo root) with:

```python
"""Drive and train one rod for puck control.

Usage:
  python rod.py collect [--rod left-defense]
  python rod.py control --target X Y [--rod left-defense]
"""

import argparse
import asyncio

from viam.robot.client import RobotClient

from robot.const import ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID
from robot.rod_model import dataset_path, load_dataset, RodModel
from robot.rod_session import collect_dataset, carry_puck


async def _connect():
    opts = RobotClient.Options.with_api_key(
        api_key=ROBOT_API_KEY, api_key_id=ROBOT_API_KEY_ID,
    )
    return await RobotClient.at_address(ROBOT_ADDRESS, opts)


async def _main():
    parser = argparse.ArgumentParser(description="One-rod puck control")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Drive the rod and record real samples")
    p_collect.add_argument("--rod", default="left-defense-hockey-player")

    p_control = sub.add_parser("control", help="Carry the puck to a target")
    p_control.add_argument("--rod", default="left-defense-hockey-player")
    p_control.add_argument("--target", nargs=2, type=float, required=True,
                           metavar=("X", "Y"))

    args = parser.parse_args()

    machine = await _connect()
    try:
        if args.cmd == "collect":
            await collect_dataset(machine, args.rod)
        else:
            samples = load_dataset(dataset_path(args.rod))
            print(f"Loaded {len(samples)} samples; fitting model.")
            model = RodModel(samples)
            await carry_puck(machine, args.rod, model, tuple(args.target))
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(_main())
```

- [ ] **Step 2: Verify the CLI parses**

Run: `uv run python rod.py --help`
Expected: usage text listing the `collect` and `control` subcommands. No connection attempted (argparse prints help and exits before `_connect`).

- [ ] **Step 3: Verify a subcommand parses**

Run: `uv run python rod.py control --help`
Expected: help text showing `--rod` and `--target X Y`.

- [ ] **Step 4: Run the test suite**

Run: `uv run pytest -q`
Expected: PASS — all tests pass.

- [ ] **Step 5: Commit**

```bash
git add rod.py
git commit -m "Add rod.py CLI for collect and control"
```

---

## Task 9: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the one-rod tool in `README.md`**

In `README.md`, after the "Drive all players to a pose (smoke test)" subsection (the `### ...` heading, its text, and its closing ```` ``` ```` fence), insert:

````markdown

### One-rod puck control (learns from reality)

Trains a single rod to control the puck by learning the rig's real behaviour
from `vision-1` + `hockey-player` data — no simulation, no hardcoded geometry.

```bash
python rod.py collect                       # drive the rod, record real samples
python rod.py control --target 300 220      # carry the puck to a pixel target
```

`collect` writes `data/rod_<rod>.jsonl`. `control` fits a locally-weighted
model from that dataset and runs a closed-loop controller. Tuning lives in
`robot/const.py` (`ROD_*`). See
`docs/superpowers/specs/2026-05-21-one-rod-puck-control-design.md`.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document the one-rod puck-control tool"
```

---

## Manual Hardware Validation (after all tasks)

Not automated — run on the rig once the plan is complete:

1. Place the puck on the `left-defense` rod. Run `uv run python rod.py collect`.
   Confirm it records samples and produces `data/rod_left-defense-hockey-player.jsonl`
   with a mix of contact (puck moved) and no-contact samples.
2. Run `uv run python rod.py control --target X Y` for a target near the rod's
   reachable area. Watch the controller close the gap to the puck and carry it.
3. If control is jittery or weak, tune `ROD_*` in `robot/const.py` (step sizes,
   `ROD_MODEL_K`, `ROD_MAX_PUCK_STEP_PX`) and/or run `collect` again to enrich
   the dataset — it is append-only and the controller also adds samples online.

---

## Self-Review Notes

- **Spec coverage:** interfaces & files (Tasks 1–9); data collection / "training
  on reality" (Task 6); the locally-weighted-regression model (Tasks 4–5);
  dataset I/O (Task 2); `puck_step_toward` (Task 3); the closed-loop controller
  with re-acquire and online sample-adding (Task 7); CLI `collect`/`control`
  (Task 8); constants (Task 1); testing — pure model unit-tested, hardware
  validated manually (Tasks 4–5 tests + manual section). All spec sections map
  to a task.
- **Deviation from spec:** the spec's controller "step 7" adds a sample only on
  surprising prediction error; this plan appends *every* control move as a
  sample (simpler, strictly more data), so `ROD_SURPRISE_TOL_PX` is not used and
  is omitted from the constants.
- **Type consistency:** `Sample` field order (`t, r, puck_x, puck_y, dt, dr,
  puck_x2, puck_y2`) is consistent across `load_dataset`, `append_sample`, the
  tests, and both hardware functions. `RodModel.predict`/`.solve`,
  `puck_step_toward`, `dataset_path`, `collect_dataset`, `carry_puck` signatures
  match every call site.
