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


def puck_step_toward(puck, target, max_step):
    """Return a displacement vector from puck toward target, capped at max_step."""
    dx = target[0] - puck[0]
    dy = target[1] - puck[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= max_step or dist == 0.0:
        return (dx, dy)
    return (dx / dist * max_step, dy / dist * max_step)


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
