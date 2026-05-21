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
