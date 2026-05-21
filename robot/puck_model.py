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
