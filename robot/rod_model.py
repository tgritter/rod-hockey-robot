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
