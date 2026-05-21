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
