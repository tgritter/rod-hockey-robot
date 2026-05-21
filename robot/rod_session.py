"""Hardware session for one-rod puck control: data collection and control.

See docs/superpowers/specs/2026-05-21-one-rod-puck-control-design.md.
"""

import asyncio

from viam.components.generic import Generic

from robot.const import (
    ROD_COLLECT_DT, ROD_COLLECT_DR, ROD_COLLECT_GRID_T, ROD_COLLECT_GRID_R,
    ROD_MOVE_SPEED_MM_S, ROD_MAX_PUCK_STEP_PX, ROD_TARGET_TOL_PX,
    ROD_MAX_CONTROL_ITERS, ROD_CONTACT_MOVE_PX,
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
