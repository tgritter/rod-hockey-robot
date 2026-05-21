"""Hardware session for one-rod puck control: data collection and control.

See docs/superpowers/specs/2026-05-21-one-rod-puck-control-design.md.
"""

import random

from viam.components.generic import Generic

from robot.const import (
    ROD_COLLECT_DT, ROD_COLLECT_DR,
    ROD_CARRY_STEP_MIN, ROD_CARRY_STEP_MAX, ROD_CARRY_WOBBLE_DR,
    ROD_MOVE_SPEED_MM_S, ROD_MAX_PUCK_STEP_PX, ROD_TARGET_TOL_PX,
    ROD_MAX_CONTROL_ITERS, ROD_CONTACT_MOVE_PX,
)
from robot.rod_model import dataset_path, append_sample, puck_step_toward
from robot.vision import detect_puck_px


def _dist(a, b):
    """Euclidean distance between two (x, y) points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


async def _record_move(machine, comp, path, t, r, d_t, d_r):
    """Command one gentle move and record it as a sample.

    Returns (new_t, new_r, puck2, moved_px). puck2 and moved_px are None if the
    puck was not seen before or after the move.
    """
    puck = await detect_puck_px(machine)
    if puck is None:
        return t, r, None, None
    t2 = min(1.0, max(0.0, t + d_t))
    r2 = (r + d_r) % 360.0
    resp = await comp.do_command({"t": t2, "r": r2,
                                  "speed_mm_per_sec": ROD_MOVE_SPEED_MM_S})
    puck2 = await detect_puck_px(machine)
    new_t, new_r = resp.get("t_final", t2), resp.get("r_final", r2)
    if puck2 is None:
        return new_t, new_r, None, None
    append_sample(path, t=t, r=r, puck=puck, dt=t2 - t, dr=d_r, puck2=puck2)
    return new_t, new_r, puck2, _dist(puck, puck2)


async def collect_dataset(machine, rod_name):
    """Collect contact samples by carrying the puck across the rod once.

    Pushing the puck (a positive-dt move into it) is the only motion proven to
    keep contact, so collection is a single carry: read the rod's current
    (t, r), then push toward t=1 with small positive-dt steps, each with a
    slight rotation wobble so the dataset also sees how rotation affects the
    carry. Every step is a recorded contact sample. The puck must already be
    placed against the player at the t=0 end. Samples append to
    data/rod_<rod>.jsonl, so repeated carries at different rotations build the
    dataset up. Homes the rod on exit.
    """
    comp = Generic.from_robot(robot=machine, name=rod_name)
    path = dataset_path(rod_name)
    pos = await comp.do_command({"cmd": "get_position"})
    t, r = pos["t"], pos["r"]
    n = 0
    try:
        print(f"Carry starting at t={t:.2f} r={r:.0f}", flush=True)
        while t < 1.0 - 1e-9:
            d_t = min(random.uniform(ROD_CARRY_STEP_MIN, ROD_CARRY_STEP_MAX),
                      1.0 - t)
            d_r = random.uniform(-ROD_CARRY_WOBBLE_DR, ROD_CARRY_WOBBLE_DR)
            t, r, puck2, moved = await _record_move(machine, comp, path,
                                                    t, r, d_t, d_r)
            if moved is None:
                print("  lost sight of the puck — ending carry", flush=True)
                break
            n += 1
            print(f"sample {n}: t={t:.2f} r={r:.0f} moved {moved:.0f}px",
                  flush=True)
        print(f"Carry done: {n} samples -> {path}", flush=True)
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
