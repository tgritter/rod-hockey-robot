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
