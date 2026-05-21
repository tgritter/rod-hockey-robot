"""Coordinated five-rod relay routine.

Relays the puck through every rod in order:
  Left D -> Right D -> Center -> Left Wing -> Right Wing -> SHOT

Each receiving rod follows the puck's detected position; vision confirms the
puck has reached the next rod before that rod acts. See
docs/superpowers/specs/2026-05-21-coordinated-relay-routine-design.md.
"""

import asyncio

from engine.constants import (
    PlayerID,
    center_x, min_y_center, max_y_center,
    right_wing_x, min_y_right_wing, max_y_right_wing,
    right_d_x, min_y_right_d, max_y_right_d,
    left_d_x, min_y_left_d, max_y_left_d,
    left_wing_x, min_y_left_wing, max_y_left_wing,
)

from robot.const import (
    RELAY_CAMERA,
    RELAY_GATE_TOLERANCE_PX,
    RELAY_GATE_TIMEOUT_S,
    RELAY_VISION_POLL_INTERVAL_S,
)
from robot.playbook import RELAY

from viam.components.generic import Generic

from robot.execution import _PLAYER_TO_COMPONENT
from robot.vision import detect_puck

# Per-rod translation band: t=0 -> min_y, t=1 -> max_y (game pixels).
_ROD_Y_BAND = {
    PlayerID.LEFT_D:     (min_y_left_d, max_y_left_d),
    PlayerID.RIGHT_D:    (min_y_right_d, max_y_right_d),
    PlayerID.CENTER:     (min_y_center, max_y_center),
    PlayerID.LEFT_WING:  (min_y_left_wing, max_y_left_wing),
    PlayerID.RIGHT_WING: (min_y_right_wing, max_y_right_wing),
}


def puck_y_to_t(player_id: PlayerID, puck_y: float) -> float:
    """Map a puck game-y coordinate to a normalized [0, 1] translation for a rod."""
    min_y, max_y = _ROD_Y_BAND[player_id]
    t = (puck_y - min_y) / (max_y - min_y)
    return max(0.0, min(1.0, t))


# Per-rod x position (game pixels) — the gate axis for puck-arrival checks.
_ROD_X = {
    PlayerID.LEFT_D:     left_d_x,
    PlayerID.RIGHT_D:    right_d_x,
    PlayerID.CENTER:     center_x,
    PlayerID.LEFT_WING:  left_wing_x,
    PlayerID.RIGHT_WING: right_wing_x,
}


def puck_reached_rod(puck_x: float, rod_x: float, tol: float) -> bool:
    """True if the puck's game-x is within `tol` pixels of a rod's x position."""
    return abs(puck_x - rod_x) <= tol


def format_relay_plan() -> str:
    """Return a human-readable description of the relay plan (no hardware)."""
    lines = ["Relay plan (Left D -> Right D -> Center -> Left Wing -> Right Wing):"]
    last = len(RELAY) - 1
    for i, leg in enumerate(RELAY):
        player = leg["player"]
        action = "SHOT" if i == last else "pass"
        lines.append(
            f"  Leg {i + 1}: {player.name:11s} "
            f"receive r={leg['receive_r']} (t follows puck_y), "
            f"{action} {leg['pass_step']}"
        )
        if i != last:
            nxt = RELAY[i + 1]["player"]
            lines.append(
                f"          gate: wait until puck_x within "
                f"{RELAY_GATE_TOLERANCE_PX:.0f}px of {nxt.name} "
                f"(x={_ROD_X[nxt]:.0f})"
            )
    return "\n".join(lines)


async def wait_for_puck_at_rod(machine, rod_x,
                               tol=RELAY_GATE_TOLERANCE_PX,
                               timeout=RELAY_GATE_TIMEOUT_S,
                               interval=RELAY_VISION_POLL_INTERVAL_S) -> bool:
    """Poll vision until the puck's x is within `tol` of `rod_x`.

    Returns True once the puck arrives, or False if `timeout` seconds elapse
    first. `machine` is an open RobotClient connection.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        puck_x, _ = await detect_puck(machine, RELAY_CAMERA)
        if puck_x is not None and puck_reached_rod(puck_x, rod_x, tol):
            return True
        await asyncio.sleep(interval)
    return False


async def home_all(components: dict) -> None:
    """Return every rod in `components` to home pose (t=0, r=0) concurrently."""
    print("Homing all rods.")
    await asyncio.gather(*[c.do_command({"t": 0, "r": 0}) for c in components.values()])


async def run_relay(machine) -> None:
    """Run the full five-rod relay on an open RobotClient connection.

    Detects the puck, then walks each leg: position the receiving rod to the
    puck's y, fire the pass, and (except on the last leg) wait for vision to
    confirm the puck reached the next rod. All rods are homed on exit, whether
    the relay finishes or aborts.
    """
    components = {
        pid: Generic.from_robot(robot=machine, name=_PLAYER_TO_COMPONENT[pid])
        for pid in _ROD_X
    }
    try:
        puck_x, puck_y = await detect_puck(machine, RELAY_CAMERA)
        if puck_x is None:
            print("No puck detected — aborting relay.")
            return

        first = RELAY[0]["player"]
        if not puck_reached_rod(puck_x, _ROD_X[first], RELAY_GATE_TOLERANCE_PX):
            print(f"Puck not on {first.name}'s rod (puck_x={puck_x:.0f}, "
                  f"expected ~{_ROD_X[first]:.0f}) — place the puck there to start.")
            return

        last = len(RELAY) - 1
        for i, leg in enumerate(RELAY):
            player = leg["player"]
            comp = components[player]

            # Legs after the first: re-detect the puck the gate just confirmed.
            if i > 0:
                puck_x, puck_y = await detect_puck(machine, RELAY_CAMERA)
                if puck_x is None:
                    print(f"Lost the puck before {player.name}'s leg — aborting.")
                    return

            t = puck_y_to_t(player, puck_y)
            print(f"Leg {i + 1}: {player.name} — receive t={t:.2f}, r={leg['receive_r']}")
            await comp.do_command({"t": t, "r": leg["receive_r"]})
            await comp.do_command(leg["pass_step"])

            if i != last:
                nxt = RELAY[i + 1]["player"]
                print(f"  waiting for puck to reach {nxt.name}...")
                arrived = await wait_for_puck_at_rod(machine, _ROD_X[nxt])
                if not arrived:
                    print(f"Puck never reached {nxt.name} — aborting relay.")
                    return

        print("Relay complete.")
    finally:
        await home_all(components)
