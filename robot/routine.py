"""Contact-based visual-servo relay routine.

Relays the puck through the rods and finishes with a shot. Each rod catches the
puck by sweeping its translation until the puck moves (contact detected by
vision-1), then swings to pass it toward the next rod. The puck is the only
sensor — no player detection, no table geometry. See
docs/superpowers/specs/2026-05-21-visual-servo-relay-design-v2.md.
"""

import asyncio

from viam.components.generic import Generic

from robot.const import RELAY_SWEEP_STEP_T, RELAY_CONTACT_MOVE_PX
from robot.execution import _PLAYER_TO_COMPONENT
from robot.playbook import RELAY
from robot.vision import detect_puck_px


def _dist(a, b):
    """Euclidean distance between two (x, y) points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def format_relay_plan() -> str:
    """Return a human-readable description of the relay plan (no hardware)."""
    lines = ["Relay plan (contact-based catch, then pass):"]
    last = len(RELAY) - 1
    for i, leg in enumerate(RELAY):
        player = leg["player"]
        action = "SHOT" if i == last else "pass"
        lines.append(
            f"  Leg {i + 1}: {player.name:11s} "
            f"catch r={leg['receive_r']} (sweep t until contact), "
            f"{action} {leg['pass_step']}"
        )
    return "\n".join(lines)


async def sweep_for_contact(machine, component, catch_r, step=RELAY_SWEEP_STEP_T):
    """Sweep a rod's translation until the puck moves (contact).

    Parks the rod at t=0 with the catch rotation, records the puck's position,
    then steps t up to 1.0. When the puck's detected position moves more than
    RELAY_CONTACT_MOVE_PX from its pre-sweep position, the player has touched
    it. Returns (contacted: bool, puck_pos: (x, y) | None).
    """
    await component.do_command({"t": 0.0, "r": catch_r})
    p0 = await detect_puck_px(machine)
    if p0 is None:
        return False, None
    t = 0.0
    while t < 1.0 - 1e-9:
        t = min(1.0, t + step)
        await component.do_command({"t": t})
        p = await detect_puck_px(machine)
        if p is not None and _dist(p, p0) > RELAY_CONTACT_MOVE_PX:
            return True, p
    return False, p0


async def home_all(components):
    """Return every rod in `components` to home pose (t=0, r=0) concurrently."""
    print("Homing all rods.")
    await asyncio.gather(*[c.do_command({"t": 0, "r": 0}) for c in components.values()])


async def run_relay(machine):
    """Run the contact-based relay on an open RobotClient connection.

    Sweeps rods in leg order; the first to contact the puck starts the relay.
    Each caught rod passes (or, on the last leg, shoots), and the next rod must
    then catch. Once the relay has started, a rod that cannot catch means the
    previous pass missed — abort. All rods are homed on exit, success or not.
    """
    components = {
        leg["player"]: Generic.from_robot(
            robot=machine, name=_PLAYER_TO_COMPONENT[leg["player"]]
        )
        for leg in RELAY
    }
    try:
        started = False
        last = len(RELAY) - 1
        for i, leg in enumerate(RELAY):
            player = leg["player"]
            comp = components[player]
            print(f"Leg {i + 1}: {player.name} — sweeping for the puck...")
            contacted, pos = await sweep_for_contact(machine, comp, leg["receive_r"])
            if not contacted:
                if started:
                    print(f"  {player.name} could not catch the puck — "
                          f"previous pass missed, aborting relay.")
                    return
                print(f"  puck not on {player.name}, trying the next rod.")
                continue
            started = True
            is_shot = i == last
            print(f"  {player.name} caught the puck at {pos} — "
                  f"{'shooting' if is_shot else 'passing'}.")
            await comp.do_command(leg["pass_step"])
        if started:
            print("Relay complete.")
        else:
            print("No rod ever found the puck — is it on the table?")
    finally:
        await home_all(components)
