# tests/test_select_playbook.py
# Uses the seeded robot/zones.json from Task 3.
from engine.constants import PlayerID
from robot.playbook import select_playbook, RIGHT_WING_LEFT, CENTER_LEFT


def test_right_wing_left_zone():
    # Legacy right_wing/left box was px x[155,315] y[25,75] on a 538x284 crop.
    # Center of that box -> normalized ~ (0.437, 0.176).
    player, seq = select_playbook(235.0 / 538, 50.0 / 284)
    assert player == PlayerID.RIGHT_WING
    # RW sequence starts with the position block for that side.
    assert seq[0] == RIGHT_WING_LEFT[0]


def test_center_left_zone():
    # Legacy center/left box px x[150,300] y[135,185] -> center normalized.
    player, seq = select_playbook(225.0 / 538, 160.0 / 284)
    assert player == PlayerID.CENTER
    assert seq is CENTER_LEFT


def test_no_zone_returns_none():
    player, seq = select_playbook(0.99, 0.99)
    assert (player, seq) == (None, None)
