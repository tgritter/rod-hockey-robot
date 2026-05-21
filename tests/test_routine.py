"""Unit tests for the relay routine's pure functions."""

from engine.constants import (
    min_y_center, max_y_center,
    min_y_left_d, max_y_left_d,
)
from engine.constants import PlayerID
from robot.routine import puck_y_to_t


def test_puck_y_to_t_endpoints():
    assert puck_y_to_t(PlayerID.CENTER, min_y_center) == 0.0
    assert puck_y_to_t(PlayerID.CENTER, max_y_center) == 1.0


def test_puck_y_to_t_midpoint():
    mid = (min_y_center + max_y_center) / 2
    assert puck_y_to_t(PlayerID.CENTER, mid) == 0.5


def test_puck_y_to_t_clamps_below_min():
    assert puck_y_to_t(PlayerID.CENTER, min_y_center - 100) == 0.0


def test_puck_y_to_t_clamps_above_max():
    assert puck_y_to_t(PlayerID.CENTER, max_y_center + 100) == 1.0


def test_puck_y_to_t_uses_per_rod_band():
    # Left D has a different band than Center
    assert puck_y_to_t(PlayerID.LEFT_D, min_y_left_d) == 0.0
    assert puck_y_to_t(PlayerID.LEFT_D, max_y_left_d) == 1.0


from robot.routine import puck_reached_rod


def test_puck_reached_rod_inside_tolerance():
    assert puck_reached_rod(puck_x=205.0, rod_x=200.0, tol=30.0) is True


def test_puck_reached_rod_outside_tolerance():
    assert puck_reached_rod(puck_x=260.0, rod_x=200.0, tol=30.0) is False


def test_puck_reached_rod_exactly_at_tolerance():
    assert puck_reached_rod(puck_x=230.0, rod_x=200.0, tol=30.0) is True


from robot.routine import format_relay_plan


def test_format_relay_plan_lists_all_five_legs():
    text = format_relay_plan()
    assert "Leg 1:" in text
    assert "Leg 5:" in text
    assert "LEFT_D" in text
    assert "RIGHT_WING" in text


def test_format_relay_plan_marks_the_shot():
    text = format_relay_plan()
    assert "SHOT" in text


def test_format_relay_plan_describes_gates():
    text = format_relay_plan()
    # Four gates (one between each pair of legs), none after the last leg.
    assert text.count("gate:") == 4
