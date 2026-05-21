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
