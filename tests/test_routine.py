"""Unit tests for the relay routine's pure functions."""

from robot.routine import _dist, format_relay_plan


def test_dist_identical_points_is_zero():
    assert _dist((10, 20), (10, 20)) == 0.0


def test_dist_pythagorean():
    assert _dist((0, 0), (3, 4)) == 5.0


def test_dist_is_symmetric():
    assert _dist((1, 2), (4, 6)) == _dist((4, 6), (1, 2))


def test_format_relay_plan_lists_all_five_legs():
    text = format_relay_plan()
    assert "Leg 1:" in text
    assert "Leg 5:" in text
    assert "LEFT_D" in text
    assert "RIGHT_WING" in text


def test_format_relay_plan_marks_the_shot():
    assert "SHOT" in format_relay_plan()
