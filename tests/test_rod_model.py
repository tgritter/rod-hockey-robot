"""Unit tests for the one-rod puck-control model."""

import os

from robot.rod_model import Sample, dataset_path, load_dataset, append_sample


def test_dataset_path_uses_data_dir_and_rod_name():
    assert dataset_path("left-defense") == os.path.join("data", "rod_left-defense.jsonl")


def test_append_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "ds.jsonl")
    append_sample(path, t=0.2, r=30.0, puck=(100.0, 200.0),
                  dt=0.05, dr=10.0, puck2=(108.0, 205.0))
    append_sample(path, t=0.3, r=40.0, puck=(108.0, 205.0),
                  dt=-0.05, dr=-10.0, puck2=(101.0, 199.0))
    samples = load_dataset(path)
    assert len(samples) == 2
    assert samples[0] == Sample(0.2, 30.0, 100.0, 200.0, 0.05, 10.0, 108.0, 205.0)
    assert samples[1].puck_x2 == 101.0


def test_load_dataset_skips_blank_lines(tmp_path):
    path = str(tmp_path / "ds.jsonl")
    append_sample(path, t=0.0, r=0.0, puck=(1.0, 2.0), dt=0.0, dr=0.0, puck2=(1.0, 2.0))
    with open(path, "a") as f:
        f.write("\n")
    assert len(load_dataset(path)) == 1


import math

from robot.rod_model import puck_step_toward


def test_puck_step_toward_within_range_returns_full_vector():
    assert puck_step_toward((100.0, 100.0), (110.0, 100.0), max_step=25.0) == (10.0, 0.0)


def test_puck_step_toward_clamps_to_max_step():
    step = puck_step_toward((0.0, 0.0), (300.0, 400.0), max_step=25.0)
    assert math.isclose(math.hypot(*step), 25.0, rel_tol=1e-9)


def test_puck_step_toward_at_target_is_zero():
    assert puck_step_toward((50.0, 50.0), (50.0, 50.0), max_step=25.0) == (0.0, 0.0)


import numpy as np

from robot.rod_model import RodModel


def _linear_samples(jacobian, n=40, seed=0):
    """Build n Samples whose puck response is exactly jacobian @ (dt, dr)."""
    rng = np.random.default_rng(seed)
    j = np.array(jacobian, float)
    out = []
    for _ in range(n):
        t = rng.uniform(0, 1)
        r = rng.uniform(0, 360)
        px = rng.uniform(0, 540)
        py = rng.uniform(0, 300)
        dt = rng.uniform(-0.1, 0.1)
        dr = rng.uniform(-30, 30)
        dp = j @ np.array([dt, dr])
        out.append(Sample(t, r, px, py, dt, dr, px + dp[0], py + dp[1]))
    return out


def test_predict_recovers_a_global_linear_jacobian():
    j = [[2.0, 0.1], [0.3, -1.5]]
    model = RodModel(_linear_samples(j))
    dx, dy = model.predict(0.5, 180.0, (270.0, 150.0), d_t=0.05, d_r=10.0)
    expected = np.array(j) @ np.array([0.05, 10.0])
    assert abs(dx - expected[0]) < 1e-3
    assert abs(dy - expected[1]) < 1e-3


def test_rodmodel_rejects_empty_dataset():
    try:
        RodModel([])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_solve_inverts_the_jacobian():
    j = [[2.0, 0.1], [0.3, -1.5]]
    model = RodModel(_linear_samples(j))
    d_t, d_r, controllable = model.solve(0.5, 180.0, (270.0, 150.0),
                                         d_puck_desired=(0.2, -0.3))
    # The desired displacement is small enough that the solved move stays
    # within the clamps, so applying it reproduces the desired puck step.
    dx, dy = model.predict(0.5, 180.0, (270.0, 150.0), d_t, d_r)
    assert abs(dx - 0.2) < 1e-2
    assert abs(dy - (-0.3)) < 1e-2
    assert controllable is True


def test_solve_clamps_the_move():
    j = [[0.05, 0.0], [0.0, 0.05]]   # tiny gains -> huge move requested
    model = RodModel(_linear_samples(j))
    d_t, d_r, _ = model.solve(0.5, 180.0, (270.0, 150.0),
                              d_puck_desired=(500.0, 500.0))
    assert abs(d_t) <= 0.12 + 1e-9
    assert abs(d_r) <= 45.0 + 1e-9


def test_solve_reports_not_controllable_when_puck_never_moves():
    # Samples where the puck barely responds to any move.
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(40):
        t = rng.uniform(0, 1); r = rng.uniform(0, 360)
        px = rng.uniform(0, 540); py = rng.uniform(0, 300)
        dt = rng.uniform(-0.1, 0.1); dr = rng.uniform(-30, 30)
        samples.append(Sample(t, r, px, py, dt, dr, px, py))   # no puck motion
    model = RodModel(samples)
    _, _, controllable = model.solve(0.5, 180.0, (270.0, 150.0),
                                     d_puck_desired=(10.0, 10.0))
    assert controllable is False
