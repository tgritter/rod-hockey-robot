"""Unit tests for the autonomous puck-control model."""

import os

from robot.puck_model import Sample, dataset_path, load_dataset, append_sample


def test_dataset_path_uses_data_dir_and_player():
    assert dataset_path("left-defense") == os.path.join("data", "left-defense.jsonl")


def test_load_dataset_missing_file_is_empty():
    assert load_dataset("data/does-not-exist-xyz.jsonl") == []


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


import numpy as np

from robot.puck_model import PuckModel


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
    model = PuckModel(_linear_samples(j))
    dx, dy = model.predict((0.5, 180.0, 270.0, 150.0), d_t=0.05, d_r=10.0)
    expected = np.array(j) @ np.array([0.05, 10.0])
    assert abs(dx - expected[0]) < 1e-3
    assert abs(dy - expected[1]) < 1e-3


def test_predict_with_empty_model_is_zero():
    assert PuckModel([]).predict((0.5, 180.0, 270.0, 150.0), 0.05, 10.0) == (0.0, 0.0)


def test_confidence_is_one_when_samples_sit_at_the_query():
    state = (0.5, 180.0, 270.0, 150.0)
    samples = [Sample(*state, 0.05, 10.0, 271.0, 151.0) for _ in range(10)]
    assert PuckModel(samples).confidence(state) == 1.0


def test_confidence_is_near_zero_far_from_the_data():
    state = (0.5, 180.0, 270.0, 150.0)
    samples = [Sample(*state, 0.05, 10.0, 271.0, 151.0) for _ in range(10)]
    far = (0.95, 350.0, 520.0, 290.0)
    assert PuckModel(samples).confidence(far) < 0.1


def test_confidence_with_empty_model_is_zero():
    assert PuckModel([]).confidence((0.5, 180.0, 270.0, 150.0)) == 0.0


def test_solve_inverts_the_jacobian():
    j = [[2.0, 0.1], [0.3, -1.5]]
    model = PuckModel(_linear_samples(j))
    state = (0.5, 180.0, 270.0, 150.0)
    d_t, d_r, conf = model.solve(state, d_puck_desired=(0.2, -0.3))
    # Small desired step -> solved move stays within clamps -> reproduces it.
    dx, dy = model.predict(state, d_t, d_r)
    assert abs(dx - 0.2) < 1e-2
    assert abs(dy - (-0.3)) < 1e-2
    assert 0.0 <= conf <= 1.0


def test_solve_clamps_the_move():
    j = [[0.05, 0.0], [0.0, 0.05]]
    model = PuckModel(_linear_samples(j))
    d_t, d_r, _ = model.solve((0.5, 180.0, 270.0, 150.0),
                              d_puck_desired=(500.0, 500.0))
    assert abs(d_t) <= 0.10 + 1e-9
    assert abs(d_r) <= 30.0 + 1e-9


def test_solve_with_empty_model_is_zero():
    assert PuckModel([]).solve((0.5, 180.0, 270.0, 150.0), (10.0, 10.0)) == (0.0, 0.0, 0.0)
