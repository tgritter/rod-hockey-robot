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
