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
