from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from crossy_road.reporting import (
    load_monitor_history,
    plot_learning_curve,
    rolling_mean,
)


def test_rolling_mean_handles_short_prefix():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])

    smoothed = rolling_mean(values, window=3)

    assert np.allclose(smoothed, np.asarray([1.0, 1.5, 2.0, 3.0]))


def test_loaders_and_plot_generate_learning_curve(tmp_path: Path):
    monitor_csv = tmp_path / "monitor.csv"
    with monitor_csv.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["episode", "reward", "length"])
        writer.writerow([1, -1.0, 10])
        writer.writerow([2, 0.5, 12])
        writer.writerow([3, 2.0, 8])

    evaluations_npz = tmp_path / "evaluations.npz"
    np.savez(
        evaluations_npz,
        timesteps=np.asarray([100, 200, 300]),
        results=np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 3.0]]),
        ep_lengths=np.asarray([[10, 11], [12, 13], [14, 15]]),
    )

    episodes, rewards = load_monitor_history(monitor_csv)
    assert episodes.tolist() == [1, 2, 3]
    assert rewards.tolist() == [-1.0, 0.5, 2.0]

    output_path = plot_learning_curve(monitor_csv, evaluations_npz, tmp_path / "learning_curve.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
