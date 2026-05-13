from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def load_monitor_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    episodes: list[int] = []
    rewards: list[float] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))

    return np.asarray(episodes, dtype=np.int64), np.asarray(rewards, dtype=np.float64)


def load_evaluation_history(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    timesteps = np.asarray(data["timesteps"], dtype=np.int64)
    results = np.asarray(data["results"], dtype=np.float64)
    mean_rewards = results.mean(axis=1) if results.size else np.asarray([], dtype=np.float64)
    std_rewards = results.std(axis=1) if results.size else np.asarray([], dtype=np.float64)
    return timesteps, mean_rewards, std_rewards


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window, dtype=np.float64) / window
    valid = np.convolve(values, kernel, mode="valid")
    prefix = np.cumsum(values[: window - 1], dtype=np.float64) / np.arange(1, window)
    return np.concatenate([prefix, valid])


def plot_learning_curve(
    monitor_csv: Path,
    evaluations_npz: Path | None,
    output_path: Path,
    smoothing_window: int = 25,
) -> Path:
    episodes, rewards = load_monitor_history(monitor_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(episodes, rewards, color="#9ecae1", alpha=0.45, linewidth=1.0, label="Episode reward")
    ax.plot(
        episodes,
        rolling_mean(rewards, smoothing_window),
        color="#08519c",
        linewidth=2.5,
        label=f"Moving average ({min(smoothing_window, max(1, rewards.size))})",
    )
    ax.set_title("Learning Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
