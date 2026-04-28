from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from crossy_road import CrossyRoadEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    env = Monitor(CrossyRoadEnv())
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.25,
        exploration_final_eps=0.05,
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    model_path = args.outdir / "dqn_crossy_road.zip"
    model.save(model_path)

    monitor_csv = args.outdir / "monitor.csv"
    with monitor_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length"])
        for idx, (r, l) in enumerate(zip(env.episode_returns, env.episode_lengths), start=1):
            writer.writerow([idx, float(r), int(l)])

    summary = {
        "timesteps": args.timesteps,
        "episodes": len(env.episode_returns),
        "mean_reward_last_20": float(sum(env.episode_returns[-20:]) / max(1, len(env.episode_returns[-20:]))),
        "model_path": str(model_path),
        "monitor_csv": str(monitor_csv),
    }
    (args.outdir / "train_summary.json").write_text(json.dumps(summary, indent=2))

    env.close()


if __name__ == "__main__":
    main()
