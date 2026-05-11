from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from crossy_road import ACTION_UP, ACTION_WAIT, ActionSubsetWrapper, CrossyRoadEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"))
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    env_config = {"max_steps": args.max_steps}
    train_action_map = [ACTION_UP, ACTION_WAIT]
    env = Monitor(
        ActionSubsetWrapper(
            CrossyRoadEnv(render_mode=args.render_mode, config=env_config),
            actions=train_action_map,
        )
    )
    eval_env = Monitor(
        ActionSubsetWrapper(
            CrossyRoadEnv(config=env_config),
            actions=train_action_map,
        )
    )
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.4,
        exploration_final_eps=0.02,
        verbose=1,
        seed=args.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.outdir),
        log_path=str(args.outdir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        warn=False,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=eval_callback)

    model_path = args.outdir / "dqn_crossy_road.zip"
    best_model_path = args.outdir / "best_model.zip"
    if best_model_path.exists():
        model = DQN.load(str(best_model_path))
    model.save(model_path)

    monitor_csv = args.outdir / "monitor.csv"
    with monitor_csv.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["episode", "reward", "length"])
        for idx, (r, l) in enumerate(zip(env.episode_returns, env.episode_lengths), start=1):
            writer.writerow([idx, float(r), int(l)])

    summary = {
        "timesteps": args.timesteps,
        "episodes": len(env.episode_returns),
        "mean_reward_last_20": float(sum(env.episode_returns[-20:]) / max(1, len(env.episode_returns[-20:]))),
        "env_config": env_config,
        "training_actions": train_action_map,
        "eval_config": {
            "eval_freq": args.eval_freq,
            "eval_episodes": args.eval_episodes,
            "best_mean_reward": float(eval_callback.best_mean_reward),
            "best_model_path": str(best_model_path),
        },
        "dqn_config": {
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "learning_starts": 5_000,
            "batch_size": 64,
            "gamma": 0.99,
            "train_freq": 4,
            "target_update_interval": 500,
            "exploration_fraction": 0.4,
            "exploration_final_eps": 0.02,
        },
        "model_path": str(model_path),
        "monitor_csv": str(monitor_csv),
    }
    (args.outdir / "train_summary.json").write_text(json.dumps(summary, indent=2))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
