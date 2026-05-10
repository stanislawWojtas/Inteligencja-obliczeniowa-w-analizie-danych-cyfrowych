from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from crossy_road import CrossyRoadEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/dqn_crossy_road.zip"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval_summary.json"))
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default=None)
    parser.add_argument("--render-baseline", action="store_true")
    parser.add_argument("--max-steps", type=int, default=600)
    return parser.parse_args()


def evaluate(
    model: DQN, episodes: int, seed: int, render_mode: str | None = None, max_steps: int = 600
) -> dict:
    env = CrossyRoadEnv(render_mode=render_mode, config={"max_steps": max_steps})
    wins = 0
    collisions = 0
    rewards = []
    steps = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            done = terminated or truncated

        if info.get("finished", False):
            wins += 1
        if info.get("collision", False):
            collisions += 1
        rewards.append(total_reward)
        steps.append(int(info.get("steps", 0)))

    env.close()

    return {
        "episodes": episodes,
        "win_rate": wins / max(1, episodes),
        "collision_rate": collisions / max(1, episodes),
        "mean_reward": float(np.mean(rewards) if rewards else 0.0),
        "mean_steps": float(np.mean(steps) if steps else 0.0),
        "survival_time": float(np.mean(steps) if steps else 0.0),
    }


def random_baseline(episodes: int, seed: int, render_mode: str | None = None, max_steps: int = 600) -> dict:
    env = CrossyRoadEnv(render_mode=render_mode, config={"max_steps": max_steps})
    wins = 0
    collisions = 0
    steps = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + 10_000 + ep)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info.get("finished", False):
            wins += 1
        if info.get("collision", False):
            collisions += 1
        steps.append(int(info.get("steps", 0)))

    env.close()
    return {
        "episodes": episodes,
        "win_rate": wins / max(1, episodes),
        "collision_rate": collisions / max(1, episodes),
        "mean_steps": float(np.mean(steps) if steps else 0.0),
    }


def main() -> None:
    args = parse_args()
    model = DQN.load(str(args.model))

    model_metrics = evaluate(
        model, episodes=args.episodes, seed=args.seed, render_mode=args.render_mode, max_steps=args.max_steps
    )
    baseline_render_mode = args.render_mode if args.render_baseline else None
    baseline_metrics = random_baseline(
        episodes=args.episodes,
        seed=args.seed,
        render_mode=baseline_render_mode,
        max_steps=args.max_steps,
    )

    payload = {
        "model": model_metrics,
        "random_baseline": baseline_metrics,
        "delta_win_rate": model_metrics["win_rate"] - baseline_metrics["win_rate"],
        "delta_survival": model_metrics["survival_time"] - baseline_metrics["mean_steps"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
