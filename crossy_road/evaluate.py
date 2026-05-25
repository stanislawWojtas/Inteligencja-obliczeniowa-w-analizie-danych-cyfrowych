from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from crossy_road import ACTION_UP, ACTION_WAIT, ALL_ACTIONS, ActionSubsetWrapper, CrossyRoadEnv
from crossy_road.env import ACTION_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/dqn_crossy_road.zip"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval_summary.json"))
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default=None)
    parser.add_argument("--render-baseline", action="store_true")
    parser.add_argument("--max-steps", type=int, default=500)
    return parser.parse_args()


def _action_metrics(action_counts: Counter[int]) -> dict:
    total = sum(action_counts.values())
    counts_by_name = {
        ACTION_NAMES[action]: int(action_counts.get(action, 0))
        for action in sorted(ACTION_NAMES)
    }
    rates_by_name = {
        name: (count / total if total else 0.0)
        for name, count in counts_by_name.items()
    }
    return {
        "action_counts": counts_by_name,
        "action_rates": rates_by_name,
        "wait_rate": rates_by_name[ACTION_NAMES[ACTION_WAIT]],
    }


def _risk_metrics(risk_counts: Counter[str]) -> dict:
    forward_risk_steps = risk_counts["forward_risk_steps"]
    wait_risk_steps = risk_counts["wait_risk_steps"]
    return {
        "forward_risk_steps": int(forward_risk_steps),
        "wait_risk_steps": int(wait_risk_steps),
        "wait_when_forward_risky_rate": (
            risk_counts["wait_when_forward_risky"] / forward_risk_steps
            if forward_risk_steps
            else 0.0
        ),
        "wait_when_wait_risky_rate": (
            risk_counts["wait_when_wait_risky"] / wait_risk_steps
            if wait_risk_steps
            else 0.0
        ),
    }


def evaluate(
    model: DQN, episodes: int, seed: int, render_mode: str | None = None, max_steps: int = 500
) -> dict:
    env = ActionSubsetWrapper(
        CrossyRoadEnv(render_mode=render_mode, config={"max_steps": max_steps}),
        actions=ALL_ACTIONS,
    )
    wins = 0
    collisions = 0
    truncations = 0
    rewards = []
    steps = []
    action_counts: Counter[int] = Counter()
    risk_counts: Counter[str] = Counter()

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0

        while not done:
            base_env = env.unwrapped
            forward_risky = base_env._forward_path_has_projected_traffic(ACTION_UP)
            wait_risky = base_env._position_has_projected_collision(base_env.player_x, base_env.player_y, 1)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            mapped_action = int(info.get("mapped_action", action))
            action_counts[mapped_action] += 1
            if forward_risky:
                risk_counts["forward_risk_steps"] += 1
                if mapped_action == ACTION_WAIT:
                    risk_counts["wait_when_forward_risky"] += 1
            if wait_risky:
                risk_counts["wait_risk_steps"] += 1
                if mapped_action == ACTION_WAIT:
                    risk_counts["wait_when_wait_risky"] += 1
            total_reward += float(reward)
            done = terminated or truncated

        if info.get("finished", False):
            wins += 1
        if info.get("collision", False):
            collisions += 1
        if truncated:
            truncations += 1
        rewards.append(total_reward)
        steps.append(int(info.get("steps", 0)))

    env.close()

    return {
        "episodes": episodes,
        "win_rate": wins / max(1, episodes),
        "collision_rate": collisions / max(1, episodes),
        "truncation_rate": truncations / max(1, episodes),
        "mean_reward": float(np.mean(rewards) if rewards else 0.0),
        "mean_steps": float(np.mean(steps) if steps else 0.0),
        "survival_time": float(np.mean(steps) if steps else 0.0),
        **_action_metrics(action_counts),
        **_risk_metrics(risk_counts),
    }


def random_baseline(episodes: int, seed: int, render_mode: str | None = None, max_steps: int = 500) -> dict:
    env = ActionSubsetWrapper(
        CrossyRoadEnv(render_mode=render_mode, config={"max_steps": max_steps}),
        actions=ALL_ACTIONS,
    )
    wins = 0
    collisions = 0
    truncations = 0
    steps = []
    action_counts: Counter[int] = Counter()
    risk_counts: Counter[str] = Counter()

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + 10_000 + ep)
        done = False
        while not done:
            base_env = env.unwrapped
            forward_risky = base_env._forward_path_has_projected_traffic(ACTION_UP)
            wait_risky = base_env._position_has_projected_collision(base_env.player_x, base_env.player_y, 1)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            mapped_action = int(info.get("mapped_action", action))
            action_counts[mapped_action] += 1
            if forward_risky:
                risk_counts["forward_risk_steps"] += 1
                if mapped_action == ACTION_WAIT:
                    risk_counts["wait_when_forward_risky"] += 1
            if wait_risky:
                risk_counts["wait_risk_steps"] += 1
                if mapped_action == ACTION_WAIT:
                    risk_counts["wait_when_wait_risky"] += 1
            done = terminated or truncated

        if info.get("finished", False):
            wins += 1
        if info.get("collision", False):
            collisions += 1
        if truncated:
            truncations += 1
        steps.append(int(info.get("steps", 0)))

    env.close()
    return {
        "episodes": episodes,
        "win_rate": wins / max(1, episodes),
        "collision_rate": collisions / max(1, episodes),
        "truncation_rate": truncations / max(1, episodes),
        "mean_steps": float(np.mean(steps) if steps else 0.0),
        **_action_metrics(action_counts),
        **_risk_metrics(risk_counts),
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
