from crossy_road.env import CrossyRoadEnv, ALL_ACTIONS
from crossy_road.wrappers import ActionSubsetWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENTS = [
    {
        "name": "baseline_64x64",
        "params": {
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_steps": 2048,
            "gamma": 0.99,
            "policy_kwargs": {"net_arch": [64, 64]},
        },
    },
    {
        "name": "deep_128x128",
        "params": {
            "learning_rate": 1e-4,
            "batch_size": 128,
            "n_steps": 1024,
            "gamma": 0.98,
            "policy_kwargs": {"net_arch": [128, 128]},
        },
    },
    {
        "name": "fast_64x64",
        "params": {
            "learning_rate": 5e-4,
            "batch_size": 256,
            "n_steps": 512,
            "gamma": 0.995,
            "policy_kwargs": {"net_arch": [64, 64]},
        },
    },
]


def read_monitor_file(monitor_path: Path) -> tuple[np.ndarray, np.ndarray]:
    timesteps: list[float] = []
    rewards: list[float] = []
    cumulative = 0

    with monitor_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header and header[0].startswith("#"):
            header = next(reader, None)

        for row in reader:
            if not row:
                continue
            reward = float(row[0])
            length = int(float(row[1]))
            cumulative += length
            timesteps.append(float(cumulative))
            rewards.append(reward)

    return np.asarray(timesteps), np.asarray(rewards)


def plot_learning_curves(base_dir: Path) -> None:
    print("Aggregating learning curves")
    figure, axis = plt.subplots(figsize=(10, 6))

    for config_index, experiment in enumerate(EXPERIMENTS, start=1):
        seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for seed in range(10):
            monitor_path = (
                base_dir / f"config_{config_index}" / f"seed_{seed}" / "monitor.csv"
            )
            if not monitor_path.exists():
                print(f"  Missing monitor file: {monitor_path}")
                continue

            timesteps, rewards = read_monitor_file(monitor_path)
            if timesteps.size == 0:
                continue
            seed_curves.append((timesteps, rewards))

        if not seed_curves:
            print(f"  No data for config {config_index}")
            continue

        max_common = min(curve[0][-1] for curve in seed_curves)
        if max_common <= 0:
            continue

        grid = np.linspace(0, max_common, num=200)
        interpolated = [np.interp(grid, curve[0], curve[1]) for curve in seed_curves]
        reward_array = np.vstack(interpolated)
        mean_reward = reward_array.mean(axis=0)
        std_reward = reward_array.std(axis=0)

        label = experiment["name"]
        axis.plot(grid, mean_reward, label=label)
        axis.fill_between(grid, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)

    axis.set_xlabel("Timesteps")
    axis.set_ylabel("Episodic Reward")
    axis.legend()
    axis.grid(True, alpha=0.3)

    output_path = base_dir / "learning_curves.png"
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    print(f"Saved learning curves to {output_path}")


def evaluate_deterministic(model_path: Path, seed: int) -> float:
    eval_env = CrossyRoadEnv(config={"max_steps": 500})
    eval_env = ActionSubsetWrapper(eval_env, actions=list(ALL_ACTIONS))
    eval_model = PPO.load(model_path, env=eval_env)

    episode_rewards: list[float] = []
    for episode in range(10):
        obs, _ = eval_env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = eval_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        episode_rewards.append(total_reward)

    eval_env.close()
    return float(np.mean(episode_rewards))


def run_experiments() -> None:
    base_dir = Path("artifacts") / "experiments"

    for config_index, experiment in enumerate(EXPERIMENTS, start=1):
        config_name = experiment["name"]
        ppo_params = experiment["params"]
        print(f"Starting config {config_index}: {config_name}")

        for seed in range(10):
            print(f"  Seed {seed}: initializing environment")
            log_dir = base_dir / f"config_{config_index}" / f"seed_{seed}"
            log_dir.mkdir(parents=True, exist_ok=True)

            env = CrossyRoadEnv(config={"max_steps": 500})
            env = ActionSubsetWrapper(env, actions=list(ALL_ACTIONS))
            env = Monitor(env, filename=str(log_dir))

            print("  Building PPO model")
            model = PPO("MlpPolicy", env, seed=seed, **ppo_params)

            print("  Training for 50000 timesteps")
            model.learn(total_timesteps=50_000)

            print("  Saving model")
            model_path = log_dir / "model.zip"
            model.save(model_path)

            print("  Running deterministic evaluation")
            mean_reward = evaluate_deterministic(model_path, seed)
            summary_path = log_dir / "eval_summary.json"
            summary = {
                "config_index": config_index,
                "config_name": config_name,
                "seed": seed,
                "episodes": 10,
                "mean_deterministic_reward": mean_reward,
            }
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)

            env.close()
            print(f"  Done seed {seed}")

        print(f"Finished config {config_index}: {config_name}")

    plot_learning_curves(base_dir)


if __name__ == "__main__":
    run_experiments()
