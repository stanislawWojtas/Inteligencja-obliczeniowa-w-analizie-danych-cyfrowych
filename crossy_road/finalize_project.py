import time
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from crossy_road.env import CrossyRoadEnv, ALL_ACTIONS
from crossy_road.wrappers import ActionSubsetWrapper

def main():
    experiments_dir = Path("artifacts/experiments")
    
    print("--- 1. SZUKANIE NAJLEPSZEGO AGENTA ---")
    best_mean_reward = -float("inf")
    best_model_dir = None
    
    if not experiments_dir.exists():
        print("Brak katalogu artifacts/experiments.")
        return

    for monitor_file in experiments_dir.rglob("monitor.csv"):
        try:
            df = pd.read_csv(monitor_file, skiprows=1)
            if len(df) == 0: continue
            
            last_100_mean = df['r'].tail(100).mean() 
            if last_100_mean > best_mean_reward:
                best_mean_reward = last_100_mean
                best_model_dir = monitor_file.parent
        except Exception:
            continue
            
    if best_model_dir is None:
        print("Nie znaleziono logów treningowych.")
        return
        
    print(f"Najlepszy model: {best_model_dir}")
    print(f"Średnia nagroda (końcówka uczenia): {best_mean_reward:.2f}\n")

    print("--- 2. PARAMETRY ŚRODOWISKA I SIECI ---")
    env = CrossyRoadEnv(config={"max_steps": 500})
    env = ActionSubsetWrapper(env, actions=list(ALL_ACTIONS))
    
    print(f"Wielkość wejścia (stan): {env.observation_space.shape[0]}")
    print(f"Wielkość wyjścia (akcje): {env.action_space.n}")
    
    print("\nMierzenie czasu środowiska (10 000 kroków)...")
    env.reset()
    start_time = time.perf_counter()
    for _ in range(10000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    end_time = time.perf_counter()
    
    time_per_step_ms = ((end_time - start_time) * 1000) / 10000
    time_per_episode_ms = time_per_step_ms * 500
    
    print(f"Czas jednego kroku: {time_per_step_ms:.3f} ms")
    print(f"Estymowany czas epizodu (500 kroków): {time_per_episode_ms:.2f} ms\n")

    print("--- 3. SYMULACJA DETERMINISTYCZNA ---")
    model_path = best_model_dir / "model.zip"
    if not model_path.exists():
        print(f"Brak pliku model.zip w {best_model_dir}")
        return
        
    model = PPO.load(str(model_path))
    eval_episodes = 20
    deterministic_rewards = []
    
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        deterministic_rewards.append(ep_reward)
        
    mean_det = np.mean(deterministic_rewards)
    std_det = np.std(deterministic_rewards)
    print(f"Wynik z wyłączoną eksploracją (20 ep): {mean_det:.2f} +/- {std_det:.2f}")

    env.close()

if __name__ == '__main__':
    main()