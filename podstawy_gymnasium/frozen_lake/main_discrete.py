import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

    states_num = env.observation_space.n
    actions_num = env.action_space.n
    q_table = np.zeros((states_num, actions_num))

    alpha = 0.8
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    episodes = 2000

    rewards_all_episodes = []

    print("Rozpoczynam trenowanie agenta...")
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]
            )

            state = new_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_all_episodes.append(total_reward)

    env.close()
    print("Trening zakończony!\n")

    window = 100
    smoothed_rewards = [
        np.mean(rewards_all_episodes[max(0, i - window) : (i + 1)])
        for i in range(len(rewards_all_episodes))
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards)
    plt.title("Krzywa uczenia - Frozen Lake (Q-Learning)")
    plt.xlabel("Epizod")
    plt.ylabel(f"Średnia nagroda (okno={window})")
    plt.grid(True)
    plt.savefig("krzywa_uczenia.png")
    print("Wykres 'krzywa_uczenia.png' został zapisany.")

    print("\nOto jak agent radzi sobie po nauce:")

    screenshots_dir = "screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    env_eval = gym.make(
        "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array"
    )
    state, info = env_eval.reset()
    frame = env_eval.render()
    plt.imsave(os.path.join(screenshots_dir, "frozenlake_start.png"), frame)
    done = False

    while not done:
        action = np.argmax(q_table[state, :])
        state, reward, terminated, truncated, info = env_eval.step(action)
        done = terminated or truncated

    frame = env_eval.render()
    plt.imsave(os.path.join(screenshots_dir, "frozenlake_end.png"), frame)

    env_eval.close()
    print(f"Prezentacja zakończona! Zrzuty zapisane w katalogu: {screenshots_dir}")


if __name__ == "__main__":
    main()