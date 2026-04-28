import csv
import os
import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DiscretizedQLearning:
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.bins = [
            np.linspace(-4.8, 4.8, 6),
            np.linspace(-4, 4, 6),
            np.linspace(-0.418, 0.418, 6),
            np.linspace(-4, 4, 6),
        ]
        self.q_table = np.zeros(([7] * 4 + [env.action_space.n]))

    def discretize(self, state):
        discrete_state = []
        for i in range(len(state)):
            discrete_state.append(np.digitize(state[i], self.bins[i]))
        return tuple(discrete_state)

    def train(self, episodes=1000, track_discounted=False):
        rewards_history = []
        discounted_history = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            state_adj = self.discretize(state)
            done = False
            total_reward = 0
            discounted_return = 0.0
            discount = 1.0

            while not done:
                if np.random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state_adj])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state_adj = self.discretize(next_state)

                best_next_action = np.argmax(self.q_table[next_state_adj])
                td_target = reward + self.gamma * self.q_table[next_state_adj][
                    best_next_action
                ]
                self.q_table[state_adj][action] += self.alpha * (
                    td_target - self.q_table[state_adj][action]
                )

                state_adj = next_state_adj
                total_reward += reward
                if track_discounted:
                    discounted_return += discount * reward
                    discount *= self.gamma

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            if track_discounted:
                discounted_history.append(discounted_return)

        if track_discounted:
            return rewards_history, discounted_history
        return rewards_history


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(
        self,
        env,
        lr=0.001,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 64

        self.memory = deque(maxlen=10000)
        self.model = DQNNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train(self, episodes=1000, track_discounted=False):
        rewards_history = []
        discounted_history = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            discounted_return = 0.0
            discount = 1.0

            while not done:
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                    action = torch.argmax(q_values).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                if track_discounted:
                    discounted_return += discount * reward
                    discount *= self.gamma

                if len(self.memory) > self.batch_size:
                    batch = random.sample(self.memory, self.batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states_t = torch.FloatTensor(np.array(states))
                    actions_t = torch.LongTensor(actions).unsqueeze(1)
                    rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states_t = torch.FloatTensor(np.array(next_states))
                    dones_t = torch.FloatTensor(dones).unsqueeze(1)

                    curr_q = self.model(states_t).gather(1, actions_t)
                    next_q = self.model(next_states_t).max(1)[0].unsqueeze(1)
                    target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

                    loss = self.loss_fn(curr_q, target_q.detach())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            if track_discounted:
                discounted_history.append(discounted_return)
            print(f"DQN Epizod {episode + 1}/{episodes} - Nagroda: {total_reward}")

        if track_discounted:
            return rewards_history, discounted_history
        return rewards_history


def smooth(data, window=50):
    """Wygładzanie wykresów za pomocą średniej kroczącej"""
    return [np.mean(data[max(0, i - window) : (i + 1)]) for i in range(len(data))]


def write_hyperparam_results(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pick_best_result(rows):
    return max(rows, key=lambda r: r["discounted_sum"])


def save_frame(frame, path):
    plt.imsave(path, frame)


def run_eval_with_screenshots(env_name, output_dir, agent_label, act_fn):
    env_eval = gym.make(env_name, render_mode="rgb_array")
    state, _ = env_eval.reset()
    frame = env_eval.render()
    save_frame(frame, os.path.join(output_dir, f"{agent_label}_start.png"))

    done = False
    while not done:
        action = act_fn(state)
        state, _, terminated, truncated, _ = env_eval.step(action)
        done = terminated or truncated

    frame = env_eval.render()
    save_frame(frame, os.path.join(output_dir, f"{agent_label}_end.png"))
    env_eval.close()


def main():
    env = gym.make("CartPole-v1")
    episodes_num = 1000
    screenshots_dir = "screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)

    print("\n--- Rozpoczynam Eksperyment 1: Analiza wsp. dyskontowego (Gamma) ---")
    gammas = [0.1, 0.9, 0.99]
    gamma_results = {}

    for g in gammas:
        print(f"Trenowanie Q-Learning (Dyskretyzacja) dla gamma={g}...")
        agent = DiscretizedQLearning(env, gamma=g)
        rewards = agent.train(episodes=episodes_num)
        gamma_results[g] = smooth(rewards)

    plt.figure(figsize=(10, 5))
    for g in gammas:
        plt.plot(gamma_results[g], label=f"Gamma = {g}")
    plt.title("Wpływ współczynnika dyskontowego na uczenie (CartPole)")
    plt.xlabel("Epizod")
    plt.ylabel("Suma nagród (wygładzona)")
    plt.legend()
    plt.savefig("gamma_comparison.png")
    print("Zapisano wykres: gamma_comparison.png")

    print("\n--- Rozpoczynam Optymalizację Hiperparametrów ---")
    q_param_grid = [
        {"alpha": 0.1, "gamma": 0.99, "epsilon_decay": 0.995},
        {"alpha": 0.15, "gamma": 0.99, "epsilon_decay": 0.99},
    ]
    dqn_param_grid = [
        {"lr": 0.001, "gamma": 0.99, "epsilon_decay": 0.995},
        {"lr": 0.0005, "gamma": 0.99, "epsilon_decay": 0.99},
    ]

    hyperparam_rows = []

    for params in q_param_grid:
        agent = DiscretizedQLearning(env, **params)
        rewards, discounted = agent.train(
            episodes=episodes_num, track_discounted=True
        )
        hyperparam_rows.append(
            {
                "algorithm": "Q-Learning",
                "alpha": params["alpha"],
                "gamma": params["gamma"],
                "epsilon_decay": params["epsilon_decay"],
                "discounted_sum": float(np.sum(discounted)),
                "avg_reward": float(np.mean(rewards)),
            }
        )

    for params in dqn_param_grid:
        agent = DQNAgent(env, **params)
        rewards, discounted = agent.train(
            episodes=episodes_num, track_discounted=True
        )
        hyperparam_rows.append(
            {
                "algorithm": "DQN",
                "lr": params["lr"],
                "gamma": params["gamma"],
                "epsilon_decay": params["epsilon_decay"],
                "discounted_sum": float(np.sum(discounted)),
                "avg_reward": float(np.mean(rewards)),
            }
        )

    write_hyperparam_results("hyperparam_results.csv", hyperparam_rows)
    print("Zapisano: hyperparam_results.csv")

    best_q = pick_best_result(
        [row for row in hyperparam_rows if row["algorithm"] == "Q-Learning"]
    )
    best_dqn = pick_best_result(
        [row for row in hyperparam_rows if row["algorithm"] == "DQN"]
    )
    print(f"Najlepszy Q-Learning: {best_q}")
    print(f"Najlepszy DQN: {best_dqn}")

    print("\n--- Rozpoczynam Eksperyment 3: Q-Learning vs DQN ---")
    print("Trenowanie Q-Learning (zoptymalizowane parametry)...")
    agent_q = DiscretizedQLearning(
        env,
        alpha=best_q["alpha"],
        gamma=best_q["gamma"],
        epsilon_decay=best_q["epsilon_decay"],
    )
    rewards_q = agent_q.train(episodes=episodes_num)

    print("Trenowanie DQN (zoptymalizowane parametry, to potrwa chwilę)...")
    agent_dqn = DQNAgent(
        env,
        lr=best_dqn["lr"],
        gamma=best_dqn["gamma"],
        epsilon_decay=best_dqn["epsilon_decay"],
    )
    rewards_dqn = agent_dqn.train(episodes=episodes_num)

    plt.figure(figsize=(10, 5))
    plt.plot(smooth(rewards_q), label="Q-Learning (Dyskretyzacja)")
    plt.plot(smooth(rewards_dqn), label="DQN (PyTorch)")
    plt.title("Porównanie algorytmów w pierwszych 1000 epizodach (CartPole)")
    plt.xlabel("Epizod")
    plt.ylabel("Suma nagród (wygładzona)")
    plt.legend()
    plt.savefig("algorithm_comparison.png")
    print("Zapisano wykres: algorithm_comparison.png")

    print("Zapisywanie zrzutów ekranu z rozgrywek...")
    run_eval_with_screenshots(
        "CartPole-v1",
        screenshots_dir,
        "cartpole_qlearning",
        lambda state: np.argmax(agent_q.q_table[agent_q.discretize(state)]),
    )
    run_eval_with_screenshots(
        "CartPole-v1",
        screenshots_dir,
        "cartpole_dqn",
        lambda state: int(
            torch.argmax(
                agent_dqn.model(torch.FloatTensor(state).unsqueeze(0))
            ).item()
        ),
    )
    print(f"Zapisano zrzuty w katalogu: {screenshots_dir}")

    env.close()


if __name__ == "__main__":
    main()