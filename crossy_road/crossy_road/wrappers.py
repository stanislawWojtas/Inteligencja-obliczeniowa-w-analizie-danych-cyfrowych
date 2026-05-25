from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
from gymnasium import spaces

from crossy_road.env import ACTION_UP, ACTION_WAIT


class ActionSubsetWrapper(gym.Wrapper):
    """Expose a smaller discrete action space and map it to env actions."""

    def __init__(self, env: gym.Env, actions: Sequence[int] = (ACTION_UP, ACTION_WAIT)):
        if not actions:
            raise ValueError("actions must contain at least one action")
        super().__init__(env)
        self.actions = tuple(int(action) for action in actions)
        self.action_space = spaces.Discrete(len(self.actions))

    def step(self, action: int):
        agent_action = int(action)
        if not self.action_space.contains(agent_action):
            raise ValueError(f"Invalid training action: {action}")

        mapped_action = self.actions[agent_action]
        obs, reward, terminated, truncated, info = self.env.step(mapped_action)
        info = dict(info)
        info["agent_action"] = agent_action
        info["mapped_action"] = mapped_action
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any):
        return self.env.reset(**kwargs)
