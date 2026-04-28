from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class Car:
    lane: int
    x: float
    speed: float
    direction: int


class CrossyRoadEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, config: dict[str, Any] | None = None):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        cfg = config or {}
        self.width = int(cfg.get("width", 10))
        self.height = int(cfg.get("height", 14))
        self.max_steps = int(cfg.get("max_steps", 300))
        self.cars_per_lane = int(cfg.get("cars_per_lane", 2))
        self.car_speed_min = float(cfg.get("car_speed_min", 0.05))
        self.car_speed_max = float(cfg.get("car_speed_max", 0.12))
        self.obs_radius = int(cfg.get("obs_radius", 2))
        self.reward_forward = float(cfg.get("reward_forward", 0.15))
        self.reward_step_penalty = float(cfg.get("reward_step_penalty", -0.01))
        self.reward_collision = float(cfg.get("reward_collision", -1.0))
        self.reward_finish = float(cfg.get("reward_finish", 2.0))

        self.road_start_y = 1
        self.goal_y = self.height - 1
        self.safe_start_y = 0
        self.n_lanes = self.goal_y - self.road_start_y

        self.render_mode = render_mode
        self._np_random = np.random.default_rng()
        self.player_x = self.width // 2
        self.player_y = self.safe_start_y
        self.steps = 0
        self.cars: list[Car] = []

        self.action_space = spaces.Discrete(4)

        nearest_per_lane = self.n_lanes
        local_occ_size = (2 * self.obs_radius + 1) ** 2
        obs_dim = 2 + nearest_per_lane * 3 + local_occ_size
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.window = None
        self.clock = None
        self.cell_px = 40
        self._surface = None

    def _build_cars(self) -> list[Car]:
        cars: list[Car] = []
        for lane in range(self.road_start_y, self.goal_y):
            direction = int(self._np_random.choice([-1, 1]))
            # Keep lane cars separated and moving together to avoid overlaps.
            base_positions = np.linspace(0, self.width, num=self.cars_per_lane, endpoint=False)
            lane_shift = float(self._np_random.uniform(0.0, self.width / max(1, self.cars_per_lane)))
            for base_x in base_positions:
                x = float((base_x + lane_shift) % self.width)
                speed = float(self._np_random.uniform(self.car_speed_min, self.car_speed_max))
                cars.append(Car(lane=lane, x=x, speed=speed, direction=direction))
        return cars

    def _lane_index(self, lane_y: int) -> int:
        return lane_y - self.road_start_y

    def _move_player(self, action: int) -> None:
        if action == 0:  # up
            self.player_y = min(self.goal_y, self.player_y + 1)
        elif action == 1:  # down
            self.player_y = max(self.safe_start_y, self.player_y - 1)
        elif action == 2:  # left
            self.player_x = max(0, self.player_x - 1)
        elif action == 3:  # right
            self.player_x = min(self.width - 1, self.player_x + 1)

    def _update_cars(self) -> None:
        for car in self.cars:
            car.x += car.direction * car.speed
            if car.x < -1.0:
                car.x = self.width + 0.99
            elif car.x > self.width + 1.0:
                car.x = -0.99

    def _has_collision(self) -> bool:
        if self.player_y < self.road_start_y or self.player_y >= self.goal_y:
            return False
        for car in self.cars:
            if car.lane == self.player_y and abs(car.x - self.player_x) < 0.55:
                return True
        return False

    def _get_obs(self) -> np.ndarray:
        pieces: list[float] = []
        px = (self.player_x / max(1, self.width - 1)) * 2.0 - 1.0
        py = (self.player_y / max(1, self.goal_y)) * 2.0 - 1.0
        pieces.extend([px, py])

        for lane in range(self.road_start_y, self.goal_y):
            lane_cars = [c for c in self.cars if c.lane == lane]
            if not lane_cars:
                pieces.extend([1.0, 0.0, 0.0])
                continue
            nearest = min(lane_cars, key=lambda c: abs(c.x - self.player_x))
            rel_dist = (nearest.x - self.player_x) / max(1, self.width)
            rel_dist = float(np.clip(rel_dist, -1.0, 1.0))
            speed = nearest.speed / max(1e-6, self.car_speed_max)
            speed = float(np.clip(speed, 0.0, 1.0))
            direction = float(nearest.direction)
            pieces.extend([rel_dist, speed, direction])

        for dy in range(-self.obs_radius, self.obs_radius + 1):
            for dx in range(-self.obs_radius, self.obs_radius + 1):
                x = self.player_x + dx
                y = self.player_y + dy
                if x < 0 or x >= self.width or y < self.safe_start_y or y > self.goal_y:
                    pieces.append(-1.0)
                    continue
                occupied = 0.0
                for car in self.cars:
                    if car.lane == y and abs(car.x - x) < 0.55:
                        occupied = 1.0
                        break
                pieces.append(occupied)

        return np.asarray(pieces, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "player_x": self.player_x,
            "player_y": self.player_y,
            "progress": self.player_y / max(1, self.goal_y),
            "steps": self.steps,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.player_x = self.width // 2
        self.player_y = self.safe_start_y
        self.steps = 0
        self.cars = self._build_cars()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int | None):
        if action is not None and not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        prev_y = self.player_y
        self.steps += 1

        if action is not None:
            self._move_player(action)
        self._update_cars()

        reward = self.reward_step_penalty
        terminated = False
        truncated = False

        if self.player_y > prev_y:
            reward += self.reward_forward

        collision = self._has_collision()
        finished = self.player_y >= self.goal_y

        if collision:
            reward += self.reward_collision
            terminated = True
        elif finished:
            reward += self.reward_finish
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        info["collision"] = collision
        info["finished"] = finished

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _ensure_pygame(self) -> None:
        import pygame

        if self.window is None:
            pygame.init()
            w = self.width * self.cell_px
            h = self.height * self.cell_px
            self.window = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Crossy Road Env")
            self.clock = pygame.time.Clock()
            self._surface = pygame.Surface((w, h))

    def get_human_action(self, default_action: int | None = None) -> int | None:
        if self.render_mode != "human":
            return default_action

        import pygame

        self._ensure_pygame()
        action = default_action
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_w, pygame.K_UP):
                    action = 0
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    action = 1
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    action = 2
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    action = 3
        return action

    def render(self):
        if self.render_mode is None:
            return None

        import pygame

        self._ensure_pygame()

        assert self._surface is not None
        assert self.window is not None

        self._surface.fill((34, 139, 34))

        for y in range(self.road_start_y, self.goal_y):
            pygame.draw.rect(
                self._surface,
                (60, 60, 60),
                (0, (self.height - 1 - y) * self.cell_px, self.width * self.cell_px, self.cell_px),
            )

        pygame.draw.rect(
            self._surface,
            (80, 180, 255),
            (0, (self.height - 1 - self.goal_y) * self.cell_px, self.width * self.cell_px, self.cell_px),
        )

        for car in self.cars:
            x_px = int(car.x * self.cell_px)
            y_px = (self.height - 1 - car.lane) * self.cell_px
            pygame.draw.rect(self._surface, (220, 60, 60), (x_px, y_px + 6, self.cell_px, self.cell_px - 12))

        player_y_px = (self.height - 1 - self.player_y) * self.cell_px
        player_x_px = self.player_x * self.cell_px
        pygame.draw.rect(
            self._surface,
            (255, 255, 80),
            (player_x_px + 6, player_y_px + 6, self.cell_px - 12, self.cell_px - 12),
        )

        if self.render_mode == "human":
            self.window.blit(self._surface, (0, 0))
            pygame.display.flip()
            assert self.clock is not None
            self.clock.tick(self.metadata["render_fps"])
            return None

        rgb = pygame.surfarray.array3d(self._surface)
        return np.transpose(rgb, (1, 0, 2))

    def close(self):
        if self.window is not None:
            import pygame

            pygame.quit()
            self.window = None
            self.clock = None
            self._surface = None
