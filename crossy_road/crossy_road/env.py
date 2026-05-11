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


ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_WAIT = 4

ACTION_NAMES = {
    ACTION_UP: "up",
    ACTION_DOWN: "down",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
    ACTION_WAIT: "wait",
}

ACTION_VECTORS = {
    ACTION_UP: (0.0, 1.0),
    ACTION_DOWN: (0.0, -1.0),
    ACTION_LEFT: (-1.0, 0.0),
    ACTION_RIGHT: (1.0, 0.0),
    ACTION_WAIT: (0.0, 0.0),
}


class CrossyRoadEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, config: dict[str, Any] | None = None):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        cfg = config or {}
        self.width = int(cfg.get("width", 10))
        self.height = int(cfg.get("height", 14))
        self.goal_distance = int(cfg.get("goal_distance", self.height - 1))
        self.height = max(self.height, self.goal_distance + 1)
        max_steps_cfg = cfg.get("max_steps")
        self.max_steps = int(max_steps_cfg) if max_steps_cfg is not None else None
        self.cars_per_lane = int(cfg.get("cars_per_lane", 2))
        self.car_speed_min = float(cfg.get("car_speed_min", 0.05))
        self.car_speed_max = float(cfg.get("car_speed_max", 0.12))
        self.player_speed = float(cfg.get("player_speed", 0.2))
        self.obs_radius = int(cfg.get("obs_radius", 2))
        self.reward_step_penalty = float(cfg.get("reward_step_penalty", -0.01))
        self.reward_collision = float(cfg.get("reward_collision", -20.0))
        self.reward_finish = float(cfg.get("reward_finish", 20.0))
        self.reward_progress = float(cfg.get("reward_progress", cfg.get("reward_score", 0.75)))
        self.reward_backward = float(cfg.get("reward_backward", -0.08))
        self.reward_sideways = float(cfg.get("reward_sideways", -0.03))
        self.reward_wait = float(cfg.get("reward_wait", 0.0))
        self.reward_safe_wait = float(cfg.get("reward_safe_wait", 0.015))
        self.reward_unsafe_forward = float(cfg.get("reward_unsafe_forward", -0.15))
        self.reward_unsafe_wait = float(cfg.get("reward_unsafe_wait", -0.15))
        self.reward_stall = float(cfg.get("reward_stall", -0.03))
        self.stall_threshold = int(cfg.get("stall_threshold", 25))
        self.safety_horizon = int(cfg.get("safety_horizon", 8))
        self.safety_distance = float(cfg.get("safety_distance", 1.1))
        self.safety_margin = float(cfg.get("safety_margin", 0.05))

        self.safe_block_min = int(cfg.get("safe_block_min", 1))
        self.safe_block_max = int(cfg.get("safe_block_max", 5))
        self.safe_block_weight_power = float(cfg.get("safe_block_weight_power", 2.0))
        self.traffic_block_min = int(cfg.get("traffic_block_min", 3))
        self.traffic_block_max = int(cfg.get("traffic_block_max", 6))

        self.start_safe_rows = int(cfg.get("start_safe_rows", 3))
        self.road_start_y = max(1, self.start_safe_rows)
        self.goal_y = self.goal_distance
        self.safe_start_y = 0
        self.n_lanes = self.goal_y - self.road_start_y

        self.render_mode = render_mode
        self._np_random = np.random.default_rng()
        self.player_x = float(self.width // 2)
        self.player_y = float(self.safe_start_y)
        self.max_player_y = float(self.safe_start_y)
        self.steps = 0
        self.no_progress_steps = 0
        self.cars: list[Car] = []
        self.safe_lanes: set[int] = set()

        self.action_space = spaces.Discrete(5)

        nearest_per_lane = self.n_lanes
        local_occ_size = (2 * self.obs_radius + 1) ** 2
        obs_dim = 4 + nearest_per_lane * 3 + local_occ_size
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
        self._font = None
        self._sprite_inset = 6.0 / self.cell_px
        self._car_width = 1.0
        self._car_height = 1.0 - 2.0 * self._sprite_inset
        self._player_width = 1.0 - 2.0 * self._sprite_inset
        self._player_height = 1.0 - 2.0 * self._sprite_inset

    def _build_safe_lanes(self) -> set[int]:
        safe_lanes: set[int] = set()
        lane = self.road_start_y
        while lane < self.goal_y:
            traffic_span = int(self._np_random.integers(self.traffic_block_min, self.traffic_block_max + 1))
            lane += traffic_span
            if lane >= self.goal_y:
                break
            safe_lengths = np.arange(self.safe_block_min, self.safe_block_max + 1)
            # Favor short green segments (especially length 1) over long ones.
            weights = 1.0 / np.power(safe_lengths, max(1e-6, self.safe_block_weight_power))
            probs = weights / weights.sum()
            safe_span = int(self._np_random.choice(safe_lengths, p=probs))
            safe_end = min(self.goal_y, lane + safe_span)
            for y in range(lane, safe_end):
                safe_lanes.add(y)
            lane = safe_end
        return safe_lanes

    def _build_cars(self) -> list[Car]:
        cars: list[Car] = []
        for lane in range(self.road_start_y, self.goal_y):
            if lane in self.safe_lanes:
                continue
            direction = int(self._np_random.choice([-1, 1]))
            lane_speed = float(self._np_random.uniform(self.car_speed_min, self.car_speed_max))
            # Keep lane cars separated and moving together to avoid overlaps.
            base_positions = np.linspace(0, self.width, num=self.cars_per_lane, endpoint=False)
            lane_shift = float(self._np_random.uniform(0.0, self.width / max(1, self.cars_per_lane)))
            for base_x in base_positions:
                x = float((base_x + lane_shift) % self.width)
                cars.append(Car(lane=lane, x=x, speed=lane_speed, direction=direction))
        return cars

    def _lane_index(self, lane_y: int) -> int:
        return lane_y - self.road_start_y

    def _action_vector(self, action: int | None) -> tuple[float, float]:
        if action is None:
            return ACTION_VECTORS[ACTION_WAIT]
        return ACTION_VECTORS[int(action)]

    def _move_player(self, action: int | None) -> None:
        self.player_x, self.player_y = self._project_player_position(action)

    def _project_player_position(self, action: int | None) -> tuple[float, float]:
        dx, dy = self._action_vector(action)
        if dx == 0.0 and dy == 0.0:
            return self.player_x, self.player_y

        scale = self.player_speed / np.sqrt(2.0) if dx != 0.0 and dy != 0.0 else self.player_speed
        next_x = float(np.clip(self.player_x + dx * scale, 0.0, float(self.width - 1)))
        next_y = float(np.clip(self.player_y + dy * scale, float(self.safe_start_y), float(self.goal_y)))
        return next_x, next_y

    def _update_cars(self) -> None:
        for car in self.cars:
            car.x += car.direction * car.speed
            if car.x < -1.0:
                car.x = self.width + 0.99
            elif car.x > self.width + 1.0:
                car.x = -0.99

    def _has_collision(self) -> bool:
        player_left = self.player_x + self._sprite_inset
        player_right = player_left + self._player_width
        player_bottom = self.player_y + self._sprite_inset
        player_top = player_bottom + self._player_height

        if player_top <= self.road_start_y or player_bottom >= self.goal_y:
            return False

        for car in self.cars:
            car_bottom = car.lane + self._sprite_inset
            car_top = car_bottom + self._car_height
            if not self._overlap_1d(player_bottom, player_top, car_bottom, car_top):
                continue

            for car_x in (car.x - self.width, car.x, car.x + self.width):
                car_left = car_x
                car_right = car_left + self._car_width
                if self._overlap_1d(player_left, player_right, car_left, car_right):
                    return True
        return False

    def _position_has_projected_traffic(self, player_x: float, player_y: float, horizon: int) -> bool:
        player_bottom = player_y + self._sprite_inset
        player_top = player_bottom + self._player_height

        if player_top <= self.road_start_y or player_bottom >= self.goal_y:
            return False

        for car in self.cars:
            car_bottom = car.lane + self._sprite_inset
            car_top = car_bottom + self._car_height
            if not self._overlap_1d(player_bottom, player_top, car_bottom, car_top):
                continue

            for tick in range(horizon + 1):
                projected_x = (car.x + car.direction * car.speed * tick) % self.width
                if self._x_distance(projected_x, player_x) < self.safety_distance:
                    return True
        return False

    def _position_has_projected_collision(self, player_x: float, player_y: float, ticks_ahead: int) -> bool:
        player_left = player_x + self._sprite_inset - self.safety_margin
        player_right = player_left + self._player_width + 2.0 * self.safety_margin
        player_bottom = player_y + self._sprite_inset - self.safety_margin
        player_top = player_bottom + self._player_height + 2.0 * self.safety_margin

        if player_top <= self.road_start_y or player_bottom >= self.goal_y:
            return False

        for car in self.cars:
            car_bottom = car.lane + self._sprite_inset
            car_top = car_bottom + self._car_height
            if not self._overlap_1d(player_bottom, player_top, car_bottom, car_top):
                continue

            projected_x = (car.x + car.direction * car.speed * ticks_ahead) % self.width
            for car_x in (projected_x - self.width, projected_x, projected_x + self.width):
                car_left = car_x
                car_right = car_left + self._car_width
                if self._overlap_1d(player_left, player_right, car_left, car_right):
                    return True
        return False

    def _forward_path_has_projected_traffic(self, action: int | None = ACTION_UP) -> bool:
        player_x, player_y = self._project_player_position(action)
        for ticks_ahead in range(1, self.safety_horizon + 1):
            if self._position_has_projected_collision(player_x, player_y, ticks_ahead):
                return True
            player_x, player_y = self._project_position_from(player_x, player_y, ACTION_UP)
        return False

    def _x_distance(self, a: float, b: float) -> float:
        """Distance on wrapped X axis so edges collide correctly."""
        delta = abs(a - b)
        return min(delta, abs(self.width - delta))

    def _signed_x_delta(self, a: float, b: float) -> float:
        """Signed shortest delta from b to a on the wrapped X axis."""
        return float((a - b + self.width / 2.0) % self.width - self.width / 2.0)

    def _overlap_1d(self, a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
        return a_min < b_max and b_min < a_max

    def _project_position_from(self, player_x: float, player_y: float, action: int | None) -> tuple[float, float]:
        dx, dy = self._action_vector(action)
        if dx == 0.0 and dy == 0.0:
            return player_x, player_y

        scale = self.player_speed / np.sqrt(2.0) if dx != 0.0 and dy != 0.0 else self.player_speed
        next_x = float(np.clip(player_x + dx * scale, 0.0, float(self.width - 1)))
        next_y = float(np.clip(player_y + dy * scale, float(self.safe_start_y), float(self.goal_y)))
        return next_x, next_y

    def _get_obs(self) -> np.ndarray:
        pieces: list[float] = []
        px = (self.player_x / max(1, self.width - 1)) * 2.0 - 1.0
        py = (self.player_y / max(1, self.goal_y)) * 2.0 - 1.0
        pieces.extend([px, py])
        pieces.append(1.0 if self._forward_path_has_projected_traffic(ACTION_UP) else -1.0)
        pieces.append(1.0 if self._position_has_projected_collision(self.player_x, self.player_y, 1) else -1.0)

        for lane in range(self.road_start_y, self.goal_y):
            lane_cars = [c for c in self.cars if c.lane == lane]
            if not lane_cars:
                pieces.extend([1.0, 0.0, 0.0])
                continue
            nearest = min(lane_cars, key=lambda c: abs(self._signed_x_delta(c.x, self.player_x)))
            rel_dist = self._signed_x_delta(nearest.x, self.player_x) / max(1e-6, self.width / 2.0)
            rel_dist = float(np.clip(rel_dist, -1.0, 1.0))
            speed = nearest.speed / max(1e-6, self.car_speed_max)
            speed = float(np.clip(speed, 0.0, 1.0))
            direction = float(nearest.direction)
            pieces.extend([rel_dist, speed, direction])

        for dy in range(-self.obs_radius, self.obs_radius + 1):
            for dx in range(-self.obs_radius, self.obs_radius + 1):
                x = int(round(self.player_x)) + dx
                y = int(round(self.player_y)) + dy
                if x < 0 or x >= self.width or y < self.safe_start_y or y > self.goal_y:
                    pieces.append(-1.0)
                    continue
                occupied = 0.0
                for car in self.cars:
                    if car.lane == y and self._x_distance(car.x, float(x)) < (0.5 + 0.5):
                        occupied = 1.0
                        break
                pieces.append(occupied)

        return np.asarray(pieces, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "player_x": self.player_x,
            "player_y": self.player_y,
            "progress": self.player_y / max(1, self.goal_y),
            "score": int(self.max_player_y - self.safe_start_y),
            "steps": self.steps,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.player_x = float(self.width // 2)
        self.player_y = float(self.safe_start_y)
        self.max_player_y = float(self.safe_start_y)
        self.steps = 0
        self.no_progress_steps = 0
        self.safe_lanes = self._build_safe_lanes()
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
        prev_max_y = self.max_player_y
        unsafe_forward = self._forward_path_has_projected_traffic(ACTION_UP)
        unsafe_wait = self._position_has_projected_collision(self.player_x, self.player_y, 1)
        self.steps += 1

        self._move_player(action)
        self.max_player_y = max(self.max_player_y, self.player_y)
        self._update_cars()

        reward = self.reward_step_penalty
        terminated = False
        truncated = False
        dx, dy = self._action_vector(action)

        score_delta = max(0.0, self.max_player_y - prev_max_y)
        if score_delta > 0.0:
            reward += score_delta * self.reward_progress
            self.no_progress_steps = 0
        elif dy < 0.0 or self.player_y < prev_y:
            reward += self.reward_backward
            self.no_progress_steps += 1
        else:
            reward += self.reward_wait
            self.no_progress_steps += 1

        if dx != 0.0:
            reward += self.reward_sideways
        if dx == 0.0 and dy == 0.0 and unsafe_wait:
            reward += self.reward_unsafe_wait
        elif dx == 0.0 and dy == 0.0 and unsafe_forward:
            reward += self.reward_safe_wait
        elif dy > 0.0 and unsafe_forward:
            reward += self.reward_unsafe_forward
        if self.no_progress_steps >= self.stall_threshold:
            reward += self.reward_stall

        collision = self._has_collision()
        finished = self.player_y >= self.goal_y

        if collision:
            reward += self.reward_collision
            terminated = True
        elif finished:
            reward += self.reward_finish
            terminated = True

        if self.max_steps is not None and self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        info["collision"] = collision
        info["finished"] = finished
        info["action_name"] = ACTION_NAMES[ACTION_WAIT if action is None else int(action)]

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
            self._font = pygame.font.Font(None, 40)

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
        pressed = pygame.key.get_pressed()
        up = pressed[pygame.K_w] or pressed[pygame.K_UP]
        down = pressed[pygame.K_s] or pressed[pygame.K_DOWN]
        left = pressed[pygame.K_a] or pressed[pygame.K_LEFT]
        right = pressed[pygame.K_d] or pressed[pygame.K_RIGHT]

        if up and not down:
            action = 0
        elif down and not up:
            action = 1
        elif left and not right:
            action = 2
        elif right and not left:
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
            color = (34, 139, 34) if y in self.safe_lanes else (60, 60, 60)
            pygame.draw.rect(
                self._surface,
                color,
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

        assert self._font is not None
        score = int(self.max_player_y - self.safe_start_y)
        score_surface = self._font.render(str(score), True, (255, 255, 255))
        score_rect = score_surface.get_rect(midtop=(self.width * self.cell_px // 2, 8))
        self._surface.blit(score_surface, score_rect)

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
            self._font = None
