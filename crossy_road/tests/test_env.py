import pytest

from crossy_road.env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_UP,
    ACTION_WAIT,
    ALL_ACTIONS,
    Car,
    CrossyRoadEnv,
)
from crossy_road.wrappers import ActionSubsetWrapper


def make_env(**config):
    env = CrossyRoadEnv(config={"max_steps": 20, **config})
    env.reset(seed=123)
    return env


def test_wait_action_is_valid_and_invalid_action_raises():
    env = make_env()

    assert env.action_space.contains(ACTION_WAIT)
    assert not env.action_space.contains(5)

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(5)


def test_training_action_wrapper_maps_subset_actions():
    env = ActionSubsetWrapper(make_env(), actions=(ACTION_UP, ACTION_WAIT))

    assert env.action_space.n == 2

    env.step(0)
    assert env.unwrapped.player_y > env.unwrapped.safe_start_y

    position = (env.unwrapped.player_x, env.unwrapped.player_y)
    _, _, _, _, info = env.step(1)
    assert (env.unwrapped.player_x, env.unwrapped.player_y) == position
    assert info["mapped_action"] == ACTION_WAIT

    with pytest.raises(ValueError, match="Invalid training action"):
        env.step(2)


def test_training_pipeline_actions_cover_all_escape_moves():
    env = ActionSubsetWrapper(make_env(), actions=ALL_ACTIONS)

    assert env.action_space.n == len(ALL_ACTIONS)
    assert env.actions == ALL_ACTIONS

    start_x = env.unwrapped.player_x
    env.step(2)
    assert env.unwrapped.player_x < start_x

    moved_x = env.unwrapped.player_x
    env.step(3)
    assert env.unwrapped.player_x > moved_x

    start_y = env.unwrapped.player_y
    env.step(1)
    assert env.unwrapped.player_y <= start_y


def test_wait_keeps_player_still_while_cars_advance():
    env = make_env(start_safe_rows=1, goal_distance=4)
    env.cars = [Car(lane=env.road_start_y, x=0.0, speed=0.5, direction=1)]
    start_position = (env.player_x, env.player_y)

    _, reward, terminated, truncated, info = env.step(ACTION_WAIT)

    assert (env.player_x, env.player_y) == start_position
    assert env.cars[0].x == pytest.approx(0.5)
    assert reward == pytest.approx(env.reward_step_penalty + env.reward_wait)
    assert not terminated
    assert not truncated
    assert info["action_name"] == "wait"


def test_nearest_car_observation_uses_signed_wrapped_distance():
    env = make_env(width=10, start_safe_rows=1, goal_distance=2)
    env.player_x = 0.0
    env.cars = [Car(lane=env.road_start_y, x=9.8, speed=0.0, direction=1)]

    obs = env._get_obs()

    assert obs[4] == pytest.approx(-0.04, abs=1e-6)


def test_observation_includes_forward_risk_flag():
    env = make_env(width=10, start_safe_rows=1, goal_distance=4, safety_horizon=6, safety_distance=1.1)
    env.player_y = 0.2
    env.max_player_y = env.player_y
    env.cars = [Car(lane=env.road_start_y, x=env.player_x + 1.0, speed=0.1, direction=-1)]

    obs = env._get_obs()

    assert obs[2] == 1.0


def test_observation_includes_wait_risk_flag():
    env = make_env(width=10, start_safe_rows=1, goal_distance=4)
    env.player_y = float(env.road_start_y)
    env.max_player_y = env.player_y
    env.cars = [Car(lane=env.road_start_y, x=env.player_x, speed=0.0, direction=1)]

    obs = env._get_obs()

    assert obs[3] == 1.0


def test_wait_is_cheaper_than_sideways_or_backward_without_progress():
    env_wait = make_env(start_safe_rows=4, goal_distance=4)
    _, wait_reward, *_ = env_wait.step(ACTION_WAIT)

    env_left = make_env(start_safe_rows=4, goal_distance=4)
    _, left_reward, *_ = env_left.step(ACTION_LEFT)

    env_down = make_env(start_safe_rows=4, goal_distance=4)
    _, down_reward, *_ = env_down.step(ACTION_DOWN)

    assert wait_reward > left_reward
    assert left_reward > down_reward


def test_wait_is_rewarded_over_forward_move_when_next_position_is_unsafe():
    config = {
        "width": 10,
        "start_safe_rows": 1,
        "goal_distance": 4,
        "safety_horizon": 6,
        "safety_distance": 1.1,
    }
    env_wait = make_env(**config)
    env_wait.player_y = 0.2
    env_wait.max_player_y = env_wait.player_y
    env_wait.cars = [Car(lane=env_wait.road_start_y, x=env_wait.player_x + 1.0, speed=0.1, direction=-1)]
    _, wait_reward, wait_terminated, *_ = env_wait.step(ACTION_WAIT)

    env_up = make_env(**config)
    env_up.player_y = 0.2
    env_up.max_player_y = env_up.player_y
    env_up.cars = [Car(lane=env_up.road_start_y, x=env_up.player_x + 1.0, speed=0.1, direction=-1)]
    _, up_reward, up_terminated, *_ = env_up.step(ACTION_UP)

    assert not wait_terminated
    assert not up_terminated
    assert wait_reward > up_reward


def test_collision_terminates_with_negative_reward():
    env = make_env(start_safe_rows=1, goal_distance=4)
    env.player_y = float(env.road_start_y)
    env.max_player_y = env.player_y
    env.cars = [Car(lane=env.road_start_y, x=env.player_x, speed=0.0, direction=1)]

    _, reward, terminated, truncated, info = env.step(ACTION_WAIT)

    assert terminated
    assert not truncated
    assert info["collision"]
    assert reward < 0.0


def test_finish_terminates_with_positive_reward():
    env = make_env(start_safe_rows=1, goal_distance=1, player_speed=1.0)
    env.cars = []

    _, reward, terminated, truncated, info = env.step(ACTION_UP)

    assert terminated
    assert not truncated
    assert info["finished"]
    assert reward > 0.0
