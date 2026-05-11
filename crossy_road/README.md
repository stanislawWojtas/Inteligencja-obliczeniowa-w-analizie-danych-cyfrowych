# Crossy Road Gymnasium Environment

This project provides a custom `gymnasium.Env` for Crossy Road style gameplay with:
- Human-playable `pygame` rendering
- `rgb_array` rendering for recording
- DQN training/evaluation pipeline for report metrics

## Setup

```bash
uv venv .venv
uv sync
```

## Human Play

```bash
uv run play_human.py
```

Controls: `WASD` or arrow keys.

Config example (in code): `CrossyRoadEnv(config={"goal_distance": 200, "max_steps": None})`
- `goal_distance` controls how far the player must travel to finish.
- `max_steps=None` means no time/step limit (infinite in time).

## Train Agent (DQN)

```bash
uv run train.py --timesteps 100000 --seed 42 --max-steps 500
```

Artifacts are saved under `artifacts/`:
- `dqn_crossy_road.zip`
- `best_model.zip`
- `evaluations.npz`
- `monitor.csv`
- `train_summary.json`

Training evaluates periodically and copies the best checkpoint to `dqn_crossy_road.zip`; this avoids keeping a degraded final DQN checkpoint when longer training becomes unstable.

## Evaluate Agent vs Random

```bash
uv run evaluate.py --model artifacts/dqn_crossy_road.zip --episodes 100
```

Outputs:
- prints summary JSON
- writes `artifacts/eval_summary.json`
- includes action distribution metrics, including `wait_rate` and risk-response rates

## Actions

The base environment action space is `Discrete(5)`:
- `0`: up
- `1`: down
- `2`: left
- `3`: right
- `4`: wait/no-op

The DQN training and evaluation pipeline wraps the environment with a smaller `Discrete(2)` action space:
- `0`: up
- `1`: wait/no-op

This keeps human play flexible while preventing the trained policy from learning useless lateral/backward jitter. Crossing this map only requires advancing when safe and waiting for traffic gaps.

## Notes for TODO.md

- Base (4 pkt): `CrossyRoadEnv` with reset/step/action+observation spaces and reward shaping.
- 6 pkt: continuous `Box(float32)` observations and strategic-policy metrics against random baseline.
- 8 pkt: `render_mode="human"`, `render_mode="rgb_array"`, and `close()` implemented.
