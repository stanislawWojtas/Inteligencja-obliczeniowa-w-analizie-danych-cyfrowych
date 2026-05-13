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
- `learning_curve.png`
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

## Notebook Workflow

Open `crossy_road_workflow.ipynb` for a notebook version of the full workflow:

- setup commands with `uv`
- training and evaluation commands
- learning-curve display from `artifacts/learning_curve.png`
- inline `rgb_array` gameplay rendering in the notebook
- placeholders for screenshots from human play and notebook playback

## Actions

The base environment action space is `Discrete(5)`:

- `0`: up
- `1`: down
- `2`: left
- `3`: right
- `4`: wait/no-op

The DQN training and evaluation pipeline uses the full `Discrete(5)` action space so the policy can dodge traffic:

- `0`: up
- `1`: down
- `2`: left
- `3`: right
- `4`: wait/no-op

Existing saved models trained before this change only know `up` and `wait`; retrain to let the agent learn escape moves.

## Notes for TODO.md

- Base (4 pkt): `CrossyRoadEnv` with reset/step/action+observation spaces and reward shaping.
- 6 pkt: continuous `Box(float32)` observations and strategic-policy metrics against random baseline.
- 8 pkt: `render_mode="human"`, `render_mode="rgb_array"`, and `close()` implemented.
