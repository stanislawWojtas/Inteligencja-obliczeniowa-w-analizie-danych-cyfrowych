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

## Train Agent (DQN)

```bash
uv run train.py --timesteps 150000 --seed 42
```

Artifacts are saved under `artifacts/`:
- `dqn_crossy_road.zip`
- `monitor.csv`
- `train_summary.json`

## Evaluate Agent vs Random

```bash
uv run evaluate.py --model artifacts/dqn_crossy_road.zip --episodes 100
```

Outputs:
- prints summary JSON
- writes `artifacts/eval_summary.json`

## Notes for TODO.md

- Base (4 pkt): `CrossyRoadEnv` with reset/step/action+observation spaces and reward shaping.
- 6 pkt: continuous `Box(float32)` observations and strategic-policy metrics against random baseline.
- 8 pkt: `render_mode="human"`, `render_mode="rgb_array"`, and `close()` implemented.
