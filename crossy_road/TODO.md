# Projekt 4: Własne środowisko — Crossy Road
**Deadline:** 2026-05-20, 23:59 via MS TEAMS (kod + sprawozdanie)

---

## 4 PKT — Base

- [x] **Gymnasium env** — implement `CrossyRoadEnv(gymnasium.Env)`:
  - [x] `__init__`: define `observation_space`, `action_space` (4 directions + wait)
  - [x] Training wrapper: restrict DQN to forward/wait actions to avoid lateral/backward jitter
  - [x] Training callback: evaluate periodically and save the best checkpoint instead of the last unstable DQN weights
  - [x] `reset()`: return initial obs + info
  - [x] `step(action)`: move player, check collisions, return `obs, reward, terminated, truncated, info`
  - [x] Game logic: player crosses road, cars move, goal = reach other side
  - [x] Reward shaping: +step forward, -collision, +finish
- [x] **Agent** — write solver script (`agent.py` or `train.py`):
  - [x] Choose algorithm (Q-learning or DQN recommended)
  - [x] Train loop with episode logging
- [ ] **Agent completes game** — verify agent reaches goal consistently (log win rate)
- [ ] **Sprawozdanie** (2-3 strony A4):
  - [ ] Opis środowiska (obs space, action space, reward function)
  - [ ] Opis algorytmu
  - [ ] Eksperymenty: krzywa nagrody, win rate vs episodes
  - [ ] Wnioski

---

## 6 PKT — Continuous / Large Observation Space + Strategy

- [x] **Continuous observation space** — use `gymnasium.spaces.Box` (float32):
  - [x] Include: player position (x,y), distance to cars, car speeds, lanes state
  - [ ] OR large discrete space (e.g. full grid flattened, >1000 states)
- [x] **Strategic agent behavior** (not random):
  - [x] Use DQN / PPO (stable-baselines3 recommended) for proper policy learning
  - [ ] Show agent avoids cars, waits for gaps — not just moving randomly
  - [ ] Log + include metrics proving strategic play (e.g. survival time > random baseline)

---

## 8 PKT — Graphical Mode

- [x] **Pygame render mode** — implement `render()` in env:
  - [x] `render_mode="human"` → live pygame window
  - [x] `render_mode="rgb_array"` → return numpy array (for video recording)
  - [x] Draw: road lanes, cars (rectangles/sprites), player, goal line
  - [x] `close()` → quit pygame
- [ ] Test graphical mode works alongside trained agent (demo run)

---

## Final Checklist Before Submission

- [x] `pip install` list documented (requirements.txt or README)
- [ ] Code runs end-to-end from clean env
- [ ] Sprawozdanie PDF ready
- [ ] Upload to MS TEAMS: code + sprawozdanie
