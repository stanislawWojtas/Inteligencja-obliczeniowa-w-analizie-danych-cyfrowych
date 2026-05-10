PYTHON ?= python3
UV ?= uv
VENV ?= .venv

TIMESTEPS ?= 150000
SEED ?= 42
EPISODES ?= 100
MODEL ?= artifacts/dqn_crossy_road.zip
RENDER_MODE ?=
RENDER_BASELINE ?= 0
MAX_STEPS ?= 600

.PHONY: help venv sync setup play train eval clean

help:
	@echo "Targets:"
	@echo "  make setup                 # create venv + sync deps from pyproject.toml"
	@echo "  make play                  # run human-play mode"
	@echo "  make train                 # train DQN agent"
	@echo "  make eval                  # evaluate model vs random baseline"
	@echo "  make clean                 # remove generated artifacts"
	@echo ""
	@echo "Optional vars: TIMESTEPS=$(TIMESTEPS) SEED=$(SEED) EPISODES=$(EPISODES) MODEL=$(MODEL) RENDER_MODE=$(RENDER_MODE) RENDER_BASELINE=$(RENDER_BASELINE) MAX_STEPS=$(MAX_STEPS)"

venv:
	$(UV) venv $(VENV)

sync: venv
	$(UV) sync

setup: sync

play:
	$(UV) run play_human.py

train:
	$(UV) run train.py --timesteps $(TIMESTEPS) --seed $(SEED) $(if $(RENDER_MODE),--render-mode $(RENDER_MODE),)

eval:
	$(UV) run evaluate.py --model $(MODEL) --episodes $(EPISODES) --seed $(SEED) --max-steps $(MAX_STEPS) $(if $(RENDER_MODE),--render-mode $(RENDER_MODE),) $(if $(filter 1 true yes,$(RENDER_BASELINE)),--render-baseline,)

clean:
	rm -rf artifacts __pycache__ crossy_road/__pycache__
