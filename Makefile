UV ?= uv
CONFIG ?= configs/experiment/smoke.yaml
DATA ?= data/processed/movielens
RUN ?=
CUTOFF ?=

.PHONY: setup test lint quickstart data embeddings train-baselines train-core evaluate ablation complementarity reproduce-main
setup:
	$(UV) sync --frozen --python 3.12
test:
	$(UV) run pytest -q
lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy src
quickstart:
	$(UV) run vassago experiment run --config configs/experiment/smoke.yaml
data:
	$(UV) run vassago data download --release ml-32m --output data/raw
	$(UV) run vassago data build --source data/raw/ml-32m --output $(DATA)
train-baselines train-core ablation:
	$(UV) run vassago experiment run --config $(CONFIG) --data $(DATA)
embeddings:
	$(UV) run vassago embeddings build --data $(DATA) --config $(CONFIG) --cutoff $(CUTOFF) --output artifacts/embeddings
evaluate:
	$(UV) run vassago evaluate --run $(RUN) --output $(RUN)/reevaluated.json
complementarity:
	$(UV) run python -c "from pathlib import Path; print(Path('$(RUN)/complementarity.json').read_text())"
reproduce-main:
	$(UV) run vassago experiment run --config configs/experiment/ml32m_main.yaml --data $(DATA)
