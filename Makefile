# One-command reproduction of the paper's results.
#   make docker     -> fully turnkey: build + reproduce + validate in a container (only needs Docker)
#   make setup all  -> local: install deps, then reproduce + validate
PY ?= python3

.PHONY: help setup all reproduce validate figures docker clean fetch-data tier2 smoke

help:
	@echo "TIER 1 (CPU, no downloads, ~1 min) -- reproduces every number in the paper:"
	@echo "  make docker      build + reproduce + validate in a container (only needs Docker)"
	@echo "  make setup       pip install the Tier-1 (CPU) dependencies"
	@echo "  make all         reproduce + validate (after 'make setup')"
	@echo "  make reproduce   regenerate all tables/figures into results/"
	@echo "  make validate    assert reproduced numbers == the paper (self-test)"
	@echo ""
	@echo "TIER 2 (optional, GPU) -- validate / regenerate the model-output logs:"
	@echo "  make tier2       install GPU deps (torch/transformers/unsloth/...)"
	@echo "  make smoke       validate the inference code on a synthetic item (GPU; no dataset)"
	@echo "  make fetch-data  download official scorer repos + benchmark datasets"

setup:
	$(PY) -m pip install -r requirements-cpu.txt

all: reproduce validate

reproduce:
	$(PY) reproduce.py

validate:
	$(PY) validate.py

figures:
	$(PY) src/make_paper_figs.py

docker:
	docker build -f Dockerfile.cpu -t grader-audit .
	docker run --rm grader-audit

fetch-data:
	bash scripts/fetch_official_repos.sh
	bash scripts/fetch_datasets.sh

tier2:
	$(PY) -m pip install -r requirements-gpu.txt
	@echo "GPU deps installed. See README 'Tier 2' for the per-benchmark regeneration commands."

smoke:
	$(PY) smoke.py

clean:
	rm -f results/*.json results/*.csv results/*.png results/*.pdf
