.DEFAULT_GOAL := help
IMAGE ?= opencv-face-mask-detection:dev
export MD_TEST_NATS_URL ?= nats://127.0.0.1:4222
export MD_TEST_S3_ENDPOINT ?= http://127.0.0.1:8333

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install dependencies (uv) and git hooks
	uv sync --all-groups
	uv run pre-commit install

.PHONY: lint
lint: ## Ruff lint + format check + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

.PHONY: fmt
fmt: ## Auto-fix and format
	uv run ruff check . --fix
	uv run ruff format .

.PHONY: test
test: ## Unit tests (no external services)
	uv run pytest -m "not integration"

.PHONY: test-integration
test-integration: ## Integration tests against `make up` (NATS + S3 on localhost)
	uv run pytest -m integration --cov-fail-under=0

.PHONY: test-all
test-all: ## Unit + integration with the CI coverage gate
	uv run pytest --cov-fail-under=90

.PHONY: run-api
run-api: ## Run the API locally with auto-reload
	uv run uvicorn mask_detection.api.app:create_app --factory --reload --port 8000

.PHONY: docker
docker: ## Build the container image
	docker build -t $(IMAGE) --build-arg REVISION=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) .

.PHONY: up
up: ## Start the full local stack (API, workers, NATS, S3, observability)
	docker compose --profile observability up -d --build --wait

.PHONY: down
down: ## Stop the local stack and delete its volumes
	docker compose --profile observability down -v

.PHONY: train
train: ## Retrain the mask classifier (needs PyTorch; see training/README.md)
	@echo 'See training/README.md: python training/train.py --data /tmp/maskdata --out models'

.PHONY: helm-lint
helm-lint: ## Lint and render the Helm chart
	helm lint deploy/helm/mask-detection --strict
	helm template fd deploy/helm/mask-detection --set auth.apiKeyHashes=$$(printf x | shasum -a 256 | cut -d' ' -f1) >/dev/null

.PHONY: bench
bench: ## Load-test a running API (see scripts/bench.py)
	uv run python scripts/bench.py --url http://127.0.0.1:8000 --image tests/data/astronaut.jpg

.PHONY: dashboard
dashboard: ## Regenerate the Grafana dashboard JSON
	python3 scripts/gen_dashboard.py > deploy/compose/grafana/dashboards/mask-detection.json
	cp deploy/compose/grafana/dashboards/mask-detection.json deploy/helm/mask-detection/files/mask-detection-dashboard.json
