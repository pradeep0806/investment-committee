.PHONY: install test run up down lint

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest tests/ -v

run:
	.venv/bin/python -m committee.cli.main run --thesis "$(THESIS)" --budget $(or $(BUDGET),50000) --rounds $(or $(ROUNDS),3)

up:
	docker compose up --build

down:
	docker compose down

lint:
	.venv/bin/ruff check src/ tests/
	.venv/bin/mypy src/
