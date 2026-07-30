.PHONY: install run test lint

install:
	uv sync --extra dev

run:
	uv run panel serve examples/app.py --show --autoreload

test:
	uv run pytest

lint:
	uv run ruff check .
