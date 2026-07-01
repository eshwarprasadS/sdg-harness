.PHONY: test lint typecheck

test:
	python -m pytest

lint:
	ruff check .

typecheck:
	mypy src/sdg_harness
