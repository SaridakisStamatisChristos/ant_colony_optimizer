.PHONY: quickstart test-all slow-checks

quickstart:
	python scripts/quickstart.py

test-all:
	python -m pip install ".[backtest]"
	pytest -q

slow-checks:
	python scripts/run_slow_checks.py
