.PHONY: quickstart test-all

quickstart:
	python scripts/quickstart.py

test-all:
	python -m pip install ".[backtest]"
	pytest -q
