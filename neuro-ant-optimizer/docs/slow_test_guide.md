# Running slow tests

This project marks a subset of the pytest suite with the `slow` marker. These
checks cover multiprocessing and service end-to-end behaviour, so we run them as
part of a full release sign-off. Follow the steps below to execute them locally.

## 1. Install dependencies

Create or activate your virtual environment and install the project in editable
mode with the optional extras that slow tests rely on:

```bash
pip install -e ".[dev,backtest,io,polars]"
```

The extras above pull in pandas, pyarrow, and other utilities that the slow
backtest and service suites need.

## 2. Run the slow checks without pytest

Some deployment environments forbid invoking `pytest` directly. The repository
includes a helper script that re-implements the behaviour of the slow-marked
tests using plain Python assertions so you can still verify the release build.

From the repository root run the script and let it progress through each check:

```bash
python scripts/run_slow_checks.py
```

Each section mirrors the matching pytest module:

- `backtest_sweep_*`: materialises two deterministic backtest runs for a small
  CSV dataset and writes the same equity/weights artefacts that the sweep tests
  assert on.
- `parallel_*`: exercises the multiprocessing path and the factor/slippage
  configuration using a trimmed optimizer (fewer ants and iterations) so the
  checks complete in well under a minute while still comparing the full result
  structures.
- `service_end_to_end`: boots the FastAPI app in memory using `TestClient` and
  exercises the authenticated download endpoints.

You can focus on an individual check by passing its name, e.g.

```bash
python scripts/run_slow_checks.py service_end_to_end
```

> **Tip for restricted environments:** If you cannot reach PyPI to install the
> optional dependencies, create an offline wheelhouse ahead of time (for
> example with `pip download --dest wheelhouse ...`) and set `PIP_FIND_LINKS`
> to that directory before running the `pip install -e ...` command above.
> This avoids mid-run failures when the tests import pandas, polars, or scipy.

## 3. (Optional) Run the entire suite including slow tests

If you would rather execute everything in one go before shipping but still need
to avoid pytest, the helper script without arguments covers the same scenarios
sequentially.

## 4. Troubleshooting tips

- If pandas is missing you will see skips or import errors. Re-run the install
  command above to ensure the `backtest` extra is present.
- The helper script logs a single `Iter 000` line from the optimized neuro-ant
  loop. A quick burst of three log entries per run is expected while the
  low-iteration optimizer searches the toy datasets.
- The service tests rely on the local FastAPI app in `src/service`. No external
  services are required, but if you have custom `RUNS_DIR` or
  `SERVICE_AUTH_TOKEN` environment variables set, clear them before running the
  suite.
- To reduce noise, the project's pytest configuration uses the quiet `-q`
  option. Remove it (e.g. `pytest -m slow -vv`) when debugging failures.

Following these steps ensures the slow regression coverage passes before the
final release commit.
