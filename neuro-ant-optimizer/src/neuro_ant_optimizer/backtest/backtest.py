"""Walk-forward backtesting utilities built around the neuro-ant optimizer."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.optimizer import (
    NeuroAntPortfolioOptimizer,
    OptimizationObjective,
    OptimizerConfig,
)
from neuro_ant_optimizer.utils import nearest_psd, set_seed, shrink_covariance

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    pd = None  # type: ignore


def ewma_cov(returns: np.ndarray, span: int = 60) -> np.ndarray:
    """Compute an exponentially weighted covariance matrix."""

    if span <= 1:
        raise ValueError("span must be greater than one for EWMA covariance")

    lam = max(1.0 - 2.0 / (1.0 + span), 0.0)
    demeaned = returns - returns.mean(axis=0, keepdims=True)
    cov = np.zeros((demeaned.shape[1], demeaned.shape[1]), dtype=float)
    for row in demeaned:
        cov = lam * cov + (1.0 - lam) * np.outer(row, row)
    cov = 0.5 * (cov + cov.T)
    return cov


def turnover(previous: Optional[np.ndarray], current: np.ndarray) -> float:
    """Compute the L1 turnover between two weight vectors."""

    if previous is None:
        return float(np.abs(current).sum())
    return float(np.abs(current - previous).sum())


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Return the maximum drawdown for the supplied equity curve."""

    equity = np.asarray(equity_curve, dtype=float)
    if equity.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - np.divide(
        equity,
        running_peak,
        out=np.ones_like(equity),
        where=running_peak > 0,
    )
    return float(np.max(drawdown))


def _build_optimizer(n_assets: int, seed: int) -> NeuroAntPortfolioOptimizer:
    cfg = OptimizerConfig(
        n_ants=24,
        max_iter=25,
        topk_refine=6,
        topk_train=6,
        use_shrinkage=False,
        shrinkage_delta=0.15,
        cvar_alpha=0.05,
        seed=seed,
    )
    return NeuroAntPortfolioOptimizer(n_assets, cfg)


def _build_constraints(n_assets: int) -> PortfolioConstraints:
    return PortfolioConstraints(
        min_weight=0.0,
        max_weight=1.0,
        equality_enforce=True,
        leverage_limit=1.0,
        sector_map=None,
        max_sector_concentration=1.0,
        prev_weights=None,
        max_turnover=1.0,
    )


def _frame_to_numpy(frame: Any) -> np.ndarray:
    if hasattr(frame, "to_numpy"):
        return frame.to_numpy(dtype=float)  # type: ignore[no-any-return]
    return np.asarray(frame, dtype=float)


def _frame_index(frame: Any, length: int) -> List[Any]:
    if hasattr(frame, "index"):
        return list(frame.index)  # type: ignore[attr-defined]
    return list(range(length))


_OBJECTIVE_MAP: Dict[str, OptimizationObjective] = {
    "sharpe": OptimizationObjective.SHARPE_RATIO,
    "max_return": OptimizationObjective.MAX_RETURN,
    "min_variance": OptimizationObjective.MIN_VARIANCE,
    "risk_parity": OptimizationObjective.RISK_PARITY,
    "min_cvar": OptimizationObjective.MIN_CVAR,
}


def backtest(
    df: Any,
    lookback: int = 252,
    step: int = 21,
    ewma_span: Optional[int] = None,
    objective: str = "sharpe",
    seed: int = 7,
) -> Dict[str, Any]:
    """Run a rolling-window backtest on a return dataframe."""

    if lookback <= 0 or step <= 0:
        raise ValueError("lookback and step must be positive integers")
    if objective not in _OBJECTIVE_MAP:
        raise ValueError(f"Unknown objective '{objective}'")

    returns = _frame_to_numpy(df)
    if returns.size == 0:
        raise ValueError("input dataframe must contain returns")

    set_seed(seed)
    n_periods, n_assets = returns.shape
    dates = _frame_index(df, n_periods)

    optimizer = _build_optimizer(n_assets, seed)
    constraints = _build_constraints(n_assets)

    weights: List[np.ndarray] = []
    realized_returns: List[float] = []
    realized_dates: List[Any] = []
    turnovers: List[float] = []
    prev_weights: Optional[np.ndarray] = None

    for start in range(lookback, n_periods, step):
        end = min(start + step, n_periods)
        train = returns[start - lookback : start]
        test = returns[start:end]
        mu = train.mean(axis=0)
        if ewma_span is not None:
            cov_raw = ewma_cov(train, span=ewma_span)
        else:
            cov_raw = np.cov(train, rowvar=False)
        if optimizer.cfg.use_shrinkage:
            cov_raw = shrink_covariance(cov_raw, delta=optimizer.cfg.shrinkage_delta)
        cov = nearest_psd(cov_raw)
        result = optimizer.optimize(
            mu,
            cov,
            constraints,
            objective=_OBJECTIVE_MAP[objective],
        )
        w = result.weights
        weights.append(w)
        block_returns = test @ w
        realized_returns.extend(block_returns.tolist())
        turnovers.append(turnover(prev_weights, w))
        prev_weights = w
        realized_dates.extend(dates[start:end])

    realized_returns_arr = np.asarray(realized_returns, dtype=float)
    equity = np.cumprod(1.0 + realized_returns_arr)

    ann_vol = float(np.std(realized_returns_arr) * math.sqrt(252)) if realized_returns_arr.size else 0.0
    ann_return = float(np.mean(realized_returns_arr) * 252) if realized_returns_arr.size else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0
    mdd = max_drawdown(equity)
    avg_turn = float(np.mean(turnovers)) if turnovers else 0.0

    return {
        "dates": realized_dates,
        "returns": realized_returns_arr,
        "equity": equity,
        "weights": np.asarray(weights),
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "avg_turnover": avg_turn,
    }


def _write_metrics(metrics_path: Path, results: Dict[str, Any]) -> None:
    with metrics_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key in ["sharpe", "ann_return", "ann_vol", "max_drawdown", "avg_turnover"]:
            writer.writerow([key, results[key]])


def _write_equity(equity_path: Path, results: Dict[str, Any]) -> None:
    if pd is not None:
        frame = pd.DataFrame(
            {
                "date": results["dates"],
                "equity": results["equity"],
                "ret": results["returns"],
            }
        )
        frame.to_csv(equity_path, index=False)
        return

    data = np.column_stack([
        np.asarray(results["dates"], dtype=str),
        np.asarray(results["equity"], dtype=float),
        np.asarray(results["returns"], dtype=float),
    ])
    header = "date,equity,ret"
    np.savetxt(equity_path, data, fmt="%s", delimiter=",", header=header, comments="")


def _read_csv(csv_path: Path):
    if pd is not None:
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=str)
    if raw.size == 0:
        values = np.empty((0, 0), dtype=float)
        dates: Sequence[str] = []
    else:
        raw = np.atleast_2d(raw)
        dates = raw[:, 0]
        values = raw[:, 1:].astype(float)

    class _Frame:
        def __init__(self, arr: np.ndarray, idx: Sequence[str]):
            self._arr = arr
            self._idx = [np.datetime64(d) for d in idx]

        def to_numpy(self, dtype=float):
            return self._arr.astype(dtype)

        @property
        def index(self):
            return self._idx

    return _Frame(values, dates)


def main(args: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV of daily returns with date index")
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--ewma_span", type=int, default=60)
    parser.add_argument(
        "--objective",
        choices=sorted(_OBJECTIVE_MAP.keys()),
        default="sharpe",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="bt_out")
    parsed = parser.parse_args(args=args)

    df = _read_csv(Path(parsed.csv))
    results = backtest(
        df,
        lookback=parsed.lookback,
        step=parsed.step,
        ewma_span=parsed.ewma_span,
        objective=parsed.objective,
        seed=parsed.seed,
    )

    out_dir = Path(parsed.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir / "metrics.csv", results)
    _write_equity(out_dir / "equity.csv", results)

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.plot(results["dates"], results["equity"])
        plt.title(f"Equity — {parsed.objective}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(out_dir / "equity.png", dpi=160)
        plt.close()
    except Exception:
        pass

    print(f"Wrote {out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
