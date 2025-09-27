import argparse, time
import numpy as np
from pathlib import Path

from neuro_ant_optimizer.optimizer import (
    NeuroAntPortfolioOptimizer,
    OptimizerConfig,
    OptimizationObjective,
)
from neuro_ant_optimizer.utils import set_seed, nearest_psd, shrink_covariance


def mean_variance(mu, cov, wmin=0.0, wmax=1.0):
    # Closed-form unconstrained MV -> project to box + renormalize
    inv = np.linalg.pinv(cov)
    w = inv @ mu
    w = np.clip(w, wmin, wmax)
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w) / len(w)


def equal_weight(n):
    return np.ones(n) / n


def run_once(mu, cov, obj, seed=42, alpha=0.05):
    n = len(mu)
    cfg = OptimizerConfig(
        seed=seed,
        n_ants=24,
        max_iter=25,
        topk_refine=6,
        topk_train=6,
        cvar_alpha=alpha,
    )
    opt = NeuroAntPortfolioOptimizer(n, cfg)
    constraints = type(
        "C",
        (),
        dict(
            min_weight=0.0,
            max_weight=1.0,
            equality_enforce=True,
            leverage_limit=1.0,
            sector_map=None,
            max_sector_concentration=1.0,
            prev_weights=None,
            max_turnover=1.0,
        ),
    )()
    t0 = time.perf_counter()
    res = opt.optimize(mu, cov, constraints, objective=obj)
    dt = time.perf_counter() - t0
    return res, dt


def metrics(w, mu, cov, risk_free=0.02):
    er = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sr = (er - risk_free) / vol if vol > 1e-12 else 0.0
    return dict(expected_return=er, volatility=vol, sharpe=sr, l1=np.abs(w).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.05)  # for CVaR mode
    ap.add_argument("--out", type=str, default="bench_results.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # synthetic “returns” and PSD covariance (with shrinkage)
    mu = rng.normal(0.08, 0.06, size=args.n)
    A = rng.normal(size=(args.n, args.n))
    cov0 = 0.2 * (A @ A.T) / args.n
    cov = nearest_psd(shrink_covariance(cov0, 0.15))

    # baselines
    w_ew = equal_weight(args.n)
    w_mv = mean_variance(mu, cov)

    # neuro-ant: Sharpe & CVaR
    res_sr, dt_sr = run_once(mu, cov, OptimizationObjective.SHARPE_RATIO, seed=args.seed, alpha=args.alpha)
    res_cvar, dt_cvar = run_once(
        mu,
        cov,
        OptimizationObjective.MIN_CVAR,
        seed=args.seed,
        alpha=args.alpha,
    )

    rows = []
    rows.append(dict(model="equal_weight", **metrics(w_ew, mu, cov), time=0.0))
    rows.append(dict(model="mean_variance", **metrics(w_mv, mu, cov), time=0.0))
    rows.append(
        dict(
            model="neuro_ant_sharpe",
            **metrics(res_sr.weights, mu, cov),
            time=dt_sr,
        )
    )
    rows.append(
        dict(
            model="neuro_ant_cvar",
            **metrics(res_cvar.weights, mu, cov),
            time=dt_cvar,
        )
    )

    import csv

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", out_path)
if __name__ == "__main__":
    main()
