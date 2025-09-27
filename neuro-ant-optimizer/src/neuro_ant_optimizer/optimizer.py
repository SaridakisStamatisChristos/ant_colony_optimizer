from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import math
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .models import RiskAssessmentNetwork, PheromoneNetwork
from .colony import Ant, AntColony
from .constraints import PortfolioConstraints
from .refine import refine_slsqp
from .utils import nearest_psd, shrink_covariance, set_seed

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class OptimizerConfig:
    """Configuration container for :class:`NeuroAntPortfolioOptimizer`."""

    n_ants: int = 16
    max_iter: int = 20
    patience: int = 5
    seed: int = 42
    lr: float = 5e-4
    risk_free: float = 0.02
    evaporation: float = 0.45
    Q: float = 75.0
    topk_refine: int = 4
    topk_train: int = 4
    use_risk_head: bool = True
    refine_maxiter: int = 30
    grad_clip: float = 1.0
    min_alloc: float = 0.01
    base_alloc: float = 0.06
    risk_weight: float = 0.4
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    max_runtime: float = 2.0
    use_shrinkage: bool = True
    shrinkage_delta: float = 0.15
    cvar_alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.n_ants <= 0:
            raise ValueError("n_ants must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if not (0.0 <= self.evaporation <= 1.0):
            raise ValueError("evaporation must lie in [0, 1]")
        if self.topk_refine <= 0:
            raise ValueError("topk_refine must be positive")
        if self.topk_train <= 0:
            raise ValueError("topk_train must be positive")
        if self.min_alloc < 0.0 or self.base_alloc <= 0.0:
            raise ValueError("allocation hyper-parameters must be non-negative")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype instance")
        if self.max_runtime <= 0.0:
            raise ValueError("max_runtime must be positive")
        if not (0.0 <= self.shrinkage_delta <= 1.0):
            raise ValueError("shrinkage_delta must lie in [0, 1]")
        if not (0.0 < self.cvar_alpha < 0.5):
            raise ValueError("cvar_alpha must lie in (0, 0.5)")

    @classmethod
    def from_overrides(cls, overrides: Optional[Dict[str, Any]] = None) -> "OptimizerConfig":
        if overrides is None:
            return cls()
        base = asdict(cls())
        base.update(overrides)
        return cls(**base)


@dataclass
class OptimizationResult:
    """Structured result returned by :meth:`NeuroAntPortfolioOptimizer.optimize`."""

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    optimization_time: float
    convergence_status: bool
    iteration_count: int
    risk_contributions: np.ndarray
    message: str

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.risk_contributions = np.asarray(self.risk_contributions, dtype=float)


@dataclass
class BenchmarkStats:
    """Summary statistics describing a benchmark return series."""

    mean: float
    variance: float
    cov_vector: np.ndarray

    def __post_init__(self) -> None:
        self.cov_vector = np.asarray(self.cov_vector, dtype=float)
        if self.cov_vector.ndim != 1:
            raise ValueError("cov_vector must be one-dimensional")
        if self.variance < 0.0:
            raise ValueError("variance must be non-negative")


class OptimizationObjective(Enum):
    SHARPE_RATIO = "sharpe_ratio"
    MAX_RETURN = "max_return"
    MIN_VARIANCE = "min_variance"
    RISK_PARITY = "risk_parity"
    MIN_CVAR = "min_cvar"
    TRACKING_ERROR_MIN = "tracking_error_min"
    INFO_RATIO_MAX = "info_ratio_max"


class NeuroAntPortfolioOptimizer:
    """Hybrid ant-colony optimizer with neural pheromone and risk models."""

    def __init__(self, n_assets: int, config: Optional[Dict[str, Any] | OptimizerConfig] = None):
        if n_assets <= 1:
            raise ValueError("n_assets must be greater than one")

        if isinstance(config, OptimizerConfig):
            self.cfg = config
        else:
            self.cfg = OptimizerConfig.from_overrides(config)

        self.n_assets = n_assets
        self.device = torch.device(self.cfg.device)

        self.risk_net = RiskAssessmentNetwork(n_assets).to(self.device, dtype=self.cfg.dtype) if self.cfg.use_risk_head else None
        self.phero_net = PheromoneNetwork(n_assets).to(self.device, dtype=self.cfg.dtype)
        self.colony = AntColony(n_assets, evap=self.cfg.evaporation, Q=self.cfg.Q)

        if self.risk_net is not None:
            self.risk_optim = torch.optim.Adam(self.risk_net.parameters(), lr=self.cfg.lr)
        else:
            self.risk_optim = None

        self.mse = nn.MSELoss()
        # Stable policy trainer (KL + entropy)
        self.policy_trainer = PolicyTrainer(
            self.phero_net,
            device=self.device,
            dtype=self.cfg.dtype,
            lr=self.cfg.lr,
        )

        self.history: List[Dict[str, float]] = []
        self.best_w: Optional[np.ndarray] = None
        self.best_score: float = -np.inf

    def optimize(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        constraints: PortfolioConstraints,
        objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO,
        refine: bool = True,
        benchmark: Optional[BenchmarkStats] = None,
    ) -> OptimizationResult:
        """Run the optimization loop and return an :class:`OptimizationResult`."""

        set_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)
        start_time = perf_counter()

        mu = np.asarray(returns, dtype=float).ravel()
        cov_raw = np.asarray(covariance, dtype=float)
        if self.cfg.use_shrinkage:
            cov_raw = shrink_covariance(cov_raw, delta=self.cfg.shrinkage_delta)
        cov = nearest_psd(cov_raw)
        self._validate(mu, cov)

        sigma = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
        denom = np.outer(sigma, sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.divide(cov, denom, out=np.ones_like(cov), where=denom > 0)

        best_w = np.ones(self.n_assets) / self.n_assets
        best_score = -np.inf
        no_improve = 0
        converged = False
        message = "OK"

        for iteration in range(self.cfg.max_iter):
            ants = [Ant(self.n_assets) for _ in range(self.cfg.n_ants)]
            # Cache transition matrix once for this iteration
            with torch.no_grad():
                cached_T = (
                    self.phero_net.transition_matrix()
                    .detach()
                    .cpu()
                    .numpy()
                )
            portfolios: List[np.ndarray] = []
            scores: List[float] = []

            initial_states = rng.integers(0, self.n_assets, size=len(ants))
            for ant, start_node in zip(ants, initial_states):
                weights = ant.build(
                    self.phero_net,
                    self.risk_net,
                    alpha=1.0,
                    beta=self.cfg.risk_weight,
                    trans_matrix=cached_T,
                    rng=rng,
                    initial=int(start_node),
                )
                weights = self._apply_constraints(weights, constraints)
                score = self._score(
                    weights,
                    mu,
                    cov,
                    objective,
                    constraints,
                    benchmark=benchmark,
                )
                portfolios.append(weights)
                scores.append(score)

            if portfolios and refine:
                self._refine_topk(
                    portfolios,
                    scores,
                    mu,
                    cov,
                    objective,
                    constraints,
                    benchmark=benchmark,
                )

            self.colony.update_pheromone(ants, scores)

            if self.risk_net is not None:
                _ = self._train_risk(mu, sigma, corr)  # update risk model
            self._train_pheromone(portfolios, scores)

            if scores:
                best_idx = int(np.argmax(scores))
                current_best = float(scores[best_idx])
                if current_best > best_score + 1e-12:
                    best_score = current_best
                    best_w = portfolios[best_idx].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                self.history.append({
                    "iter": float(iteration),
                    "best": float(best_score),
                    "avg": float(np.mean(scores)),
                })

                if iteration % 10 == 0:
                    logger.info(
                        "Iter %03d | best=%.6f avg=%.6f",
                        iteration,
                        best_score,
                        float(np.mean(scores)),
                    )

                if no_improve >= self.cfg.patience:
                    converged = True
                    message = f"Early stop at iter {iteration}"
                    break
                if perf_counter() - start_time >= self.cfg.max_runtime:
                    message = f"Runtime budget reached at iter {iteration}"
                    break
            else:
                message = "No portfolios generated"
                break

        final_weights = self._apply_constraints(best_w, constraints)
        elapsed = perf_counter() - start_time

        return OptimizationResult(
            weights=final_weights,
            expected_return=float(final_weights @ mu),
            volatility=float(math.sqrt(max(final_weights @ cov @ final_weights, 0.0))),
            sharpe_ratio=self._sharpe(final_weights, mu, cov),
            optimization_time=elapsed,
            convergence_status=converged,
            iteration_count=len(self.history),
            risk_contributions=self._risk_contrib(final_weights, cov),
            message=message,
        )

    # ---- internals ----
    def _score(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        objective: OptimizationObjective,
        constraints: PortfolioConstraints,
        benchmark: Optional[BenchmarkStats] = None,
    ) -> float:
        if not self._feasible(weights, constraints):
            return -1e9

        if objective == OptimizationObjective.SHARPE_RATIO:
            return self._sharpe(weights, mu, cov)
        if objective == OptimizationObjective.MAX_RETURN:
            return float(weights @ mu)
        if objective == OptimizationObjective.MIN_VARIANCE:
            return -float(math.sqrt(max(weights @ cov @ weights, 0.0)))
        if objective == OptimizationObjective.RISK_PARITY:
            rc = self._risk_contrib(weights, cov)
            total = rc.sum()
            if total <= 0:
                return -1e6
            rc = rc / total
            equal = np.ones_like(rc) / len(rc)
            return -float(np.linalg.norm(rc - equal))
        if objective == OptimizationObjective.MIN_CVAR:
            cvar = self._cvar_normal(weights, mu, cov, self.cfg.cvar_alpha)
            return -cvar
        if objective == OptimizationObjective.TRACKING_ERROR_MIN:
            if benchmark is None:
                raise ValueError("Benchmark statistics required for tracking error objective")
            te = self._tracking_error(weights, mu, cov, benchmark)
            return -te
        if objective == OptimizationObjective.INFO_RATIO_MAX:
            if benchmark is None:
                raise ValueError("Benchmark statistics required for information ratio objective")
            return self._information_ratio(weights, mu, cov, benchmark)
        return self._sharpe(weights, mu, cov)

    def _apply_constraints(self, weights: np.ndarray, constraints: PortfolioConstraints) -> np.ndarray:
        clipped = np.clip(weights, constraints.min_weight, constraints.max_weight)

        if constraints.equality_enforce and abs(constraints.leverage_limit - 1.0) < 1e-12:
            total = clipped.sum()
            if total <= 0:
                clipped = np.ones_like(clipped) / len(clipped)
            else:
                clipped = clipped / total
            clipped = np.clip(clipped, constraints.min_weight, constraints.max_weight)
            clipped = clipped / max(clipped.sum(), 1e-12)
        elif clipped.sum() > constraints.leverage_limit:
            clipped *= constraints.leverage_limit / (clipped.sum() + 1e-12)

        if constraints.sector_map is not None:
            clipped = self._enforce_sector_caps(clipped, constraints)
        if constraints.factors_enabled():
            F = np.asarray(constraints.factor_loadings, dtype=float)
            target = np.asarray(constraints.factor_targets, dtype=float)
            A = F.T  # shape (K, N)
            residual = target - A @ clipped
            if np.linalg.norm(residual, ord=np.inf) > constraints.factor_tolerance:
                lam = 1e-6
                identity = np.eye(self.n_assets, dtype=float)
                if (
                    constraints.equality_enforce
                    and abs(constraints.leverage_limit - 1.0) < 1e-12
                ):
                    u = np.ones((self.n_assets, 1), dtype=float)
                    denom = float((u.T @ u).item())
                    if denom > 0:
                        projector = identity - (u @ u.T) / denom
                    else:
                        projector = identity
                    At = A @ projector
                    rhs = At.T @ residual
                    system = At.T @ At + lam * identity
                    try:
                        delta = projector @ np.linalg.solve(system, rhs)
                    except np.linalg.LinAlgError:
                        delta = projector @ np.linalg.lstsq(system, rhs, rcond=None)[0]
                else:
                    rhs = A.T @ residual
                    system = A.T @ A + lam * identity
                    try:
                        delta = np.linalg.solve(system, rhs)
                    except np.linalg.LinAlgError:
                        delta = np.linalg.lstsq(system, rhs, rcond=None)[0]
                clipped = np.clip(
                    clipped + delta,
                    constraints.min_weight,
                    constraints.max_weight,
                )
                total = clipped.sum()
                if (
                    constraints.equality_enforce
                    and abs(constraints.leverage_limit - 1.0) < 1e-12
                ):
                    if total > 0:
                        clipped = clipped / total
                elif total > constraints.leverage_limit:
                    clipped *= constraints.leverage_limit / (total + 1e-12)
                if constraints.sector_map is not None:
                    clipped = self._enforce_sector_caps(clipped, constraints)
        if constraints.prev_weights is not None:
            clipped = self._enforce_turnover(clipped, constraints)

        return clipped

    def _feasible(self, weights: np.ndarray, constraints: PortfolioConstraints, tol: float = 1e-8) -> bool:
        if np.any(weights < constraints.min_weight - tol) or np.any(weights > constraints.max_weight + tol):
            return False
        if weights.sum() > constraints.leverage_limit + tol:
            return False
        if (
            constraints.equality_enforce
            and abs(constraints.leverage_limit - 1.0) < 1e-12
            and abs(weights.sum() - 1.0) > 1e-6
        ):
            return False
        if constraints.sector_map is not None:
            sectors = np.array(constraints.sector_map, dtype=int)
            for sector in np.unique(sectors):
                if weights[sectors == sector].sum() > constraints.max_sector_concentration + tol:
                    return False
        if constraints.factors_enabled():
            F = np.asarray(constraints.factor_loadings, dtype=float)
            target = np.asarray(constraints.factor_targets, dtype=float)
            diff = F.T @ weights - target
            if np.linalg.norm(diff, ord=np.inf) > constraints.factor_tolerance + tol:
                return False
        if constraints.prev_weights is not None:
            if np.abs(weights - constraints.prev_weights).sum() > constraints.max_turnover + tol:
                return False
        return True

    def _enforce_sector_caps(self, weights: np.ndarray, constraints: PortfolioConstraints) -> np.ndarray:
        sectors = np.array(constraints.sector_map, dtype=int)
        adjusted = weights.copy()
        for sector in np.unique(sectors):
            mask = sectors == sector
            total = adjusted[mask].sum()
            cap = constraints.max_sector_concentration
            if total > cap:
                adjusted[mask] *= cap / (total + 1e-12)
        if constraints.equality_enforce and abs(constraints.leverage_limit - 1.0) < 1e-12:
            total = adjusted.sum()
            if total > 0:
                adjusted = adjusted / total
        else:
            adjusted = np.minimum(adjusted, constraints.max_weight)
        return adjusted

    def _enforce_turnover(self, weights: np.ndarray, constraints: PortfolioConstraints) -> np.ndarray:
        previous = np.asarray(constraints.prev_weights, dtype=float)
        diff = weights - previous
        l1_norm = np.abs(diff).sum()
        if l1_norm <= constraints.max_turnover + 1e-12:
            adjusted = weights
        else:
            alpha = constraints.max_turnover / (l1_norm + 1e-12)
            adjusted = previous + alpha * diff

        adjusted = np.clip(adjusted, constraints.min_weight, constraints.max_weight)

        if constraints.equality_enforce and abs(constraints.leverage_limit - 1.0) < 1e-12:
            total = adjusted.sum()
            if total > 0:
                adjusted = adjusted / total
        elif adjusted.sum() > constraints.leverage_limit:
            adjusted *= constraints.leverage_limit / (adjusted.sum() + 1e-12)
        return adjusted

    def _risk_contrib(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        variance = float(weights @ cov @ weights)
        if variance <= 1e-18:
            return np.zeros_like(weights)
        marginal = cov @ weights
        return weights * marginal / variance

    def _sharpe(self, weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
        volatility = float(math.sqrt(max(weights @ cov @ weights, 0.0)))
        if volatility <= 1e-12:
            return 0.0
        return (float(weights @ mu) - self.cfg.risk_free) / volatility

    def _cvar_normal(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        alpha: float,
    ) -> float:
        from math import sqrt
        from statistics import NormalDist

        mean_loss = -float(weights @ mu)
        variance = float(weights @ cov @ weights)
        std_loss = sqrt(max(variance, 0.0))
        alpha = float(np.clip(alpha, 1e-6, 0.5))
        nd = NormalDist()
        z = nd.inv_cdf(alpha)
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return mean_loss + std_loss * (phi / alpha)

    def _tracking_error(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        benchmark: BenchmarkStats,
    ) -> float:
        if benchmark.cov_vector.shape[0] != weights.shape[0]:
            raise ValueError("Benchmark covariance vector dimension mismatch")
        active_variance = float(
            weights @ cov @ weights
            + benchmark.variance
            - 2.0 * (weights @ benchmark.cov_vector)
        )
        if active_variance < 0.0 and active_variance > -1e-12:
            active_variance = 0.0
        return math.sqrt(max(active_variance, 0.0))

    def _information_ratio(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        benchmark: BenchmarkStats,
    ) -> float:
        active_return = float(weights @ mu - benchmark.mean)
        te = self._tracking_error(weights, mu, cov, benchmark)
        if te <= 1e-12:
            if abs(active_return) <= 1e-12:
                return 0.0
            return math.copysign(1e6, active_return)
        return active_return / te

    def _validate(self, mu: np.ndarray, cov: np.ndarray) -> None:
        n = mu.shape[0]
        if cov.shape != (n, n):
            raise ValueError("covariance must be (n,n)")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("covariance must be symmetric")

    def _refine_topk(
        self,
        portfolios: List[np.ndarray],
        scores: List[float],
        mu: np.ndarray,
        cov: np.ndarray,
        objective: OptimizationObjective,
        constraints: PortfolioConstraints,
        benchmark: Optional[BenchmarkStats] = None,
    ) -> None:
        if not portfolios:
            return

        topk = min(self.cfg.topk_refine, len(portfolios))
        if topk <= 0:
            return

        indices = np.argsort(scores)[-topk:]
        bounds = [(constraints.min_weight, constraints.max_weight)] * self.n_assets
        Aeq_rows: List[np.ndarray] = []
        beq_vals: List[np.ndarray] = []
        Aineq_rows: List[np.ndarray] = []
        bineq_vals: List[float] = []
        if (
            constraints.equality_enforce
            and abs(constraints.leverage_limit - 1.0) < 1e-12
        ):
            Aeq_rows.append(np.ones((1, self.n_assets), dtype=float))
            beq_vals.append(np.array([1.0], dtype=float))
        if constraints.factors_enabled():
            F = np.asarray(constraints.factor_loadings, dtype=float)
            b = np.asarray(constraints.factor_targets, dtype=float)
            Aeq_rows.append(F.T.astype(float))
            beq_vals.append(b.astype(float))
        if constraints.sector_map is not None:
            sectors = np.asarray(constraints.sector_map, dtype=int)
            cap = float(constraints.max_sector_concentration)
            for sector in np.unique(sectors):
                row = np.zeros(self.n_assets, dtype=float)
                row[sectors == sector] = 1.0
                Aineq_rows.append(row)
                bineq_vals.append(cap)
        Aeq = (
            np.vstack(Aeq_rows) if Aeq_rows else None
        )
        beq = (
            np.concatenate(beq_vals) if beq_vals else None
        )
        Aineq = np.vstack(Aineq_rows) if Aineq_rows else None
        bineq = np.asarray(bineq_vals, dtype=float) if bineq_vals else None
        prev = (
            np.asarray(constraints.prev_weights, dtype=float)
            if constraints.prev_weights is not None
            else None
        )
        T = float(constraints.max_turnover) if prev is not None else 0.0

        for idx in indices:
            initial = portfolios[idx]

            def score_fn(weights: Sequence[float]) -> float:
                return self._score(
                    np.asarray(weights, dtype=float),
                    mu,
                    cov,
                    objective,
                    constraints,
                    benchmark=benchmark,
                )

            refined, res = refine_slsqp(
                score_fn,
                initial,
                bounds,
                Aeq=Aeq,
                beq=beq,
                Aineq=Aineq,
                bineq=bineq,
                prev=prev,
                T=T,
            )
            if res.success:
                portfolios[idx] = refined
                scores[idx] = score_fn(refined)

    # ---- learning routines ----
    def _train_risk(self, mu: np.ndarray, sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """One-step self-supervised update of risk_net and return fresh scores."""
        if self.risk_net is None or self.risk_optim is None:
            return np.clip(sigma / (sigma.max() + 1e-12), 0, 1)

        target_vec = (sigma / (sigma.max() + 1e-12)).astype(np.float32)
        basis = torch.eye(self.n_assets, device=self.device, dtype=self.cfg.dtype)
        target = torch.diag(torch.from_numpy(target_vec)).to(self.device, dtype=self.cfg.dtype)

        self.risk_net.train()
        self.risk_optim.zero_grad()
        pred = self.risk_net(basis)
        loss = self.mse(pred, target)
        loss.backward()
        clip_grad_norm_(self.risk_net.parameters(), self.cfg.grad_clip)
        self.risk_optim.step()
        self.risk_net.eval()

        with torch.no_grad():
            out = self.risk_net(basis).detach().cpu().numpy()
        return np.clip(np.diag(out), 0.0, 1.0)

    def _train_pheromone(self, portfolios: List[np.ndarray], scores: List[float]) -> None:
        if not portfolios:
            return

        k = min(self.cfg.topk_train, len(portfolios))
        idx = np.argsort(scores)[-k:]
        # Build a small batch of (N,N) targets from top portfolios
        targets = np.stack(
            [np.tile(np.clip(portfolios[i], 0.0, 1.0), (self.n_assets, 1)) for i in idx],
            axis=0,
        ).astype(np.float32)
        # One stable trainer step (KL to EMA + entropy bonus)
        _ = self.policy_trainer.step(targets)


class PolicyTrainer:
    """
    Stable trainer for the pheromone policy: KL(pred || EMA(target)) + entropy bonus.
    Keeps your net/API intact; call .step(batch_of_target_mats).
    """

    def __init__(
        self,
        phero_net,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        temperature: float = 0.7,
        entropy_coeff: float = 1e-3,
        lr: float = 3e-4,
    ) -> None:
        self.net = phero_net
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.tau = float(temperature)
        self.entropy_coeff = float(entropy_coeff)
        self._target_ema: Optional[torch.Tensor] = None
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def step(self, target_mats_np: np.ndarray) -> float:
        """
        target_mats_np: (B, N, N) stochastic matrices; we fit KL(pred || EMA(target)).
        Returns scalar loss.
        """

        self.net.train()
        with torch.no_grad():
            tgt = torch.from_numpy(
                np.clip(target_mats_np.mean(axis=0), 1e-6, 1 - 1e-6)
            ).to(self.device, self.dtype)
            tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            if self._target_ema is None:
                self._target_ema = tgt.clone()
            else:
                self._target_ema.mul_(0.9).add_(0.1 * tgt)

        pred = torch.clamp(self.net.transition_matrix(), 1e-6, 1 - 1e-6)
        pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        log_pred = torch.log(pred)
        kl = F.kl_div(log_pred, self._target_ema, reduction="batchmean")
        ent = -(pred * log_pred).sum() / pred.numel()
        loss = kl - self.entropy_coeff * ent
        self.opt.zero_grad()
        loss.backward()
        clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach().cpu().item())
