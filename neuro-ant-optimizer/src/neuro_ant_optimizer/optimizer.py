from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import math
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    te_target: float = 0.0
    lambda_te: float = 0.0
    gamma_turnover: float = 0.0

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
        if self.te_target < 0.0:
            raise ValueError("te_target must be non-negative")
        if self.lambda_te < 0.0:
            raise ValueError("lambda_te must be non-negative")
        if self.gamma_turnover < 0.0:
            raise ValueError("gamma_turnover must be non-negative")

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
    feasible: bool
    projection_iterations: int

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.risk_contributions = np.asarray(self.risk_contributions, dtype=float)
        self.feasible = bool(self.feasible)
        self.projection_iterations = int(self.projection_iterations)


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


@dataclass
class ConstraintWorkspace:
    """Cache of constraint-aligned arrays reused across projections/refinement."""

    lower: np.ndarray
    upper: np.ndarray
    benchmark: Optional[np.ndarray]
    active_group_bounds: Dict[int, Tuple[float, float]]
    active_group_masks: Dict[int, np.ndarray]
    active_group_rows: Dict[int, np.ndarray]
    factor_T: Optional[np.ndarray]
    factor_targets: Optional[np.ndarray]
    factor_lower: Optional[np.ndarray]
    factor_upper: Optional[np.ndarray]
    has_factor_equality: bool
    has_factor_bounds: bool
    factor_tolerance: float


class OptimizationObjective(Enum):
    SHARPE_RATIO = "sharpe_ratio"
    MAX_RETURN = "max_return"
    MIN_VARIANCE = "min_variance"
    RISK_PARITY = "risk_parity"
    MIN_CVAR = "min_cvar"
    TRACKING_ERROR_MIN = "tracking_error_min"
    INFO_RATIO_MAX = "info_ratio_max"
    TRACKING_ERROR_TARGET = "tracking_error_target"
    MULTI_TERM = "multi_term"


ObjectiveFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, PortfolioConstraints, Optional[BenchmarkStats]],
    float,
]
ObjectiveSpec = Union[OptimizationObjective, ObjectiveFn]


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
        self._last_projection_iterations: int = 0

    def optimize(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        constraints: PortfolioConstraints,
        objective: ObjectiveSpec = OptimizationObjective.SHARPE_RATIO,
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

        workspace = self._build_constraint_workspace(constraints)

        sigma = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
        denom = np.outer(sigma, sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.divide(cov, denom, out=np.ones_like(cov), where=denom > 0)

        best_w = np.ones(self.n_assets) / self.n_assets
        best_score = -np.inf
        no_improve = 0
        converged = False
        message = "OK"

        alpha = 1.0
        beta = float(self.cfg.risk_weight)

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
            trans_log = np.log(np.clip(cached_T, 1e-12, None)) * float(alpha)
            if self.risk_net is not None and beta:
                with torch.no_grad():
                    identity = torch.eye(
                        self.n_assets,
                        device=self.risk_net.param_device,
                        dtype=self.risk_net.param_dtype,
                    )
                    diag = (
                        torch.diagonal(self.risk_net(identity), dim1=-2, dim2=-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                risk_bias = np.log(np.clip(diag, 1e-6, None)) * beta
            else:
                risk_bias = np.zeros(self.n_assets, dtype=float)
            portfolios: List[np.ndarray] = []
            scores: List[float] = []

            initial_states = rng.integers(0, self.n_assets, size=len(ants))
            for ant, start_node in zip(ants, initial_states):
                weights = ant.build(
                    self.phero_net,
                    self.risk_net,
                    alpha=alpha,
                    beta=beta,
                    trans_matrix=cached_T,
                    trans_log=trans_log,
                    risk_bias=risk_bias,
                    rng=rng,
                    initial=int(start_node),
                )
                weights = self._apply_constraints(weights, constraints, workspace)
                score = self._score(
                    weights,
                    mu,
                    cov,
                    objective,
                    constraints,
                    benchmark=benchmark,
                    workspace=workspace,
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
                    workspace=workspace,
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

        final_weights = self._apply_constraints(best_w, constraints, workspace)
        feasible_flag = self._feasible(final_weights, constraints, workspace=workspace)
        projection_steps = int(self._last_projection_iterations)
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
            feasible=feasible_flag,
            projection_iterations=projection_steps,
        )

    # ---- internals ----
    def _score(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        objective: ObjectiveSpec,
        constraints: PortfolioConstraints,
        benchmark: Optional[BenchmarkStats] = None,
        workspace: Optional[ConstraintWorkspace] = None,
    ) -> float:
        if not self._feasible(weights, constraints, workspace=workspace):
            return -1e9

        if callable(objective):
            try:
                return float(
                    objective(
                        weights,
                        mu,
                        cov,
                        constraints,
                        benchmark,
                        workspace=workspace,
                    )
                )
            except TypeError:
                return float(objective(weights, mu, cov, constraints, benchmark))

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
        if objective == OptimizationObjective.TRACKING_ERROR_TARGET:
            if benchmark is None:
                return -1e6
            te = self._tracking_error(weights, mu, cov, benchmark)
            target = float(self.cfg.te_target)
            return -float((te - target) ** 2)
        if objective == OptimizationObjective.MULTI_TERM:
            if benchmark is None:
                return -1e6
            base = self._information_ratio(weights, mu, cov, benchmark)
            if self.cfg.lambda_te > 0.0:
                te = self._tracking_error(weights, mu, cov, benchmark)
                base -= float(self.cfg.lambda_te) * te
            if (
                self.cfg.gamma_turnover > 0.0
                and constraints.prev_weights is not None
            ):
                prev = np.asarray(constraints.prev_weights, dtype=float)
                turnover = float(np.abs(weights - prev).sum())
                base -= float(self.cfg.gamma_turnover) * turnover
            return float(base)
        return self._sharpe(weights, mu, cov)

    def _build_constraint_workspace(
        self, constraints: PortfolioConstraints
    ) -> ConstraintWorkspace:
        lower = np.full(self.n_assets, constraints.min_weight, dtype=float)
        upper = np.full(self.n_assets, constraints.max_weight, dtype=float)

        bench_vec: Optional[np.ndarray]
        if constraints.benchmark_weights is not None:
            bench = np.asarray(constraints.benchmark_weights, dtype=float).ravel()
            if bench.shape[0] != self.n_assets:
                raise ValueError("benchmark_weights dimension mismatch")
            if constraints.benchmark_mask is not None:
                mask = np.asarray(constraints.benchmark_mask, dtype=bool).ravel()
                if mask.shape[0] != self.n_assets:
                    raise ValueError("benchmark_mask dimension mismatch")
            else:
                mask = np.ones(self.n_assets, dtype=bool)
            bench_vec = np.where(mask, bench, 0.0)
            min_active = float(constraints.min_active_weight)
            max_active = float(constraints.max_active_weight)
            if np.isfinite(min_active):
                lower = np.where(
                    mask,
                    np.maximum(lower, bench_vec + min_active),
                    lower,
                )
            if np.isfinite(max_active):
                upper = np.where(
                    mask,
                    np.minimum(upper, bench_vec + max_active),
                    upper,
                )
        else:
            bench_vec = None

        lower = np.minimum(lower, upper)

        active_group_bounds = dict(constraints.active_group_bounds or {})
        active_group_masks: Dict[int, np.ndarray] = {}
        active_group_rows: Dict[int, np.ndarray] = {}
        if constraints.active_group_map is not None and active_group_bounds:
            groups = np.asarray(constraints.active_group_map, dtype=int).ravel()
            if groups.shape[0] != self.n_assets:
                raise ValueError("active_group_map dimension mismatch")
            for gid, bound in active_group_bounds.items():
                mask = groups == gid
                if not np.any(mask):
                    continue
                active_group_masks[gid] = mask
                row = np.zeros(self.n_assets, dtype=float)
                row[mask] = 1.0
                active_group_rows[gid] = row

        factor_T: Optional[np.ndarray] = None
        factor_targets: Optional[np.ndarray] = None
        factor_lower: Optional[np.ndarray] = None
        factor_upper: Optional[np.ndarray] = None
        has_factor_equality = False
        has_factor_bounds = False
        if constraints.factor_loadings is not None:
            F = np.asarray(constraints.factor_loadings, dtype=float)
            if F.ndim == 2 and F.shape[0] == self.n_assets:
                factor_T = F.T.copy()
                if constraints.factor_targets is not None:
                    b = np.asarray(constraints.factor_targets, dtype=float).ravel()
                    if b.shape[0] == factor_T.shape[0]:
                        factor_targets = b
                        has_factor_equality = True
                if constraints.factor_lower_bounds is not None:
                    lower_arr = np.asarray(constraints.factor_lower_bounds, dtype=float).ravel()
                    if lower_arr.shape[0] == factor_T.shape[0]:
                        factor_lower = lower_arr
                        has_factor_bounds = True
                if constraints.factor_upper_bounds is not None:
                    upper_arr = np.asarray(constraints.factor_upper_bounds, dtype=float).ravel()
                    if upper_arr.shape[0] == factor_T.shape[0]:
                        factor_upper = upper_arr
                        has_factor_bounds = True

        return ConstraintWorkspace(
            lower=lower,
            upper=upper,
            benchmark=bench_vec,
            active_group_bounds=active_group_bounds,
            active_group_masks=active_group_masks,
            active_group_rows=active_group_rows,
            factor_T=factor_T,
            factor_targets=factor_targets,
            factor_lower=factor_lower,
            factor_upper=factor_upper,
            has_factor_equality=has_factor_equality,
            has_factor_bounds=has_factor_bounds,
            factor_tolerance=float(constraints.factor_tolerance),
        )

    def _apply_constraints(
        self,
        weights: np.ndarray,
        constraints: PortfolioConstraints,
        workspace: Optional[ConstraintWorkspace] = None,
    ) -> np.ndarray:
        workspace = workspace or self._build_constraint_workspace(constraints)
        lower = workspace.lower
        upper = workspace.upper
        bench = workspace.benchmark
        projection_iters = 0

        def project_leverage(vec: np.ndarray) -> np.ndarray:
            clipped = np.clip(vec, lower, upper)
            if (
                constraints.equality_enforce
                and abs(constraints.leverage_limit - 1.0) < 1e-12
            ):
                return self._project_sum_with_bounds(clipped, lower, upper, 1.0)
            limit = float(constraints.leverage_limit)
            if clipped.sum() > limit:
                return self._project_sum_with_bounds(clipped, lower, upper, limit)
            return clipped

        adjusted = project_leverage(weights)

        if constraints.sector_map is not None:
            adjusted = project_leverage(
                self._enforce_sector_caps(adjusted, constraints)
            )

        def _enforce_active(adjusted_weights: np.ndarray) -> Tuple[np.ndarray, int]:
            if bench is None or not workspace.active_group_masks:
                return adjusted_weights, 0
            steps = 0
            updated = adjusted_weights
            for _ in range(5):
                steps += 1
                updated = project_leverage(
                    self._enforce_active_groups(
                        updated,
                        bench,
                        constraints,
                        lower,
                        upper,
                        workspace,
                    )
                )
                if self._active_groups_feasible(
                    updated, bench, constraints, workspace
                ):
                    break
            return updated, steps

        def _enforce_factors(adjusted_weights: np.ndarray) -> Tuple[np.ndarray, int]:
            if not (workspace.has_factor_equality or workspace.has_factor_bounds):
                return adjusted_weights, 0
            steps = 0
            updated = adjusted_weights
            for _ in range(12):
                steps += 1
                updated = project_leverage(
                    self._enforce_factor_constraints(
                        updated, constraints, lower, upper, workspace
                    )
                )
                if self._factor_constraints_satisfied(
                    updated, constraints, workspace, tol=0.0
                ):
                    break
            return updated, steps

        adjusted, steps = _enforce_active(adjusted)
        projection_iters += steps
        adjusted, steps = _enforce_factors(adjusted)
        projection_iters += steps

        if constraints.prev_weights is not None:
            adjusted = project_leverage(
                self._enforce_turnover(adjusted, constraints, lower, upper)
            )
            adjusted, steps = _enforce_active(adjusted)
            projection_iters += steps
            adjusted, steps = _enforce_factors(adjusted)
            projection_iters += steps

        adjusted = project_leverage(adjusted)
        adjusted = np.clip(adjusted, lower, upper)
        self._last_projection_iterations = projection_iters
        return adjusted

    def _feasible(
        self,
        weights: np.ndarray,
        constraints: PortfolioConstraints,
        tol: float = 1e-8,
        workspace: Optional[ConstraintWorkspace] = None,
    ) -> bool:
        workspace = workspace or self._build_constraint_workspace(constraints)
        lower, upper, bench = self._compute_weight_bounds(constraints, workspace)
        if np.any(weights < lower - tol) or np.any(weights > upper + tol):
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
        if bench is not None and not self._active_groups_feasible(
            weights, bench, constraints, workspace, tol=tol
        ):
            return False
        if not self._factor_constraints_satisfied(
            weights, constraints, workspace, tol=tol
        ):
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

    def _enforce_turnover(
        self,
        weights: np.ndarray,
        constraints: PortfolioConstraints,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        previous = np.asarray(constraints.prev_weights, dtype=float)
        diff = weights - previous
        l1_norm = np.abs(diff).sum()
        if l1_norm <= constraints.max_turnover + 1e-12:
            adjusted = weights
        else:
            alpha = constraints.max_turnover / (l1_norm + 1e-12)
            adjusted = previous + alpha * diff

        lb = lower if lower is not None else np.full(self.n_assets, constraints.min_weight)
        ub = upper if upper is not None else np.full(self.n_assets, constraints.max_weight)
        adjusted = np.clip(adjusted, lb, ub)

        if constraints.equality_enforce and abs(constraints.leverage_limit - 1.0) < 1e-12:
            total = adjusted.sum()
            if total > 0:
                adjusted = adjusted / total
        elif adjusted.sum() > constraints.leverage_limit:
            adjusted *= constraints.leverage_limit / (adjusted.sum() + 1e-12)
        return adjusted

    def _compute_weight_bounds(
        self,
        constraints: PortfolioConstraints,
        workspace: Optional[ConstraintWorkspace] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        workspace = workspace or self._build_constraint_workspace(constraints)
        lower = workspace.lower.copy()
        upper = workspace.upper.copy()
        bench_vec = (
            workspace.benchmark.copy()
            if workspace.benchmark is not None
            else None
        )
        return lower, upper, bench_vec

    def _project_sum_with_bounds(
        self,
        weights: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        target: float,
        tol: float = 1e-10,
    ) -> np.ndarray:
        projected = np.clip(weights, lower, upper)
        if not np.isfinite(target):
            return projected
        for _ in range(5 * self.n_assets):
            diff = target - projected.sum()
            if abs(diff) <= tol:
                break
            if diff > 0:
                slack = np.clip(upper - projected, 0.0, None)
                total_slack = slack.sum()
                if total_slack <= tol:
                    break
                step = np.minimum(slack, diff * slack / max(total_slack, 1e-12))
                projected = np.clip(projected + step, lower, upper)
            else:
                slack = np.clip(projected - lower, 0.0, None)
                total_slack = slack.sum()
                if total_slack <= tol:
                    break
                step = np.minimum(slack, (-diff) * slack / max(total_slack, 1e-12))
                projected = np.clip(projected - step, lower, upper)
        return np.clip(projected, lower, upper)

    def _enforce_active_groups(
        self,
        weights: np.ndarray,
        benchmark: np.ndarray,
        constraints: PortfolioConstraints,
        lower: np.ndarray,
        upper: np.ndarray,
        workspace: ConstraintWorkspace,
    ) -> np.ndarray:
        adjusted = weights.copy()
        if not workspace.active_group_bounds:
            return np.clip(adjusted, lower, upper)
        for gid, bound in workspace.active_group_bounds.items():
            mask = workspace.active_group_masks.get(gid)
            if mask is None or not np.any(mask):
                continue
            lower_b, upper_b = bound
            bench_sum = float(benchmark[mask].sum())
            current = float(adjusted[mask].sum())
            if np.isfinite(upper_b):
                target = max(bench_sum + float(upper_b), 0.0)
                if current > target + 1e-12:
                    adjusted = self._set_group_sum(adjusted, mask, target, lower, upper, benchmark)
                    current = float(adjusted[mask].sum())
            if np.isfinite(lower_b):
                target = max(bench_sum + float(lower_b), 0.0)
                current = float(adjusted[mask].sum())
                if current < target - 1e-12:
                    adjusted = self._set_group_sum(adjusted, mask, target, lower, upper, benchmark)
            if (
                constraints.equality_enforce
                and abs(constraints.leverage_limit - 1.0) < 1e-12
            ):
                adjusted = self._redistribute_outside_group(
                    adjusted,
                    mask,
                    target_total=1.0,
                    lower=lower,
                    upper=upper,
                )
        return np.clip(adjusted, lower, upper)

    def _active_groups_feasible(
        self,
        weights: np.ndarray,
        benchmark: np.ndarray,
        constraints: PortfolioConstraints,
        workspace: ConstraintWorkspace,
        tol: float = 1e-8,
    ) -> bool:
        if not workspace.active_group_bounds:
            return True
        for gid, bound in workspace.active_group_bounds.items():
            mask = workspace.active_group_masks.get(gid)
            if mask is None or not np.any(mask):
                continue
            active_sum = float(weights[mask].sum() - benchmark[mask].sum())
            lower_b, upper_b = bound
            if np.isfinite(upper_b) and active_sum > float(upper_b) + tol:
                return False
            if np.isfinite(lower_b) and active_sum < float(lower_b) - tol:
                return False
        return True

    def _set_group_sum(
        self,
        weights: np.ndarray,
        mask: np.ndarray,
        target: float,
        lower: np.ndarray,
        upper: np.ndarray,
        benchmark: np.ndarray,
    ) -> np.ndarray:
        adjusted = weights.copy()
        current = float(adjusted[mask].sum())
        if current <= 0.0:
            base = benchmark[mask]
            if base.sum() <= 0:
                base = np.ones(np.count_nonzero(mask), dtype=float)
            weights_slice = np.clip(base, 1e-12, None)
            weights_slice = target * weights_slice / max(weights_slice.sum(), 1e-12)
            adjusted[mask] = weights_slice
        else:
            if target <= 0.0:
                adjusted[mask] = 0.0
            else:
                ratio = target / max(current, 1e-12)
                adjusted[mask] = adjusted[mask] * ratio
        return np.clip(adjusted, lower, upper)

    def _redistribute_outside_group(
        self,
        weights: np.ndarray,
        mask: np.ndarray,
        target_total: float,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        adjusted = weights.copy()
        diff = adjusted.sum() - target_total
        if abs(diff) <= 1e-12:
            return adjusted
        comp = ~mask
        if not np.any(comp):
            return adjusted
        if diff > 0:
            slack = np.clip(adjusted[comp] - lower[comp], 0.0, None)
            total_slack = slack.sum()
            if total_slack > 0:
                step = np.minimum(slack, diff * slack / max(total_slack, 1e-12))
                adjusted[comp] = adjusted[comp] - step
        else:
            slack = np.clip(upper[comp] - adjusted[comp], 0.0, None)
            total_slack = slack.sum()
            if total_slack > 0:
                step = np.minimum(slack, (-diff) * slack / max(total_slack, 1e-12))
                adjusted[comp] = adjusted[comp] + step
        return np.clip(adjusted, lower, upper)

    def _enforce_factor_constraints(
        self,
        weights: np.ndarray,
        constraints: PortfolioConstraints,
        lower: np.ndarray,
        upper: np.ndarray,
        workspace: ConstraintWorkspace,
    ) -> np.ndarray:
        A = workspace.factor_T
        if A is None:
            return weights
        exposures = A @ weights
        if workspace.has_factor_equality and workspace.factor_targets is not None:
            desired = workspace.factor_targets.copy()
        else:
            desired = exposures.copy()
        lower_bounds = workspace.factor_lower
        upper_bounds = workspace.factor_upper
        if lower_bounds is not None:
            desired = np.maximum(desired, lower_bounds)
        if upper_bounds is not None:
            desired = np.minimum(desired, upper_bounds)
        if not workspace.has_factor_equality and (
            lower_bounds is not None or upper_bounds is not None
        ):
            lower_clip = lower_bounds if lower_bounds is not None else -np.inf
            upper_clip = upper_bounds if upper_bounds is not None else np.inf
            desired = np.clip(desired, lower_clip, upper_clip)
        residual = desired - exposures
        tol = workspace.factor_tolerance
        if np.linalg.norm(residual, ord=np.inf) <= tol:
            return weights
        try:
            if constraints.equality_enforce and abs(constraints.leverage_limit - 1.0) < 1e-12:
                u = np.ones((self.n_assets, 1), dtype=float)
                denom = float((u.T @ u).item())
                projector = (
                    np.eye(self.n_assets, dtype=float) - (u @ u.T) / denom
                    if denom > 0
                    else np.eye(self.n_assets, dtype=float)
                )
                At = A @ projector
                delta = projector @ np.linalg.lstsq(At, residual, rcond=None)[0]
            else:
                delta = np.linalg.lstsq(A, residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            if constraints.equality_enforce and abs(constraints.leverage_limit - 1.0) < 1e-12:
                return weights
            return weights
        adjusted = np.clip(weights + delta, lower, upper)
        if constraints.equality_enforce and abs(constraints.leverage_limit - 1.0) < 1e-12:
            adjusted = self._project_sum_with_bounds(adjusted, lower, upper, 1.0)
        elif adjusted.sum() > constraints.leverage_limit:
            adjusted = self._project_sum_with_bounds(
                adjusted, lower, upper, constraints.leverage_limit
            )
        return adjusted

    def _factor_constraints_satisfied(
        self,
        weights: np.ndarray,
        constraints: PortfolioConstraints,
        workspace: ConstraintWorkspace,
        tol: float = 1e-8,
    ) -> bool:
        if not (workspace.has_factor_equality or workspace.has_factor_bounds):
            return True
        A = workspace.factor_T
        if A is None:
            return True
        exposures = A @ weights
        tol_total = workspace.factor_tolerance + tol
        if workspace.has_factor_equality and workspace.factor_targets is not None:
            if np.linalg.norm(exposures - workspace.factor_targets, ord=np.inf) > tol_total:
                return False
        if workspace.has_factor_bounds:
            lower_bounds = workspace.factor_lower
            upper_bounds = workspace.factor_upper
            if lower_bounds is not None:
                if np.any(exposures < lower_bounds - tol_total):
                    return False
            if upper_bounds is not None:
                if np.any(exposures > upper_bounds + tol_total):
                    return False
        return True

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
        objective: ObjectiveSpec,
        constraints: PortfolioConstraints,
        benchmark: Optional[BenchmarkStats] = None,
        workspace: Optional[ConstraintWorkspace] = None,
    ) -> None:
        if not portfolios:
            return

        topk = min(self.cfg.topk_refine, len(portfolios))
        if topk <= 0:
            return

        workspace = workspace or self._build_constraint_workspace(constraints)
        indices = np.argsort(scores)[-topk:]
        lower, upper, bench = self._compute_weight_bounds(constraints, workspace)
        bounds = list(zip(lower.tolist(), upper.tolist()))
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
        if (
            workspace.has_factor_equality
            and workspace.factor_T is not None
            and workspace.factor_targets is not None
        ):
            Aeq_rows.append(workspace.factor_T.astype(float))
            beq_vals.append(workspace.factor_targets.astype(float))
        if workspace.has_factor_bounds and workspace.factor_T is not None:
            F = workspace.factor_T.T.astype(float)
            tol = workspace.factor_tolerance
            if workspace.factor_upper is not None:
                for idx, ub in enumerate(workspace.factor_upper):
                    if not np.isfinite(ub):
                        continue
                    row = F[:, idx]
                    Aineq_rows.append(row)
                    bineq_vals.append(float(ub) + tol)
            if workspace.factor_lower is not None:
                for idx, lb in enumerate(workspace.factor_lower):
                    if not np.isfinite(lb):
                        continue
                    row = -F[:, idx]
                    Aineq_rows.append(row)
                    bineq_vals.append(-(float(lb) - tol))
        if constraints.sector_map is not None:
            sectors = np.asarray(constraints.sector_map, dtype=int)
            cap = float(constraints.max_sector_concentration)
            for sector in np.unique(sectors):
                row = np.zeros(self.n_assets, dtype=float)
                row[sectors == sector] = 1.0
                Aineq_rows.append(row)
                bineq_vals.append(cap)
        if bench is not None and workspace.active_group_bounds:
            for gid, bound in workspace.active_group_bounds.items():
                mask = workspace.active_group_masks.get(gid)
                if mask is None or not np.any(mask):
                    continue
                bench_sum = float(bench[mask].sum())
                lower_b, upper_b = bound
                if np.isfinite(upper_b):
                    row = workspace.active_group_rows[gid].astype(float)
                    Aineq_rows.append(row)
                    bineq_vals.append(bench_sum + float(upper_b))
                if np.isfinite(lower_b):
                    row = -workspace.active_group_rows[gid].astype(float)
                    Aineq_rows.append(row)
                    bineq_vals.append(-(bench_sum + float(lower_b)))
        Aeq = np.vstack(Aeq_rows) if Aeq_rows else None
        beq = np.concatenate(beq_vals) if beq_vals else None
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
                    workspace=workspace,
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
                projector=lambda w, _c=constraints, _ws=workspace: self._apply_constraints(
                    np.asarray(w, dtype=float), _c, _ws
                ),
            )
            active_breaches = 0
            if bench is not None:
                active = refined - bench
                if np.isfinite(constraints.min_active_weight):
                    active_breaches += int(
                        np.count_nonzero(
                            active < float(constraints.min_active_weight) - 1e-8
                        )
                    )
                if np.isfinite(constraints.max_active_weight):
                    active_breaches += int(
                        np.count_nonzero(
                            active > float(constraints.max_active_weight) + 1e-8
                        )
                    )
                if workspace.active_group_bounds:
                    for gid, bound in workspace.active_group_bounds.items():
                        mask = workspace.active_group_masks.get(gid)
                        if mask is None or not np.any(mask):
                            continue
                        active_sum = float(active[mask].sum())
                        lower_b, upper_b = bound
                        if np.isfinite(upper_b) and active_sum > float(upper_b) + 1e-8:
                            active_breaches += 1
                        if np.isfinite(lower_b) and active_sum < float(lower_b) - 1e-8:
                            active_breaches += 1
            setattr(res, "active_breaches", active_breaches)
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
