from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from enum import Enum
import logging

from .models import RiskAssessmentNetwork, PheromoneNetwork
from .colony import Ant, AntColony
from .constraints import PortfolioConstraints
from .utils import nearest_psd, set_seed
from .refine import refine_slsqp
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

class OptimizationObjective(Enum):
    SHARPE_RATIO = "sharpe_ratio"
    MAX_RETURN = "max_return"
    MIN_VARIANCE = "min_variance"
    RISK_PARITY = "risk_parity"

class NeuroAntPortfolioOptimizer:
    def __init__(self, n_assets: int, config: Optional[Dict] = None):
        self.n_assets = n_assets
        self.cfg = config or self._default_cfg()

        self.risk_net = RiskAssessmentNetwork(n_assets) if self.cfg["use_risk_head"] else None
        self.phero_net = PheromoneNetwork(n_assets)
        self.colony = AntColony(n_assets, self.cfg["n_ants"], self.cfg["evaporation"], self.cfg["Q"])

        if self.risk_net is not None:
            self.risk_optim = torch.optim.Adam(self.risk_net.parameters(), lr=self.cfg["lr"])
        self.phero_optim = torch.optim.Adam(self.phero_net.parameters(), lr=self.cfg["lr"])
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

        self.history: List[Dict] = []
        self.best_w: Optional[np.ndarray] = None
        self.best_score: float = -np.inf

    def _default_cfg(self) -> Dict:
        return dict(
            n_ants=150, max_iter=200, patience=20, seed=42, lr=1e-3, risk_free=0.02,
            evaporation=0.5, Q=100.0, topk_refine=12, topk_train=16,
            use_risk_head=True, refine_maxiter=200
        )

    def optimize(self, returns: np.ndarray, covariance: np.ndarray, constraints: PortfolioConstraints,
                 objective: OptimizationObjective=OptimizationObjective.SHARPE_RATIO):
        set_seed(self.cfg["seed"])
        mu = np.asarray(returns, dtype=float).ravel()
        cov = nearest_psd(np.asarray(covariance, dtype=float))
        self._validate(mu, cov)
        sigma = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
        corr = cov / np.outer(sigma, sigma)

        best_w = np.ones(self.n_assets) / self.n_assets
        best_score = -np.inf
        no_improve = 0
        converged = False
        message = "OK"

        for it in range(self.cfg["max_iter"]):
            ants = self.colony.init_colony()
            portfolios: List[np.ndarray] = []
            scores: List[float] = []

            for ant in ants:
                w = ant.build(self.phero_net, self.risk_net, mu, sigma, corr)
                w = self._apply_constraints(w, constraints)
                s = self._score(w, mu, cov, objective, constraints)
                portfolios.append(w); scores.append(s)

            topk = min(self.cfg["topk_refine"], len(portfolios))
            idx = np.argsort(scores)[-topk:]
            for j in idx:
                w0 = portfolios[j]
                score_fn = lambda w: self._score(w, mu, cov, objective, constraints)
                w_ref, s_ref, ok = refine_slsqp(w0, score_fn, self.n_assets, constraints, self.cfg["refine_maxiter"])
                if ok:
                    portfolios[j] = w_ref; scores[j] = s_ref

            self.colony.update(ants, scores)

            if self.risk_net is not None:
                self._train_risk(mu, sigma, corr)
            self._train_pheromone(portfolios, scores)

            jbest = int(np.argmax(scores))
            if scores[jbest] > best_score + 1e-12:
                best_score = float(scores[jbest]); best_w = portfolios[jbest].copy(); no_improve = 0
            else:
                no_improve += 1

            self.history.append(dict(iter=it, best=best_score, avg=float(np.mean(scores))))
            if it % 10 == 0:
                logger.info(f"Iter {it:03d} | best={best_score:.6f} avg={np.mean(scores):.6f}")
            if no_improve >= self.cfg["patience"]:
                converged = True; message = f"Early stop at iter {it}"
                break

        w_final = self._apply_constraints(best_w, constraints)
        return type("OptimizationResult", (), dict(
            weights=w_final,
            expected_return=float(w_final @ mu),
            volatility=float(np.sqrt(max(w_final @ cov @ w_final, 0.0))),
            sharpe_ratio=self._sharpe(w_final, mu, cov),
            optimization_time=0.0,
            convergence_status=converged,
            iteration_count=len(self.history),
            risk_contributions=self._risk_contrib(w_final, cov),
            message=message,
        ))

    # ---- internals ----
    def _score(self, w, mu, cov, obj, c):
        if not self._feasible(w, c): return -1e9
        r = float(w @ mu); v = float(np.sqrt(max(w @ cov @ w, 0.0)))
        if obj.name == "SHARPE_RATIO":
            return (r - self.cfg["risk_free"]) / v if v > 1e-12 else -1e6
        if obj.name == "MAX_RETURN": return r
        if obj.name == "MIN_VARIANCE": return -v
        if obj.name == "RISK_PARITY":
            rc = self._risk_contrib(w, cov); s = rc.sum()
            if s <= 0: return -1e6
            rc /= s; eq = np.ones_like(rc)/len(rc)
            return -float(np.linalg.norm(rc - eq))
        return -1e9

    def _apply_constraints(self, w, c):
        w = np.clip(w, c.min_weight, c.max_weight)
        if c.equality_enforce and abs(c.leverage_limit - 1.0) < 1e-12:
            s = w.sum(); w = (w / s) if s>0 else np.ones_like(w)/len(w)
            w = np.clip(w, c.min_weight, c.max_weight); w = w / w.sum()
        else:
            if w.sum() > c.leverage_limit:
                w *= c.leverage_limit / (w.sum() + 1e-12)
        if c.sector_map is not None:
            w = self._enforce_sector_caps(w, c)
        if c.prev_weights is not None:
            w = self._enforce_turnover(w, c)
        return w

    def _feasible(self, w, c, tol=1e-8):
        if np.any(w < c.min_weight - tol) or np.any(w > c.max_weight + tol): return False
        if w.sum() > c.leverage_limit + tol: return False
        if c.equality_enforce and abs(c.leverage_limit - 1.0) < 1e-12 and abs(w.sum()-1.0) > 1e-6: return False
        if c.sector_map is not None:
            sects = np.array(c.sector_map, dtype=int)
            for s in np.unique(sects):
                if w[sects==s].sum() > c.max_sector_concentration + tol: return False
        if c.prev_weights is not None:
            if np.abs(w - c.prev_weights).sum() > c.max_turnover + tol: return False
        return True

    def _enforce_sector_caps(self, w, c):
        sects = np.array(c.sector_map, dtype=int); w = w.copy()
        for s in np.unique(sects):
            idx = (sects==s); tot = w[idx].sum(); cap = c.max_sector_concentration
            if tot > cap: w[idx] *= cap / (tot + 1e-12)
        if c.equality_enforce and abs(c.leverage_limit - 1.0) < 1e-12:
            s = w.sum(); w = (w / s) if s>0 else w
        else:
            w = np.minimum(w, c.max_weight)
        return w

    def _enforce_turnover(self, w, c):
        prev = np.asarray(c.prev_weights, dtype=float); diff = w - prev
        l1 = np.abs(diff).sum()
        if l1 <= c.max_turnover + 1e-12: return w
        if l1 > 0:
            alpha = c.max_turnover / l1; w = prev + alpha * diff
        w = np.clip(w, c.min_weight, c.max_weight)
        if c.equality_enforce and abs(c.leverage_limit - 1.0) < 1e-12:
            s = w.sum(); w = (w / s) if s>0 else w
        else:
            if w.sum() > c.leverage_limit:
                w *= c.leverage_limit / (w.sum() + 1e-12)
        return w

    def _risk_contrib(self, w, cov):
        var = float(w @ cov @ w)
        if var <= 1e-18: return np.zeros_like(w)
        mc = cov @ w; return w * mc / var

    def _sharpe(self, w, mu, cov):
        v = float(np.sqrt(max(w @ cov @ w, 0.0)))
        return 0.0 if v <= 1e-12 else (float(w @ mu) - self.cfg["risk_free"]) / v

    def _validate(self, mu, cov):
        n = mu.shape[0]
        if cov.shape != (n, n): raise ValueError("covariance must be (n,n)")
        if not np.allclose(cov, cov.T, atol=1e-10): raise ValueError("covariance must be symmetric")

    # ---- learning routines ----
    def _train_risk(self, mu: np.ndarray, sigma: np.ndarray, corr: np.ndarray) -> None:
        if self.risk_net is None:
            return

        # Features mirror the predict() pathway to keep inference consistent.
        avg_corr = (corr.sum(axis=1) - 1.0) / max(self.n_assets - 1, 1)
        feats = np.stack([mu, sigma, avg_corr], axis=1).reshape(-1).astype(np.float32)
        target = (sigma / (sigma.max() + 1e-12)).astype(np.float32)

        x = torch.from_numpy(feats).unsqueeze(0)
        y = torch.from_numpy(target).unsqueeze(0)

        self.risk_net.train()
        self.risk_optim.zero_grad()

        pred = self.risk_net(x)
        loss = self.mse(pred, y)
        loss.backward()
        clip_grad_norm_(self.risk_net.parameters(), 1.0)
        self.risk_optim.step()
        self.risk_net.eval()

    def _train_pheromone(self, portfolios: List[np.ndarray], scores: List[float]) -> None:
        if len(portfolios) == 0:
            return

        k = min(self.cfg["topk_train"], len(portfolios))
        idx = np.argsort(scores)[-k:]
        targets = [np.tile(np.clip(portfolios[i], 0.0, 1.0), (self.n_assets, 1)) for i in idx]
        target_mat = np.mean(targets, axis=0)
        target_mat = np.clip(target_mat, 1e-6, 1 - 1e-6).astype(np.float32)

        asset_idx = torch.arange(self.n_assets, dtype=torch.long)
        target_tensor = torch.from_numpy(target_mat)

        self.phero_net.train()
        self.phero_optim.zero_grad()

        pred = torch.clamp(self.phero_net(asset_idx), 1e-6, 1 - 1e-6)
        loss = self.bce(pred, target_tensor)
        loss.backward()
        clip_grad_norm_(self.phero_net.parameters(), 1.0)
        self.phero_optim.step()
        self.phero_net.eval()
