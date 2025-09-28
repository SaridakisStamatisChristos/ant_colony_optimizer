from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch

from .utils import safe_softmax


class Ant:
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.visited: List[int] = []
        self.w: np.ndarray | None = None
        self._mask = np.ones(n_assets, dtype=bool)
        self._weights = np.zeros(n_assets, dtype=float)

    def build(
        self,
        pheromone_net,
        risk_net,
        alpha: float,
        beta: float,
        trans_matrix: np.ndarray | None = None,
        trans_log: np.ndarray | None = None,
        risk_bias: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        initial: int | None = None,
    ) -> np.ndarray:
        """
        Build a tour using pheromone transitions and (optional) risk heuristics.
        Returns equal-weight portfolio over visited set (size = n_assets).
        """

        n = self.n_assets
        if rng is None:
            rng = np.random.default_rng()
        if trans_matrix is None:
            # Inference path: avoid building a grad graph
            with torch.no_grad():
                trans = (
                    pheromone_net.transition_matrix()
                    .detach()
                    .cpu()
                    .numpy()
                )
        else:
            trans = trans_matrix

        if trans_log is None:
            trans_log = np.log(np.clip(trans, 1e-12, None)) * float(alpha)
        else:
            trans_log = np.asarray(trans_log, dtype=float)

        if risk_bias is None:
            if risk_net is not None and beta:
                I = torch.eye(
                    n, device=risk_net.param_device, dtype=risk_net.param_dtype
                )
                r = risk_net(I).detach().cpu().numpy()
                risk = np.clip(np.diag(r), 1e-6, None)
                risk_bias = np.log(risk) * float(beta)
            else:
                risk_bias = np.zeros(n, dtype=float)
        start = int(initial) if initial is not None else int(rng.integers(0, n))
        self.visited = [start]
        mask = self._mask
        mask[:] = True
        mask[start] = False
        bias = np.asarray(risk_bias, dtype=float)
        while len(self.visited) < n:
            cur = self.visited[-1]
            logits = trans_log[cur] + bias
            p = safe_softmax(logits, axis=-1, mask=mask)
            nxt = int(rng.choice(n, p=p))
            self.visited.append(nxt)
            mask[nxt] = False
        w = self._weights
        w.fill(0.0)
        w[self.visited] = 1.0 / n
        self.w = w
        mask[self.visited] = True
        return w


class AntColony:
    def __init__(self, n_assets: int, evap: float = 0.1, Q: float = 1.0):
        self.n_assets = n_assets
        self.evap = float(evap)
        self.Q = float(Q)
        self.pheromone = np.ones((n_assets, n_assets), dtype=float) / n_assets

    def update_pheromone(self, ants: Iterable[Ant], scores: Iterable[float]) -> None:
        """Evaporate then deposit proportional to normalized scores."""

        self.pheromone *= 1.0 - self.evap
        s = np.asarray(list(scores), dtype=float)
        if s.size == 0:
            return
        if np.isfinite(s).all() and np.ptp(s) > 1e-12:
            s = (s - s.min()) / (np.ptp(s) + 1e-12)
        else:
            s = np.ones_like(s)
        for ant, sc in zip(ants, s):
            if sc <= 0 or len(ant.visited) < 2:
                continue
            dep = float(sc) * self.Q / (len(ant.visited) - 1)
            for a, b in zip(ant.visited[:-1], ant.visited[1:]):
                self.pheromone[a, b] += dep

