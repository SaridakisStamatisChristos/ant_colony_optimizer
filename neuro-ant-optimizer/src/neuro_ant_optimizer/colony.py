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

    def build(
        self,
        pheromone_net,
        risk_net,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """
        Build a tour using pheromone transitions and (optional) risk heuristics.
        Returns equal-weight portfolio over visited set (size = n_assets).
        """

        n = self.n_assets
        trans = pheromone_net.transition_matrix().detach().cpu().numpy()
        risk = np.ones(n, dtype=float)
        if risk_net is not None:
            I = torch.eye(n, device=risk_net.param_device, dtype=risk_net.param_dtype)
            r = risk_net(I).detach().cpu().numpy()
            risk = np.clip(np.diag(r), 1e-6, None)
        self.visited = [int(np.random.randint(0, n))]
        while len(self.visited) < n:
            cur = self.visited[-1]
            mask = np.ones(n, dtype=bool)
            mask[self.visited] = False
            logits = np.log(np.clip(trans[cur], 1e-12, None)) * float(alpha) + np.log(
                risk
            ) * float(beta)
            p = safe_softmax(logits, axis=-1, mask=mask)
            nxt = int(np.random.choice(n, p=p))
            self.visited.append(nxt)
        w = np.zeros(n, dtype=float)
        w[self.visited] = 1.0 / n
        self.w = w
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

