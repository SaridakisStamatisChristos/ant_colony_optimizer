from __future__ import annotations
import numpy as np
from typing import List
from .models import PheromoneNetwork, RiskAssessmentNetwork
from .utils import safe_softmax

class Ant:
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.visited: List[int] = []
        self.w: np.ndarray = np.zeros(n_assets, dtype=float)

    def build(self, pheromone_net: PheromoneNetwork, risk_net: RiskAssessmentNetwork|None,
              mu: np.ndarray, sigma: np.ndarray, corr: np.ndarray,
              min_alloc: float=0.01, base_alloc: float=0.10, risk_weight: float=0.5) -> np.ndarray:
        current = np.random.randint(self.n_assets)
        self.visited = [current]
        self.w[:] = 0.0

        risks = (risk_net.predict(mu, sigma, corr) if risk_net is not None
                 else np.clip(sigma / (sigma.max() + 1e-12), 0, 1))

        self.w[current] = max(min_alloc, base_alloc * (1 - risks[current]))
        while len(self.visited) < self.n_assets and self.w.sum() < 0.95:
            probs = pheromone_net.transition_probs(current, self.visited)
            heuristic = safe_softmax(mu - risk_weight * risks)
            blend = 0.6 * probs + 0.4 * heuristic
            blend = blend / blend.sum()
            unvisited = [i for i in range(self.n_assets) if i not in self.visited]
            if not unvisited:
                break
            p = blend[unvisited]
            if p.sum() <= 0:
                nxt = np.random.choice(unvisited)
            else:
                p = p / p.sum()
                nxt = np.random.choice(unvisited, p=p)
            alloc = max(min_alloc, base_alloc * (1 - risks[nxt]))
            if self.w.sum() + alloc <= 1.0:
                self.w[nxt] = alloc
                self.visited.append(nxt)
                current = nxt
            else:
                break

        s = self.w.sum()
        if s > 0:
            self.w /= s
        return self.w

class AntColony:
    def __init__(self, n_assets: int, n_ants: int=150, evaporation: float=0.5, Q: float=100.0):
        self.n_assets = n_assets
        self.n_ants = n_ants
        self.evap = float(np.clip(evaporation, 0.0, 1.0))
        self.Q = Q
        self.pheromone = np.ones((n_assets, n_assets), dtype=float) * 0.1

    def init_colony(self) -> List[Ant]:
        return [Ant(self.n_assets) for _ in range(self.n_ants)]

    def update(self, ants: List[Ant], scores: list[float]) -> None:
        self.pheromone *= (1.0 - self.evap)
        if not ants:
            return
        s = np.array(scores, dtype=float)
        if np.all(np.isfinite(s)) and s.max() > s.min():
            s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        else:
            s = np.ones_like(s)
        for ant, sc in zip(ants, s):
            if sc <= 0 or len(ant.visited) < 2:
                continue
            deposit = float(sc) * self.Q
            for a, b in zip(ant.visited[:-1], ant.visited[1:]):
                self.pheromone[a, b] += deposit
