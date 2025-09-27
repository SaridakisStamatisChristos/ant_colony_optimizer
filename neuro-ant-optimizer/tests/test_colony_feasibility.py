import numpy as np

from neuro_ant_optimizer.colony import Ant
from neuro_ant_optimizer.models import PheromoneNetwork, RiskAssessmentNetwork


def test_ant_build_feasible_equalweight():
    N = 8
    pn = PheromoneNetwork(N).to_device().eval()
    rn = RiskAssessmentNetwork(N).to_device().eval()
    ant = Ant(N)
    w = ant.build(pn, rn, alpha=1.0, beta=0.2)
    assert w.shape == (N,)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= -1e-12).all()

