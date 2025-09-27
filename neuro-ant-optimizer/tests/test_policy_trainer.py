import numpy as np

from neuro_ant_optimizer.models import PheromoneNetwork
from neuro_ant_optimizer.optimizer import PolicyTrainer


def test_policy_trainer_step_runs():
    N = 6
    net = PheromoneNetwork(N).to_device()
    trainer = PolicyTrainer(net)
    targets = np.stack([np.ones((N, N)) / N for _ in range(4)], axis=0)
    loss = trainer.step(targets)
    assert np.isfinite(loss)

