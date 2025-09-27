import torch

from neuro_ant_optimizer.models import PheromoneNetwork


def test_pheromone_grads_cpu():
    net = PheromoneNetwork(5).to("cpu")
    T = net.transition_matrix()  # should require grad by default
    loss = (T ** 2).sum()
    loss.backward()
    assert any(p.grad is not None for p in net.parameters())


def test_device_switch_cuda_if_available():
    if not torch.cuda.is_available():
        return

    net = PheromoneNetwork(4).to("cuda")
    T = net.transition_matrix()
    assert T.is_cuda
    loss = T.sum()
    loss.backward()
    assert any((p.grad is not None) and p.is_cuda for p in net.parameters())
