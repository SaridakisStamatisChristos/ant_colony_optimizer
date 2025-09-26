import numpy as np
from neuro_ant_optimizer.utils import nearest_psd

def test_nearest_psd():
    A = np.array([[1,2],[3,4.0]])
    cov = A @ A.T
    cov_psd = nearest_psd(cov)
    eig = np.linalg.eigvalsh(cov_psd)
    assert np.min(eig) >= -1e-8
