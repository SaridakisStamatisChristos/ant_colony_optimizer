import numpy as np

from neuro_ant_optimizer.utils import nearest_psd, shrink_covariance


def test_shrink_covariance_basic():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    cov = 0.5 * (A + A.T) + np.eye(6) * 0.1

    shrunk = shrink_covariance(cov, delta=0.2)

    assert np.allclose(shrunk, shrunk.T, atol=1e-12)
    assert np.allclose(np.diag(shrunk), np.diag(cov))
    assert np.all(np.abs(shrunk - np.diag(np.diag(cov))) <= np.abs(cov - np.diag(np.diag(cov))) + 1e-12)

    psd = nearest_psd(shrunk)
    eigs = np.linalg.eigvalsh(0.5 * (psd + psd.T))
    assert (eigs >= -1e-12).all()
