import numpy as np

from neuro_ant_optimizer.utils import nearest_psd, safe_softmax, set_seed


def test_nearest_psd_eigs_nonnegative():
    A = np.array([[1.0, 2.0], [3.0, 1.0]])
    P = nearest_psd(A)
    w = np.linalg.eigvalsh(0.5 * (P + P.T))
    assert (w >= -1e-12).all()


def test_safe_softmax_mask_axis():
    x = np.array([[1.0, 2.0, 3.0]])
    mask = np.array([[True, False, True]])
    p = safe_softmax(x, axis=-1, mask=mask)
    assert np.isfinite(p).all()
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0, 1] == 0.0


def test_seed_determinism_numpy_only():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)

