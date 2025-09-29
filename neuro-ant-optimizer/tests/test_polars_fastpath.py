from pathlib import Path

import numpy as np
import pytest

from neuro_ant_optimizer.data import load_returns

pl = pytest.importorskip("polars")


def test_polars_backend_reads_parquet(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "A": [0.01, 0.02, -0.01],
            "B": [0.005, 0.006, 0.007],
        }
    )
    parquet_path = tmp_path / "returns.parquet"
    df.write_parquet(parquet_path)

    frame = load_returns(parquet_path, freq="B", tz="UTC", backend="polars")
    data = frame.to_numpy(dtype=float)
    assert data.shape == (3, 2)
    np.testing.assert_allclose(data[0], [0.01, 0.005])
