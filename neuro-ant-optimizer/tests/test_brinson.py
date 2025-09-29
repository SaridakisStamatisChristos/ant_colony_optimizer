import numpy as np

from neuro_ant_optimizer.attribution.brinson import compute_brinson_attribution


def test_brinson_active_return_matches_totals():
    portfolio_weights = [
        [0.6, 0.2, 0.2],
        [0.5, 0.3, 0.2],
    ]
    benchmark_weights = [
        [0.5, 0.3, 0.2],
        [0.4, 0.4, 0.2],
    ]
    asset_returns = [
        [0.10, 0.05, 0.02],
        [0.03, 0.02, 0.01],
    ]
    sectors = ["Tech", "Tech", "Health"]
    dates = ["2020-01-31", "2020-02-29"]

    result = compute_brinson_attribution(portfolio_weights, benchmark_weights, asset_returns, sectors, dates)
    for date in dates:
        contributions = [row["total"] for row in result.total_rows() if row["date"] == date]
        assert contributions
        total_effect = float(np.sum(contributions))
        assert np.isclose(total_effect, result.active_returns[date], atol=1e-12)

    sectors_present = {row["sector"] for row in result.total_rows()}
    assert sectors_present == {"Health", "Tech"}
