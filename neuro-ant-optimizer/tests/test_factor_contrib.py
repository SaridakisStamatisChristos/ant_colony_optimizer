import numpy as np

from neuro_ant_optimizer.attribution.factors import compute_factor_contributions


def test_factor_contributions_sum_to_active_return():
    factor_names = ["Value", "Momentum"]
    factor_attr_records = [
        {"date": "2020-01-31", "Value_return": 0.01, "Momentum_return": -0.02},
        {"date": "2020-02-29", "Value_return": 0.005, "Momentum_return": 0.015},
    ]
    factor_records = [
        {
            "date": "2020-01-31",
            "exposures": np.array([0.6, -0.2]),
            "targets": np.array([0.0, 0.0]),
            "missing": False,
        },
        {
            "date": "2020-02-29",
            "exposures": np.array([0.4, 0.1]),
            "targets": np.array([0.1, 0.0]),
            "missing": False,
        },
    ]

    attribution = compute_factor_contributions(factor_attr_records, factor_records, factor_names)
    contrib_map = {}
    for row in attribution.contribution_rows():
        contrib_map.setdefault(row["date"], 0.0)
        contrib_map[row["date"]] += float(row["contribution"])

    assert np.isclose(contrib_map["2020-01-31"], 0.01, atol=1e-12)
    assert np.isclose(contrib_map["2020-02-29"], 0.003, atol=1e-12)

    cumulative_rows = attribution.cumulative_rows()
    assert cumulative_rows
    last_row = {k: v for k, v in cumulative_rows[-1].items() if k != "date"}
    assert np.isclose(last_row["Value"], 0.0075, atol=1e-12)
    assert np.isclose(last_row["Momentum"], 0.0055, atol=1e-12)
