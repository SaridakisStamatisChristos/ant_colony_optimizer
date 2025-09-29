import numpy as np

from neuro_ant_optimizer.constraints.validators import (
    build_limit_evaluator,
    post_trade_check,
    pre_trade_check,
    summarize_breaches,
)


def test_pre_and_post_trade_limit_diagnostics():
    assets = ["A", "B", "C"]
    sector_lookup = {"A": "Tech", "B": "Tech", "C": "Utilities"}
    limit_spec = {
        "per_asset": {"A": {"max": 0.5}, "C": {"min": 0.05}},
        "sectors": {"Tech": {"max": 0.7}},
        "groups": {"North": {"members": ["A", "C"], "max": 0.75}},
        "exposures": {"Beta": {"loadings": {"A": 1.0, "B": 0.5}, "max": 0.7}},
        "leverage": {"gross": 1.0, "net": 0.6},
    }

    evaluator = build_limit_evaluator(limit_spec, assets=assets, sector_lookup=sector_lookup)
    assert evaluator is not None

    pre_weights = np.array([0.55, 0.35, 0.10])
    pre_ok, pre_reasons, pre_breaches = pre_trade_check(pre_weights, evaluator)
    assert not pre_ok
    assert any(reason.startswith("PRE:ASSET:A:MAX") for reason in pre_reasons)
    assert any(reason.startswith("PRE:EXPOSURE:Beta:MAX") for reason in pre_reasons)
    assert len(pre_breaches) == len(pre_reasons)

    post_weights = np.array([0.52, 0.33, 0.15])
    post_ok, post_reasons, post_breaches = post_trade_check(post_weights, evaluator)
    assert not post_ok
    assert any(reason.startswith("POST:SECTOR:Tech:MAX") for reason in post_reasons)
    assert any(reason.startswith("POST:LEVERAGE:net:MAX") for reason in post_reasons)

    summary = summarize_breaches([*pre_breaches, *post_breaches])
    summary_map = {(entry["phase"], entry["type"]): entry["count"] for entry in summary}
    assert summary_map[("PRE", "ASSET")] >= 1
    assert summary_map[("POST", "SECTOR")] >= 1
    assert summary_map[("POST", "LEVERAGE")] >= 1
