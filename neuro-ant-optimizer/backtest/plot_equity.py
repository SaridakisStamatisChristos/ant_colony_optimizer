"""Utility script to visualize backtest equity curves with optional overlays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - allow minimal envs
    pd = None  # type: ignore


def _load_equity(csv_path: Path) -> Tuple[List[object], np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing equity file: {csv_path}")
    if pd is not None:
        frame = pd.read_csv(csv_path)
        if "date" not in frame or "equity" not in frame:
            raise ValueError(f"CSV {csv_path} must contain 'date' and 'equity' columns")
        dates = frame["date"].tolist()
        equity = frame["equity"].to_numpy(dtype=float)
        return dates, equity

    raw = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if raw.size == 0:
        return [], np.asarray([], dtype=float)
    if "date" not in raw.dtype.names or "equity" not in raw.dtype.names:
        raise ValueError(f"CSV {csv_path} must contain 'date' and 'equity' columns")
    dates = raw["date"].tolist() if raw.ndim == 0 else raw["date"].tolist()
    equity = np.asarray(raw["equity"], dtype=float)
    return dates, equity


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot equity curves from backtest output")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing equity CSVs")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Destination PNG path (defaults to <dir>/equity_overlay.png)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay gross, net-of-tx, and net-of-slippage curves when available",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    base_path = out_dir / "equity.csv"
    dates, gross_curve = _load_equity(base_path)
    overlays: List[Tuple[str, np.ndarray]] = [("gross", gross_curve)]

    if args.overlay:
        tc_path = out_dir / "equity_net_of_tc.csv"
        if tc_path.exists():
            _, tc_curve = _load_equity(tc_path)
            overlays.append(("net tx", tc_curve))
        slip_path = out_dir / "equity_net_of_slippage.csv"
        if slip_path.exists():
            _, slip_curve = _load_equity(slip_path)
            overlays.append(("net slippage", slip_curve))

    try:  # pragma: no cover - plotting dependency
        import matplotlib.pyplot as plt

        plt.figure()
        for label, curve in overlays:
            if curve.size:
                plt.plot(dates, curve, label=label)
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.title("Equity Curves")
        if len(overlays) > 1:
            plt.legend()
        plt.tight_layout()
        out_path = Path(args.out) if args.out else out_dir / "equity_overlay.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved {out_path}")
    except ModuleNotFoundError as exc:  # pragma: no cover - minimal envs
        raise RuntimeError("matplotlib is required for plotting") from exc


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
