"""Streamlit dashboard for inspecting neuro-ant backtest artifacts.

Upload the CSV outputs from a ``bt_out`` directory (either individually or
as a zipped archive) and the app will visualise headline metrics, drawdown
statistics, equity curve overlays, and scenario shocks.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd
import streamlit as st


@dataclass
class UploadedFile:
    """Container for a single artifact loaded from the upload widget."""

    name: str
    data: bytes


def _gather_files(files: Iterable["st.runtime.uploaded_file_manager.UploadedFile"]) -> Dict[str, UploadedFile]:
    """Normalise uploaded files (supporting both raw CSVs and zip archives).

    The mapping keys are the lowercase base filenames so downstream lookups can
    be case-insensitive. The original name is preserved for display.
    """

    storage: Dict[str, UploadedFile] = {}
    for uploaded in files:
        raw_name = Path(uploaded.name).name
        payload = uploaded.read()
        if raw_name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    member_name = Path(member.filename).name
                    key = member_name.lower()
                    if key in storage:
                        continue
                    storage[key] = UploadedFile(name=member_name, data=archive.read(member))
        else:
            key = raw_name.lower()
            if key in storage:
                continue
            storage[key] = UploadedFile(name=raw_name, data=payload)
    return storage


def _read_csv(uploaded: Optional[UploadedFile]) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(uploaded.data))
    except Exception as exc:  # pragma: no cover - streamlit surface
        st.warning(f"Failed to parse {uploaded.name}: {exc}")
        return None


def _render_metrics(artifacts: Mapping[str, UploadedFile]) -> None:
    metrics_df = _read_csv(artifacts.get("metrics.csv"))
    if metrics_df is None or metrics_df.empty:
        st.info("Upload metrics.csv to see the aggregated run metrics.")
        return

    if "metric" in metrics_df.columns and "value" in metrics_df.columns:
        metrics_df = metrics_df.set_index("metric")
        metrics_df["value"] = pd.to_numeric(metrics_df["value"], errors="coerce")
    st.subheader("Run metrics")
    st.dataframe(metrics_df, use_container_width=True)


def _normalise_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    copy = df.copy()
    copy["date"] = pd.to_datetime(copy["date"], errors="coerce")
    copy = copy.dropna(subset=["date"])
    return copy


def _render_equity_overlay(artifacts: Mapping[str, UploadedFile]) -> None:
    series_specs = [
        ("equity.csv", "Gross equity"),
        ("equity_net_of_tc.csv", "Net of transaction costs"),
        ("equity_net_of_slippage.csv", "Net of slippage"),
    ]
    merged: Optional[pd.DataFrame] = None
    for filename, label in series_specs:
        df = _read_csv(artifacts.get(filename))
        if df is None or "equity" not in df.columns:
            continue
        df = _normalise_date_column(df)
        if df.empty:
            continue
        subset = df[[col for col in df.columns if col in {"date", "equity"}]].copy()
        subset = subset.rename(columns={"equity": label})
        if merged is None:
            merged = subset
        else:
            merged = pd.merge(merged, subset, on="date", how="outer")
    if merged is None:
        st.info("Upload equity.csv (and optional net-of-cost files) to plot equity overlays.")
        return

    merged = merged.sort_values("date").set_index("date")
    st.subheader("Equity curve overlay")
    st.line_chart(merged)


def _render_drawdowns(artifacts: Mapping[str, UploadedFile]) -> None:
    drawdowns_df = _read_csv(artifacts.get("drawdowns.csv"))
    if drawdowns_df is None or drawdowns_df.empty:
        st.info("Upload drawdowns.csv to review peak/trough stats.")
        return

    for col in ["peak", "trough", "recovery"]:
        if col in drawdowns_df.columns:
            drawdowns_df[col] = pd.to_datetime(drawdowns_df[col], errors="coerce")
    st.subheader("Drawdown events")
    st.dataframe(drawdowns_df, use_container_width=True)
    if "trough" in drawdowns_df.columns and "depth" in drawdowns_df.columns:
        chart_df = drawdowns_df.dropna(subset=["trough", "depth"])
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("trough")["depth"], height=240)


def _extract_latest_weights(df: pd.DataFrame) -> pd.Series:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if "date" in numeric.columns:
        numeric = numeric.drop(columns="date")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        return pd.Series(dtype=float)
    return numeric.iloc[-1].dropna()


def _render_scenarios(artifacts: Mapping[str, UploadedFile]) -> None:
    shocked_df = _read_csv(artifacts.get("weights_after_shock.csv"))
    if shocked_df is None or shocked_df.empty or "asset" not in shocked_df.columns:
        st.info("Upload weights_after_shock.csv to visualise scenario deltas.")
        return

    shocked_df = shocked_df.set_index("asset")
    shocked_df = shocked_df.apply(pd.to_numeric, errors="coerce")

    weights_df = _read_csv(artifacts.get("weights.csv"))
    if weights_df is None or weights_df.empty:
        base_weights = pd.Series(0.0, index=shocked_df.index)
    else:
        base_weights = _extract_latest_weights(weights_df).reindex(shocked_df.index).fillna(0.0)

    deltas = shocked_df.subtract(base_weights, axis=0)

    st.subheader("Scenario deltas")
    scenario_names = list(shocked_df.columns)
    selected = st.selectbox("Scenario", scenario_names)
    scenario_view = pd.DataFrame(
        {
            "Base": base_weights,
            "Scenario": shocked_df[selected],
            "Delta": deltas[selected],
        }
    )
    st.dataframe(scenario_view, use_container_width=True)
    st.bar_chart(scenario_view["Delta"], height=260)

    report_df = _read_csv(artifacts.get("scenarios_report.csv"))
    if report_df is not None and not report_df.empty:
        if "breaches" in report_df.columns:
            report_df["breaches"] = report_df["breaches"].fillna(0).astype(int)
        st.markdown("#### Scenario summary")
        st.dataframe(report_df, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Neuro Ant Backtest Dashboard", layout="wide")
    st.title("Neuro Ant Backtest Dashboard")
    st.write(
        "Upload the CSV artifacts from a `bt_out` directory. You can select multiple files "
        "or provide a zipped archive – the app will unpack it automatically."
    )

    uploads = st.file_uploader(
        "bt_out artifacts",
        type=["csv", "zip"],
        accept_multiple_files=True,
        help="Drop the contents of bt_out/ or a zipped archive produced by --archive-runs.",
    )

    if not uploads:
        st.stop()

    artifacts = _gather_files(uploads)
    if not artifacts:
        st.warning("No parsable artifacts detected. Ensure you selected CSV outputs from bt_out/.")
        st.stop()

    st.success(f"Loaded {len(artifacts)} artifact files: {', '.join(sorted(v.name for v in artifacts.values()))}")

    col_left, col_right = st.columns(2)
    with col_left:
        _render_metrics(artifacts)
    with col_right:
        _render_equity_overlay(artifacts)

    st.markdown("---")
    col_bottom_left, col_bottom_right = st.columns(2)
    with col_bottom_left:
        _render_drawdowns(artifacts)
    with col_bottom_right:
        _render_scenarios(artifacts)


if __name__ == "__main__":  # pragma: no cover - streamlit entry point
    main()
