"""Anomalies page: robust Mahalanobis baseline per transaction."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from tensor_chain.models.baseline import mahalanobis_anomalies
from tensor_chain.ui.common import ETHERSCAN, dataset_version, load_transactions, require_dataset

ALPHAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
TABLE_COLUMNS = [
    "timestamp",
    "kind",
    "hash",
    "from",
    "to",
    "value_eth",
    "fee_eth",
    "gas_limit",
    "gas_used",
    "gas_price_gwei",
    "method_id",
    "score",
    "p_value",
]


@st.cache_data(show_spinner="Computing distances…", max_entries=8)
def _score(name: str, version: float, alpha: float) -> pd.DataFrame:
    return mahalanobis_anomalies(load_transactions(name, version), alpha)


def render() -> None:
    st.title("Anomalies (baseline)")
    st.caption(
        "Robust Mahalanobis (MCD) over log(value in wei), log(gas) and log(gas price), with one "
        "model per transaction segment. A transaction is anomalous when its χ² p-value is "
        "below α."
    )
    loaded = require_dataset()
    if loaded is None:
        return
    name, txs = loaded

    alpha = st.select_slider(
        "α (significance level)", ALPHAS, value=1e-3, format_func=lambda a: f"{a:.0e}"
    )
    scored = _score(name, dataset_version(name), alpha)
    anomalies = scored[scored["is_anomaly"]].sort_values("score", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalies", f"{len(anomalies):,}")
    c2.metric("Share", f"{len(anomalies) / len(scored):.2%}")
    c3.metric("Segments modeled", len(scored.attrs["thresholds"]))
    st.caption(
        "Distance threshold per segment: "
        + " · ".join(f"{kind} {t:.2f}" for kind, t in scored.attrs["thresholds"].items())
    )
    if txs["fee_estimated"].all():
        st.info("Blocks dataset: gas is the declared limit and the fee is estimated.")

    hist = px.histogram(
        scored,
        x="score",
        color="kind",
        nbins=100,
        log_y=True,
        barmode="overlay",
        title="Distance distribution per transaction segment",
    )
    st.plotly_chart(hist)

    normal = scored[~scored["is_anomaly"] & scored["score"].notna()]
    sample = pd.concat([normal.sample(min(len(normal), 20_000), random_state=0), anomalies])
    st.plotly_chart(
        px.scatter(
            sample,
            x="timestamp",
            y="score",
            color="is_anomaly",
            opacity=0.6,
            color_discrete_map={True: "crimson", False: "steelblue"},
            hover_data=["hash", "kind", "value_eth", "gas_price_gwei"],
            title="Distance per transaction over time",
        ),
    )

    table = anomalies[TABLE_COLUMNS].assign(
        hash=lambda d: ETHERSCAN + "/tx/" + d["hash"],
        **{
            "from": lambda d: ETHERSCAN + "/address/" + d["from"],
            "to": lambda d: ETHERSCAN + "/address/" + d["to"],
        },
    )
    st.dataframe(
        table,
        column_config={
            "hash": st.column_config.LinkColumn(display_text=r".*/tx/(0x.{10})"),
            "from": st.column_config.LinkColumn(display_text=r".*/address/(0x.{8})"),
            "to": st.column_config.LinkColumn(display_text=r".*/address/(0x.{8})"),
        },
        hide_index=True,
    )
    st.download_button(
        "Export CSV",
        anomalies[TABLE_COLUMNS].to_csv(index=False),
        file_name=f"anomalies-{name}.csv",
        mime="text/csv",
    )
