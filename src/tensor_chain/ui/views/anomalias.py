"""Tela Anomalias: baseline de Mahalanobis robusto por transação."""

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


@st.cache_data(show_spinner="Calculando distâncias…", max_entries=8)
def _score(name: str, version: float, alpha: float) -> pd.DataFrame:
    return mahalanobis_anomalies(load_transactions(name, version), alpha)


def render() -> None:
    st.title("Anomalias (baseline)")
    st.caption(
        "Mahalanobis robusto (MCD) sobre log(valor em wei), log(gas) e log(gas price), com um "
        "modelo por tipo de transação. Uma transação é anômala quando o p-valor χ² fica "
        "abaixo de α."
    )
    loaded = require_dataset()
    if loaded is None:
        return
    name, txs = loaded

    alpha = st.select_slider(
        "α (nível de significância)", ALPHAS, value=1e-3, format_func=lambda a: f"{a:.0e}"
    )
    scored = _score(name, dataset_version(name), alpha)
    anomalies = scored[scored["is_anomaly"]].sort_values("score", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalias", f"{len(anomalies):,}")
    c2.metric("Proporção", f"{len(anomalies) / len(scored):.2%}")
    c3.metric("Tipos modelados", len(scored.attrs["thresholds"]))
    st.caption(
        "Limiar de distância por tipo: "
        + " · ".join(f"{kind} {t:.2f}" for kind, t in scored.attrs["thresholds"].items())
    )
    if txs["fee_estimated"].all():
        st.info("Dataset por blocos: o gas usado é o limite declarado e a taxa é estimada.")

    hist = px.histogram(
        scored,
        x="score",
        color="kind",
        nbins=100,
        log_y=True,
        barmode="overlay",
        title="Distribuição das distâncias por tipo de transação",
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
            title="Distância por transação ao longo do tempo",
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
        "Exportar CSV",
        anomalies[TABLE_COLUMNS].to_csv(index=False),
        file_name=f"anomalias-{name}.csv",
        mime="text/csv",
    )
