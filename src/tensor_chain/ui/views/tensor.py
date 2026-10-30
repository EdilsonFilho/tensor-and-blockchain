"""Tela Tensor: monta 𝒳, 𝒴 e N e permite inspecionar fatias."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from tensor_chain.tensors.build import (
    FLOW_METRICS,
    FREQS,
    FeatureTensor,
    FlowTensor,
    build_feature_tensor,
    build_flow_tensor,
    build_network_matrix,
    top_addresses,
)
from tensor_chain.ui.common import (
    dataset_version,
    load_blocks,
    load_transactions,
    require_dataset,
    short,
)

METRIC_LABELS = {"count": "contagem", "value": "valor (gwei)", "fee": "taxa paga (gwei)"}


@st.cache_data(show_spinner="Montando tensores…", max_entries=8)
def _build(
    name: str, version: float, top_k: int, freq: str, metric: str, include_other: bool
) -> tuple[FlowTensor, FeatureTensor, pd.DataFrame]:
    txs = load_transactions(name, version)
    addresses = top_addresses(txs, top_k)
    flow = build_flow_tensor(txs, addresses, freq, metric, include_other=include_other)
    feats = build_feature_tensor(txs, addresses, freq)
    network = build_network_matrix(txs, load_blocks(name, version), flow.times, freq)
    return flow, feats, network


def render() -> None:
    st.title("Tensor")
    loaded = require_dataset()
    if loaded is None:
        return
    name, _ = loaded

    c1, c2, c3, c4 = st.columns(4)
    top_k = c1.number_input("Top-K endereços", 10, 5_000, 200, step=50)
    freq = c2.selectbox("Janela de tempo", FREQS, index=1)
    metric = c3.selectbox("Métrica do fluxo", FLOW_METRICS, format_func=METRIC_LABELS.get)
    include_other = c4.checkbox(
        'Bucket "outros"', value=True, help="Agrega os endereços fora do top-K em um único índice."
    )

    flow, feats, network = _build(
        name, dataset_version(name), int(top_k), freq, metric, include_other
    )
    a, _, t = flow.data.shape
    c1, c2, c3 = st.columns(3)
    c1.metric("𝒳 from × to × tempo", f"{a} × {a} × {t}")
    c2.metric("Células não nulas", f"{flow.data.nnz:,}")
    c3.metric("Densidade", f"{flow.density:.2e}")

    st.subheader("𝒳 — fluxo entre endereços")
    st.caption(f"Valores em log(1 + {METRIC_LABELS[metric]}).")
    totals = pd.Series(flow.data.sum(axis=(0, 1)).todense(), index=flow.times)
    st.plotly_chart(
        px.bar(
            totals, labels={"index": "", "value": "soma"}, title="Massa do tensor por janela"
        ).update_layout(showlegend=False)
    )

    n = st.slider("Endereços na fatia", 1, min(100, a), min(30, a)) if a > 1 else a
    when = (
        st.select_slider(
            "Janela", options=list(range(t)), format_func=lambda i: f"{flow.times[i]:%d/%m %H:%M}"
        )
        if t > 1
        else 0
    )
    labels = [short(x) for x in flow.addresses[:n]]
    slice_ = flow.data[:n, :n, when].todense()
    st.plotly_chart(
        px.imshow(
            slice_,
            x=labels,
            y=labels,
            labels={"x": "to", "y": "from", "color": "valor"},
            color_continuous_scale="Blues",
            aspect="auto",
        ),
    )

    st.subheader("𝒴 — features por endereço")
    st.caption(
        f"{feats.data.shape[0]} × {feats.data.shape[1]} × {feats.data.shape[2]} "
        "(endereço × tempo × feature)"
    )
    feature = st.selectbox("Feature", feats.features)
    rows = min(50, len(feats.addresses))
    st.plotly_chart(
        px.imshow(
            feats.data[:rows, :, feats.features.index(feature)],
            x=feats.times,
            y=[short(x) for x in feats.addresses[:rows]],
            labels={"x": "", "y": "", "color": feature},
            color_continuous_scale="Viridis",
            aspect="auto",
        ),
    )

    st.subheader("N — métricas da rede")
    long = network.reset_index(names="tempo").melt("tempo", var_name="métrica")
    fig = px.line(long, x="tempo", y="value", facet_row="métrica", height=160 * network.shape[1])
    fig.update_yaxes(matches=None, title=None).for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1])
    )
    st.plotly_chart(fig)
