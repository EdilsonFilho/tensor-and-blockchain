"""Tensor page: builds 𝒳, 𝒴 and N and lets you inspect slices."""

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

METRIC_LABELS = {"count": "count", "value": "value (gwei)", "fee": "fee paid (gwei)"}


@st.cache_data(show_spinner="Building tensors…", max_entries=8)
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
    top_k = c1.number_input("Top-K addresses", 10, 5_000, 200, step=50)
    freq = c2.selectbox("Time window", FREQS, index=1)
    metric = c3.selectbox("Flow metric", FLOW_METRICS, format_func=METRIC_LABELS.get)
    include_other = c4.checkbox(
        '"Other" bucket', value=True, help="Aggregates addresses outside the top-K into one index."
    )

    flow, feats, network = _build(
        name, dataset_version(name), int(top_k), freq, metric, include_other
    )
    a, _, t = flow.data.shape
    c1, c2, c3 = st.columns(3)
    c1.metric("𝒳 from × to × time", f"{a} × {a} × {t}")
    c2.metric("Non-zero cells", f"{flow.data.nnz:,}")
    c3.metric("Density", f"{flow.density:.2e}")

    st.subheader("𝒳 — flow between addresses")
    st.caption(f"Values in log(1 + {METRIC_LABELS[metric]}).")
    totals = pd.Series(flow.data.sum(axis=(0, 1)).todense(), index=flow.times)
    st.plotly_chart(
        px.bar(
            totals, labels={"index": "", "value": "sum"}, title="Tensor mass per window"
        ).update_layout(showlegend=False)
    )

    n = st.slider("Addresses in slice", 1, min(100, a), min(30, a)) if a > 1 else a
    when = (
        st.select_slider(
            "Window", options=list(range(t)), format_func=lambda i: f"{flow.times[i]:%b %d %H:%M}"
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
            labels={"x": "to", "y": "from", "color": "value"},
            color_continuous_scale="Blues",
            aspect="auto",
        ),
    )

    st.subheader("𝒴 — features per address")
    st.caption(
        f"{feats.data.shape[0]} × {feats.data.shape[1]} × {feats.data.shape[2]} "
        "(address × time × feature)"
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

    st.subheader("N — network metrics")
    long = network.reset_index(names="time").melt("time", var_name="metric")
    fig = px.line(long, x="time", y="value", facet_row="metric", height=160 * network.shape[1])
    fig.update_yaxes(matches=None, title=None).for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1])
    )
    st.plotly_chart(fig)
