"""Data page: collection through Etherscan V2 and summary of the selected dataset."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st

from tensor_chain.config import get_settings
from tensor_chain.data import store
from tensor_chain.data.collect import (
    BLOCKS_PER_DAY,
    block_numbers,
    collect_addresses,
    collect_blocks,
)
from tensor_chain.data.etherscan import EtherscanClient, EtherscanError
from tensor_chain.ui.common import (
    ETHERSCAN,
    dataset_path,
    dataset_version,
    load_blocks,
    require_dataset,
    select_next,
)


def _client() -> EtherscanClient:
    settings = get_settings()
    return EtherscanClient(settings.api_key, rate_limit=settings.rate_limit)


def _progress_callback(label: str):
    bar = st.progress(0.0, text=label)

    def update(done: int, total: int) -> None:
        bar.progress(done / total if total else 1.0, text=f"{label} {done}/{total}")

    return update


def _blocks_form() -> None:
    settings = get_settings()
    if st.button("Fetch latest block"):
        try:
            st.session_state["latest_block"] = _client().latest_block()
        except EtherscanError as exc:
            st.error(str(exc))
    latest = st.session_state.get("latest_block")
    if latest:
        st.caption(f"Latest block: {latest:,}")

    with st.form("blocks"):
        c1, c2, c3 = st.columns(3)
        count = c2.number_input("Number of blocks", 1, 50_000, 300)
        step = c3.number_input("Sample 1 every N blocks", 1, 1_000, 1)
        default_start = max(0, latest - count * step) if latest else 21_000_000
        start = c1.number_input("Start block", 0, value=default_start)
        name = st.text_input("Dataset name (optional)")
        minutes = count / settings.rate_limit / 60
        span_h = count * step / BLOCKS_PER_DAY * 24
        st.caption(
            f"{count:,} calls ≈ {minutes:.0f} min of collection · covers ≈ {span_h:.1f} h of "
            "network. Each transaction's gas is the declared limit (gasUsed is not available "
            "in this mode)."
        )
        submitted = st.form_submit_button("Collect", type="primary")

    if submitted:
        numbers = block_numbers(int(start), int(count), int(step))
        name = name or f"blocks-{numbers[0]}-{numbers[-1]}-s{step}"
        meta = {
            "mode": "blocks",
            "start_block": numbers[0],
            "end_block": numbers[-1],
            "step": int(step),
        }
        try:
            path = store.create_dataset(settings.data_dir, name, meta)
            collect_blocks(_client(), path, numbers, progress=_progress_callback("Blocks"))
        except (EtherscanError, ValueError) as exc:
            st.error(str(exc))
            return
        select_next(name)


def _addresses_form() -> None:
    settings = get_settings()
    with st.form("addresses"):
        text = st.text_area("Seed addresses (one per line)")
        c1, c2 = st.columns(2)
        start = c1.number_input("Start block", 0, value=0)
        end = c2.number_input("End block", 0, value=99_999_999)
        name = st.text_input("Dataset name")
        st.caption(
            "Full history with gasUsed and status. The API returns 1000 transactions per "
            "call; pagination is automatic."
        )
        submitted = st.form_submit_button("Collect", type="primary")

    if submitted:
        addresses = [a.strip().lower() for a in text.splitlines() if a.strip()]
        if not addresses or not name:
            st.error("Enter at least one address and the dataset name.")
            return
        meta = {
            "mode": "addresses",
            "start_block": int(start),
            "end_block": int(end),
            "addresses": addresses,
        }
        try:
            path = store.create_dataset(settings.data_dir, name, meta)
            collect_addresses(
                _client(),
                path,
                addresses,
                int(start),
                int(end),
                progress=_progress_callback("Addresses"),
            )
        except (EtherscanError, ValueError) as exc:
            st.error(str(exc))
            return
        select_next(name)


def _summary() -> None:
    loaded = require_dataset()
    if loaded is None:
        return
    name, txs = loaded
    meta = store.read_meta(dataset_path(name))
    blocks = load_blocks(name, dataset_version(name))

    st.subheader(name)
    mode = "blocks (entire network)" if meta["mode"] == "blocks" else "seed addresses"
    st.caption(f"Mode: {mode} · blocks {meta['start_block']:,} to {meta['end_block']:,}")
    if meta["mode"] == "blocks":
        expected = len(range(meta["start_block"], meta["end_block"] + 1, meta["step"]))
        if len(blocks) < expected:
            st.warning(
                f"Incomplete collection: {len(blocks):,} of {expected:,} blocks. "
                "Collecting again with the same parameters resumes where it stopped."
            )

    addresses = set(txs["from"]) | set(txs["to"].dropna())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(txs):,}")
    c2.metric("Unique addresses", f"{len(addresses):,}")
    c3.metric("Contract calls", f"{txs['is_contract_call'].mean():.0%}")
    c4.metric(
        "Period", f"{txs['timestamp'].min():%b %d %H:%M} – {txs['timestamp'].max():%b %d %H:%M}"
    )

    span = txs["timestamp"].max() - txs["timestamp"].min()
    freq = "10min" if span.total_seconds() < 2 * 86_400 else "1h"
    per_bin = txs.set_index("timestamp").resample(freq).size().rename("transactions")
    st.plotly_chart(px.line(per_bin, title=f"Transactions per {freq} window"))

    c1, c2 = st.columns(2)
    paid = txs[txs["value_eth"] > 0]
    c1.plotly_chart(
        px.histogram(
            x=np.log10(paid["value_eth"]),
            nbins=60,
            labels={"x": "log10(value in ETH)"},
            title="Value transferred (> 0)",
        ),
    )
    c2.plotly_chart(
        px.histogram(
            x=np.log10(txs["gas_price_gwei"] + 1e-3),
            nbins=60,
            labels={"x": "log10(gas price in gwei)"},
            title="Gas price",
        ),
    )

    preview = txs.head(500).assign(hash=lambda d: ETHERSCAN + "/tx/" + d["hash"])
    st.dataframe(
        preview,
        column_config={"hash": st.column_config.LinkColumn(display_text=r".*/tx/(0x.{10})")},
        hide_index=True,
    )


def render() -> None:
    st.title("Data")
    if not get_settings().api_key:
        st.warning("Set `ETHERSCAN_API_KEY` in the `.env` file to collect data.")
    else:
        with st.expander("New collection", expanded=not st.session_state.get("dataset")):
            by_blocks, by_addresses = st.tabs(["By blocks", "By addresses"])
            with by_blocks:
                _blocks_form()
            with by_addresses:
                _addresses_form()
    _summary()
