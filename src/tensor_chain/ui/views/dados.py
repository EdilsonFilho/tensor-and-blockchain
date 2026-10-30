"""Tela Dados: coleta via Etherscan V2 e resumo do dataset selecionado."""

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
    if st.button("Buscar bloco mais recente"):
        try:
            st.session_state["latest_block"] = _client().latest_block()
        except EtherscanError as exc:
            st.error(str(exc))
    latest = st.session_state.get("latest_block")
    if latest:
        st.caption(f"Bloco mais recente: {latest:,}")

    with st.form("blocks"):
        c1, c2, c3 = st.columns(3)
        count = c2.number_input("Quantidade de blocos", 1, 50_000, 300)
        step = c3.number_input("Amostrar 1 a cada N blocos", 1, 1_000, 1)
        default_start = max(0, latest - count * step) if latest else 21_000_000
        start = c1.number_input("Bloco inicial", 0, value=default_start)
        name = st.text_input("Nome do dataset (opcional)")
        minutes = count / settings.rate_limit / 60
        span_h = count * step / BLOCKS_PER_DAY * 24
        st.caption(
            f"{count:,} chamadas ≈ {minutes:.0f} min de coleta · cobre ≈ {span_h:.1f} h de rede. "
            "O gas de cada transação é o limite declarado (gasUsed não vem neste modo)."
        )
        submitted = st.form_submit_button("Coletar", type="primary")

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
            collect_blocks(_client(), path, numbers, progress=_progress_callback("Blocos"))
        except (EtherscanError, ValueError) as exc:
            st.error(str(exc))
            return
        select_next(name)


def _addresses_form() -> None:
    settings = get_settings()
    with st.form("addresses"):
        text = st.text_area("Endereços-semente (um por linha)")
        c1, c2 = st.columns(2)
        start = c1.number_input("Bloco inicial", 0, value=0)
        end = c2.number_input("Bloco final", 0, value=99_999_999)
        name = st.text_input("Nome do dataset")
        st.caption(
            "Histórico completo com gasUsed e status. A API devolve 1000 transações por "
            "chamada; a paginação é automática."
        )
        submitted = st.form_submit_button("Coletar", type="primary")

    if submitted:
        addresses = [a.strip().lower() for a in text.splitlines() if a.strip()]
        if not addresses or not name:
            st.error("Informe ao menos um endereço e o nome do dataset.")
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
                progress=_progress_callback("Endereços"),
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
    mode = "blocos (rede toda)" if meta["mode"] == "blocks" else "endereços-semente"
    st.caption(f"Modo: {mode} · blocos {meta['start_block']:,} a {meta['end_block']:,}")
    if meta["mode"] == "blocks":
        expected = len(range(meta["start_block"], meta["end_block"] + 1, meta["step"]))
        if len(blocks) < expected:
            st.warning(
                f"Coleta incompleta: {len(blocks):,} de {expected:,} blocos. "
                "Coletar de novo com os mesmos parâmetros retoma de onde parou."
            )

    addresses = set(txs["from"]) | set(txs["to"].dropna())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transações", f"{len(txs):,}")
    c2.metric("Endereços únicos", f"{len(addresses):,}")
    c3.metric("Chamadas a contrato", f"{txs['is_contract_call'].mean():.0%}")
    c4.metric(
        "Período", f"{txs['timestamp'].min():%d/%m %H:%M} – {txs['timestamp'].max():%d/%m %H:%M}"
    )

    span = txs["timestamp"].max() - txs["timestamp"].min()
    freq = "10min" if span.total_seconds() < 2 * 86_400 else "1h"
    per_bin = txs.set_index("timestamp").resample(freq).size().rename("transações")
    st.plotly_chart(px.line(per_bin, title=f"Transações por janela de {freq}"))

    c1, c2 = st.columns(2)
    paid = txs[txs["value_eth"] > 0]
    c1.plotly_chart(
        px.histogram(
            x=np.log10(paid["value_eth"]),
            nbins=60,
            labels={"x": "log10(valor em ETH)"},
            title="Valor transferido (> 0)",
        ),
    )
    c2.plotly_chart(
        px.histogram(
            x=np.log10(txs["gas_price_gwei"] + 1e-3),
            nbins=60,
            labels={"x": "log10(gas price em gwei)"},
            title="Preço do gas",
        ),
    )

    preview = txs.head(500).assign(hash=lambda d: ETHERSCAN + "/tx/" + d["hash"])
    st.dataframe(
        preview,
        column_config={"hash": st.column_config.LinkColumn(display_text=r".*/tx/(0x.{10})")},
        hide_index=True,
    )


def render() -> None:
    st.title("Dados")
    if not get_settings().api_key:
        st.warning("Defina `ETHERSCAN_API_KEY` no arquivo `.env` para coletar dados.")
    else:
        with st.expander("Nova coleta", expanded=not st.session_state.get("dataset")):
            by_blocks, by_addresses = st.tabs(["Por blocos", "Por endereços"])
            with by_blocks:
                _blocks_form()
            with by_addresses:
                _addresses_form()
    _summary()
