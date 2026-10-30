"""Utilitários compartilhados pelas telas: seleção de dataset e carregamento em cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from tensor_chain.config import get_settings
from tensor_chain.data import store

ETHERSCAN = "https://etherscan.io"


def dataset_path(name: str) -> Path:
    return store.dataset_path(get_settings().data_dir, name)


def dataset_version(name: str) -> float:
    """Muda sempre que uma parte nova é gravada; usado como chave de cache."""
    files = dataset_path(name).rglob("*.parquet")
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_data(show_spinner="Carregando transações…", max_entries=4)
def load_transactions(name: str, version: float) -> pd.DataFrame:
    return store.load_transactions(dataset_path(name))


@st.cache_data(show_spinner=False, max_entries=4)
def load_blocks(name: str, version: float) -> pd.DataFrame:
    return store.load_blocks(dataset_path(name))


def dataset_sidebar() -> str | None:
    names = store.list_datasets(get_settings().data_dir)
    if not names:
        st.sidebar.caption("Nenhum dataset coletado ainda.")
        return None
    # Uma coleta recém-terminada pede para selecionar o dataset novo (ver select_next).
    if "next_dataset" in st.session_state:
        st.session_state["dataset"] = st.session_state.pop("next_dataset")
    if st.session_state.get("dataset") not in names:
        st.session_state["dataset"] = names[-1]
    return st.sidebar.selectbox("Dataset", names, key="dataset")


def select_next(name: str) -> None:
    st.session_state["next_dataset"] = name
    st.rerun()


def require_dataset() -> tuple[str, pd.DataFrame] | None:
    name = st.session_state.get("dataset")
    if not name:
        st.info("Colete um dataset na tela **Dados** para começar.")
        return None
    txs = load_transactions(name, dataset_version(name))
    if txs.empty:
        st.warning(f"O dataset **{name}** ainda não tem transações.")
        return None
    return name, txs


def short(address: str) -> str:
    return address if address.startswith("__") else f"{address[:6]}…{address[-4:]}"
