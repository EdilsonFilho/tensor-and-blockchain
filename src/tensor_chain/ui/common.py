"""Helpers shared by the views: dataset selection and cached loading."""

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
    """Changes whenever a new part is written; used as a cache key."""
    files = dataset_path(name).rglob("*.parquet")
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_data(show_spinner="Loading transactions…", max_entries=4)
def load_transactions(name: str, version: float) -> pd.DataFrame:
    return store.load_transactions(dataset_path(name))


@st.cache_data(show_spinner=False, max_entries=4)
def load_blocks(name: str, version: float) -> pd.DataFrame:
    return store.load_blocks(dataset_path(name))


def dataset_sidebar() -> str | None:
    names = store.list_datasets(get_settings().data_dir)
    if not names:
        st.sidebar.caption("No dataset collected yet.")
        return None
    # A collection that just finished asks to select the new dataset (see select_next).
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
        st.info("Collect a dataset on the **Data** page to get started.")
        return None
    txs = load_transactions(name, dataset_version(name))
    if txs.empty:
        st.warning(f"Dataset **{name}** has no transactions yet.")
        return None
    return name, txs


def short(address: str) -> str:
    return address if address.startswith("__") else f"{address[:6]}…{address[-4:]}"
