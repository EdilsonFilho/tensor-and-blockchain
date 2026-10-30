"""Graphical interface. Run with `tensor-chain ui` (or `streamlit run` on this file)."""

import streamlit as st

from tensor_chain.ui.common import dataset_sidebar
from tensor_chain.ui.views import anomalies, data, tensor

st.set_page_config(page_title="Tensor Chain", page_icon="⛓️", layout="wide")

page = st.navigation(
    [
        st.Page(
            data.render, title="Data", icon=":material/database:", url_path="data", default=True
        ),
        st.Page(tensor.render, title="Tensor", icon=":material/view_in_ar:", url_path="tensor"),
        st.Page(
            anomalies.render,
            title="Anomalies (baseline)",
            icon=":material/warning:",
            url_path="anomalies",
        ),
    ]
)
dataset_sidebar()
page.run()
