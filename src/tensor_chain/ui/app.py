"""Interface gráfica. Rode com `tensor-chain ui` (ou `streamlit run` neste arquivo)."""

import streamlit as st

from tensor_chain.ui.common import dataset_sidebar
from tensor_chain.ui.views import anomalias, dados, tensor

st.set_page_config(page_title="Tensor Chain", page_icon="⛓️", layout="wide")

page = st.navigation(
    [
        st.Page(
            dados.render, title="Dados", icon=":material/database:", url_path="dados", default=True
        ),
        st.Page(tensor.render, title="Tensor", icon=":material/view_in_ar:", url_path="tensor"),
        st.Page(
            anomalias.render,
            title="Anomalias (baseline)",
            icon=":material/warning:",
            url_path="anomalias",
        ),
    ]
)
dataset_sidebar()
page.run()
