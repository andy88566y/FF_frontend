import streamlit as st

from ltt_ff_frontend.shared_components import venn_diagram


def app() -> None:
    st.title("Comparison Viewer")
    venn_diagram.gen()
