import streamlit as st


def app() -> None:
    st.title("Comparison Viewer")
    st.pyplot(gen_venn_diagram())
