import streamlit as st
import shared_components.venn_diagram

def app() -> None:
    st.title("Comparison Viewer")
    st.pyplot(gen_venn_diagram())
