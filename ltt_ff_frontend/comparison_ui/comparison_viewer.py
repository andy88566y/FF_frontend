import streamlit as st
from loguru import logger

from ltt_ff_frontend.comparison_ui import double_directory, single_directory


def app() -> None:
    logger.debug("Loading comparison viewer ...")
    st.title("Comparison Viewer")
    mode = st.segmented_control(label="Mode", options=["Single", "Double"], default="Single")

    if mode is None:
        st.error("Please select a mode!")
        return

    if mode == "Single":
        single_directory.app()
    elif mode == "Double":
        double_directory.app()
