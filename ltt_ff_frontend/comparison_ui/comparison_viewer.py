import streamlit as st
from loguru import logger

from ltt_ff_frontend.shared_components import prob_2d_distribution_fig


def app() -> None:
    logger.debug("Loading comparison viewer ...")
    st.title("Comparison Viewer")
    st.caption("Visualization of comparing 2 inference results.")

    col1, col2 = st.columns([3, 3])

    output_dir_default = "/mnt/dbpc/xxx"

    with col1:
        result_dir_1 = st.text_input("Model 1 (Base) Result Directory", value=output_dir_default)
    with col2:
        result_dir_2 = st.text_input("Model 2 (Candidate) Result Directory", value=output_dir_default)

    # Columns for drawing distribution chart and ROC curve
    col_1d_chart_column, col_roc_curve_column = st.columns(2)

    prob_2d_distribution_fig.gen(result_dir_1=result_dir_1, result_dir_2=result_dir_2)
