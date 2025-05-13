import streamlit as st
from loguru import logger

from ltt_ff_frontend.comparison_ui import comparison_viewer
from ltt_ff_frontend.defect_review_ui import defect_review_gui
from ltt_ff_frontend.defect_ui import (
    inference_api,
    lrf_processor,
    model_converter,
    retraining_api,
)
from ltt_ff_frontend.result_viewer import result_viewer


if __name__ == "__main__":
    logger.debug("Loading main UI ...")

    page_result_viewer = st.Page(
        result_viewer.app,
        url_path="result_viewer",
        title="Result Viewer",
        icon=":material/search_check_2:",
    )
    page_inference_api = st.Page(
        inference_api.app,
        url_path="inference_api",
        title="Inference",
        icon=":material/action_key:",
    )
    page_comparison = st.Page(
        comparison_viewer.app, url_path="comparison", title="Model Comparison", icon=":material/compare_arrows:"
    )
    page_defect_review_gui = st.Page(
        defect_review_gui.app, url_path="defect_review", title="Defect Review", icon=":material/image_search:"
    )
    page_retraining = st.Page(retraining_api.app, url_path="retraining", title="Retraining", icon=":material/build:")
    page_model_converter = st.Page(
        model_converter.app, url_path="model_converter", title="Model Converter", icon=":material/swap_horiz:"
    )
    page_lrf_processor = st.Page(
        lrf_processor.app, url_path="lrf_processor", title="LRF Processor", icon=":material/description:"
    )

    pg = st.navigation(
        [
            page_result_viewer,
            page_inference_api,
            page_comparison,
            page_defect_review_gui,
            page_retraining,
            page_model_converter,
            page_lrf_processor,
        ]
    )

    st.set_page_config(
        page_title="Lasertec Defect Filter UI",
        page_icon=":material/manufacturing:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.sidebar.markdown("###### FF-FE v0.8.0 IH")

    pg.run()
