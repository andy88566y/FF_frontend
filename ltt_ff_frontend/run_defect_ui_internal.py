import streamlit as st
from loguru import logger

from ltt_ff_frontend.comparison_ui import comparison_viewer
from ltt_ff_frontend.data_yaml_creator import data_yaml_creator
from ltt_ff_frontend.defect_review_ui import defect_review_gui
from ltt_ff_frontend.inference import inference
from ltt_ff_frontend.lrf_processor import lrf_processor
from ltt_ff_frontend.model_converter import model_converter
from ltt_ff_frontend.result_viewer import result_viewer
from ltt_ff_frontend.training import training, training_by_config


if __name__ == "__main__":
    logger.debug("Loading main UI ...")

    page_result_viewer = st.Page(
        result_viewer.app,
        url_path="result_viewer",
        title="Result Viewer",
        icon=":material/search_check_2:",
    )
    page_inference = st.Page(
        inference.app,
        url_path="inference",
        title="Inference",
        icon=":material/action_key:",
    )
    page_comparison = st.Page(
        comparison_viewer.app, url_path="comparison", title="Model Comparison", icon=":material/compare_arrows:"
    )
    page_defect_review_gui = st.Page(
        defect_review_gui.app, url_path="defect_review", title="Defect Review", icon=":material/image_search:"
    )
    page_training = st.Page(training.app, url_path="training", title="Training", icon=":material/build:")
    page_training_by_config = st.Page(training_by_config.app, url_path="training_by_config", title="Training By Config", icon=":material/build:")
    page_model_converter = st.Page(
        model_converter.app, url_path="model_converter", title="Model Converter", icon=":material/swap_horiz:"
    )
    page_lrf_processor = st.Page(
        lrf_processor.app, url_path="lrf_processor", title="LRF Processor", icon=":material/description:"
    )
    page_data_yaml_creator = st.Page(
        data_yaml_creator.app, url_path="data_yaml_creator", title="Data Yaml Creator", icon=":material/schema:"
    )

    pg = st.navigation(
        [
            page_result_viewer,
            page_inference,
            page_comparison,
            page_defect_review_gui,
            page_training,
            page_training_by_config,
            page_model_converter,
            page_lrf_processor,
            page_data_yaml_creator,
        ]
    )

    st.set_page_config(
        page_title="Lasertec Defect Filter UI",
        page_icon=":material/manufacturing:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.sidebar.markdown("###### FF-FE v0.9.0 IH")

    pg.run()
