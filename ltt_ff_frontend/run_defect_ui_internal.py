import streamlit as st
from loguru import logger

from ltt_ff_frontend.comparison_ui import comparison_viewer
from ltt_ff_frontend.data_yaml_processor import data_yaml_processor
from ltt_ff_frontend.defect_review_ui import defect_diff_viewer, defect_review_gui
from ltt_ff_frontend.inference import inference
from ltt_ff_frontend.lrf_processor import lrf_processor
from ltt_ff_frontend.model_converter import model_converter
from ltt_ff_frontend.regression_test_ui import regression_result_viewer, regression_test
from ltt_ff_frontend.result_viewer import result_viewer, result_viewer_summary
from ltt_ff_frontend.training import training, training_by_config


if __name__ == "__main__":
    logger.debug("Loading main UI ...")

    page_result_viewer_summary = st.Page(
        result_viewer_summary.app,
        url_path="result_viewer_summary",
        title="Result Viewer Summary",
        icon=":material/search_check_2:",
    )
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
    page_regression_result_viewer = st.Page(
        regression_result_viewer.app,
        url_path="regression_result_viewer",
        title="Regression Result Viewer",
        icon=":material/search_check_2:",
    )
    page_regression_test = st.Page(
        regression_test.app,
        url_path="regression_test",
        title="Regression Test",
        icon=":material/action_key:",
    )
    page_comparison = st.Page(
        comparison_viewer.app, url_path="comparison", title="Model Comparison", icon=":material/compare_arrows:"
    )

    page_defect_diff_viewer = st.Page(
        defect_diff_viewer.app, url_path="defect_diff_viewer", title="Defect Viewer", icon=":material/image_search:"
    )
    page_defect_review_gui = st.Page(
        defect_review_gui.app, url_path="defect_review", title="Defect Review", icon=":material/image_search:"
    )

    page_training = st.Page(training.app, url_path="training", title="Training", icon=":material/build:")
    page_training_by_config = st.Page(
        training_by_config.app, url_path="training_by_config", title="Training By Config", icon=":material/build:"
    )

    page_model_converter = st.Page(
        model_converter.app, url_path="model_converter", title="Model Converter", icon=":material/swap_horiz:"
    )

    page_lrf_processor = st.Page(
        lrf_processor.app, url_path="lrf_processor", title="LRF Processor", icon=":material/description:"
    )
    page_data_yaml_processor = st.Page(
        data_yaml_processor.app, url_path="data_yaml_processor", title="Data Yaml Processor", icon=":material/schema:"
    )

    pg = st.navigation(
        [
            page_result_viewer_summary,
            page_inference,
            page_result_viewer,
            page_comparison,
            page_defect_diff_viewer,
            page_defect_review_gui,
            page_regression_result_viewer,
            page_regression_test,
            page_training,
            page_training_by_config,
            page_model_converter,
            page_lrf_processor,
            page_data_yaml_processor,
        ]
    )

    st.set_page_config(
        page_title="Lasertec Defect Filter UI",
        page_icon=":material/manufacturing:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.sidebar.markdown("###### FF-FE v0.12.0 IH")

    pg.run()
