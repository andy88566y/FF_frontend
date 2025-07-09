import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import INFERENCE_DEFAULT_RESULT_DIR, ResultViewerComponents
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import helper, prob_2d_distribution_fig


def app() -> None:
    logger.debug("Loading comparison viewer ...")
    st.title("Comparison Viewer")
    st.caption("2D Model Comparison")

    result_dir_1_col, result_dir_2_col = st.columns([3, 3])

    with result_dir_1_col:
        result_dir_1 = st.text_input("Model 1 (Base) Result Directory", value=INFERENCE_DEFAULT_RESULT_DIR)
    with result_dir_2_col:
        result_dir_2 = st.text_input("Model 2 (Candidate) Result Directory", value=INFERENCE_DEFAULT_RESULT_DIR)

    # Early return if any output dir is invalid
    if not helper.is_valid_output_dir(result_dir_1):
        st.error(f"Model 1 (Base) Result Directory is invalid: {result_dir_1}")
        return
    elif not helper.is_valid_output_dir(result_dir_2):
        st.error(f"Model 2 (Candidate) Result Directory is invalid: {result_dir_2}")
        return

    try:
        result_viewer_components = api_helper.get_result_viewer_components(
            inference_result_dir=result_dir_1,
            secondary_inference_result_dir=result_dir_2,
            recipe=None,
            required_components=[ResultViewerComponents.TWO_D_DEFECT_DISTRIBUTION_CHART.value],
            required_input=None,
        )
    except ValueError as e:
        st.error(e)
        return

    agg_list_1 = result_viewer_components["two_d_defect_distribution_chart"]["recipe_1_aggregate_list"]
    agg_list_2 = result_viewer_components["two_d_defect_distribution_chart"]["recipe_2_aggregate_list"]
    model_count_1 = result_viewer_components["two_d_defect_distribution_chart"]["model_count_1"]
    model_count_2 = result_viewer_components["two_d_defect_distribution_chart"]["model_count_2"]
    threshold_1 = result_viewer_components["two_d_defect_distribution_chart"]["threshold_1"]
    threshold_2 = result_viewer_components["two_d_defect_distribution_chart"]["threshold_2"]

    # Early return if any recipe has more than 1 model
    if model_count_1 != 1:
        st.error("Base Model recipe has more than 1 model.")
        return
    elif model_count_2 != 1:
        st.error("Candidate Model recipe has more than 1 model.")
        return

    input_m1_threshold_col, input_m2_threshold_col = st.columns(2)
    with input_m1_threshold_col:
        input_m1_threshold = st.number_input(
            label="Model 1 threshold:",
            value=threshold_1,
            step=1e-5,
            format="%.5f",
            help="Probabilities below threshold will be considered as non-defects.",
        )
    with input_m2_threshold_col:
        input_m2_threshold = st.number_input(
            label="Model 2 threshold:",
            value=threshold_2,
            step=1e-5,
            format="%.5f",
            help="Probabilities below threshold will be considered as non-defects.",
        )

    _, plot_container, _ = st.columns([1, 8, 1])
    with plot_container:
        st.plotly_chart(
            prob_2d_distribution_fig.gen(
                aggregated_model_data_1=agg_list_1,
                aggregated_model_data_2=agg_list_2,
                m1_threshold=input_m1_threshold,
                m2_threshold=input_m2_threshold,
            )
        )
