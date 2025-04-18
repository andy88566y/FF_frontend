import streamlit as st
import yaml
from loguru import logger
from typing import Any

from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.helpers.api_helper import MultiLotModelData
from ltt_ff_frontend.shared_components import multi_lot_stats, prob_distribution_fig, roc_fig, class_type_component

def app(
    default_output_dir: str,
    inference_result_dir: str,
    recipe: dict[str, list[dict[str, str]]]
) -> None:
    logger.debug("Loading Multi Lot Result Viewer...")

    multi_lot_model_data = api_helper.get_multilot_model_data(inference_result_dir)
    if multi_lot_model_data is None:
        with error_msg_container:
            st.error(f"Error getting result data from {inference_result_dir}")
            return
    model_metadata_list = multi_lot_model_data.model_metadata_list

    # Columns for printing Model info for 1 or 2 models (Model #1/2, model name, threshold, lot ID + gen lrf button)
    vr1_col1, vr1_col2, vr1_col3, vr1_col4 = st.columns([1, 3, 3, 4])
    with st.container():
        model_info_divider = st.empty()
    vr2_col1, vr2_col2, vr2_col3, vr2_col4 = st.columns([1, 3, 3, 4])

    st.divider()

    # Defining columns to display filter results (capture rate, filter rate, etc.)
    with st.container():
        r3_header = st.empty()
        model_statistics = st.empty()
    with st.container():
        r4_header = st.empty()
        model_2_statistics = st.empty()
    with st.container():
        classtype_count = st.empty()

    st.divider()

    # Show Total/Defect/Non-defect/unlabeled count
    with r3_header:
        st.subheader(f"Recipe Results")

    with model_statistics.container():
        selected_lot_id_list = multi_lot_stats.draw_stats_df(multi_lot_model_data, 0.5, key=f"recipe_stats_df")

    # TODO: Get classtype grouping from backend
    with classtype_count:
        with st.expander(label="LRF ClassType count"):
            class_type_component.gen(inference_result_dir, model_metadata_list)
    if recipe is not None and len(recipe['recipes']) == 1:
        # Columns for drawing distribution chart and ROC curve
        col_1d_chart_column, col_roc_curve_column = st.columns(2)
        model_raw_data = (
            multi_lot_model_data.defect_id_lists,
            multi_lot_model_data.probability_lists,
            multi_lot_model_data.answer_lists
        )
        # Draw 1D comparison chart
        with col_1d_chart_column:
            st.plotly_chart(
                prob_distribution_fig.generate_multilot_1D_plot(
                    model_raw_data,
                    model_metadata_list,
                    0.5, 
                    selected_lot_id_list
                )
            )
        with col_roc_curve_column:
            roc_fig.gen_fig(inference_result_dir, 
                model_raw_data, 
                model_metadata_list, 
                0.5, 
                selected_lot_id_list
            )
    st.divider()