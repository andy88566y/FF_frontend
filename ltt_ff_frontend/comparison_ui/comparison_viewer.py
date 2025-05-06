import streamlit as st

from loguru import logger
from ltt_ff_frontend.shared_components import venn_diagram
from ltt_ff_frontend.shared_components import (
    prob_2d_distribution_fig,
    helper
)
from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    logger.debug("Loading comparison viewer ...")
    st.title("Comparison Viewer")
    st.caption("Visualization of comparing 2 inference results.")

    col1, col2= st.columns([3, 3])

    output_dir_default = "/mnt/dbpc/xxx"

    with col1:
        result_dir_1 = st.text_input("First Inference Result Directory", value=output_dir_default)
    with col2:
        result_dir_2 = st.text_input("Second Inference Result Directory", value=output_dir_default)
    venn_diagram.gen()
    
    # Columns for drawing distribution chart and ROC curve
    col_1d_chart_column, col_roc_curve_column = st.columns(2)

    if helper.is_valid_output_dir(result_dir_1) and helper.is_valid_output_dir(result_dir_2):
        multi_lot_model_data_1 = api_helper.get_multilot_model_data(result_dir_1)
        multi_lot_model_data_2 = api_helper.get_multilot_model_data(result_dir_2)
        recipe_model_count_1 = multi_lot_model_data_1.model_metadata_list[0]["model_count"]
        recipe_model_count_2 = multi_lot_model_data_2.model_metadata_list[0]["model_count"]

        if recipe_model_count_1 == 1 and recipe_model_count_2 == 1:
            m1_threshold = multi_lot_model_data_1.model_metadata_list[0]['model_threshold_0']
            m2_threshold = multi_lot_model_data_2.model_metadata_list[0]['model_threshold_0']
            aggregated_model_data_1 = helper.aggregate_lists(
                (
                    multi_lot_model_data_1.defect_id_lists,
                    multi_lot_model_data_1.probability_lists,
                    multi_lot_model_data_1.answer_lists,
                ),
                multi_lot_model_data_1.model_metadata_list,
            )
            aggregated_model_data_2 = helper.aggregate_lists(
                (
                    multi_lot_model_data_2.defect_id_lists,
                    multi_lot_model_data_2.probability_lists,
                    multi_lot_model_data_2.answer_lists,
                ),
                multi_lot_model_data_2.model_metadata_list
            )
            _, plot_container, _ = st.columns([1,8,1])
            with plot_container:
                st.plotly_chart(
                    prob_2d_distribution_fig.generate(
                        aggregated_model_data_1,
                        aggregated_model_data_2,
                        m1_threshold,
                        m2_threshold,
                    )
                )
