
import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers.api_helper import MultiLotModelData
from ltt_ff_frontend.shared_components import class_type_component, multi_lot_stats, prob_distribution_fig, roc_fig


def app(
    inference_result_dir: str,
    recipe: dict[str, list[dict[str, str]]],
    multi_lot_model_data: MultiLotModelData
) -> None:
    logger.debug("Loading Multi Lot Result Viewer...")

    model_metadata_list = multi_lot_model_data.model_metadata_list

    # Defining columns to display filter results (capture rate, filter rate, etc.)
    with st.container():
        r3_header = st.empty()
        model_statistics = st.empty()
    with st.container():
        classtype_count = st.empty()

    st.divider()

    # Show Total/Defect/Non-defect/unlabeled count
    with r3_header:
        st.subheader("Recipe Results")

    with model_statistics.container():
        selected_lot_id_list = multi_lot_stats.draw_stats_df(
            multi_lot_model_data,
            recipe,
            inference_result_dir,
            key="recipe_stats_df"
        )

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
                    recipe['recipes'][0]['threshold'],
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
