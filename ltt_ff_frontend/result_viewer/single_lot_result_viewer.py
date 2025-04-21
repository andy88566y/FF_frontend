from typing import Any

import streamlit as st
import yaml
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.helpers.api_helper import MultiLotModelData
from ltt_ff_frontend.shared_components import multi_lot_stats, prob_distribution_fig, class_type_component, roc_fig
from loguru import logger


def app(
    output_dir_default: str,
    inference_result_dir: str,
    recipe: dict[str, list[dict[str, str]]],
    multi_lot_model_data: MultiLotModelData
) -> None:
    logger.debug("Loading Single Lot Result Viewer...")
    model_metadata = multi_lot_model_data.model_metadata_list[0]
    # TODO: refactor all functions using model_raw_data to explicitly named argument
    model_raw_data = (
        multi_lot_model_data.defect_id_lists[0],
        multi_lot_model_data.probability_lists[0],
        multi_lot_model_data.answer_lists[0]
    )

    # Show result database details
    st.subheader(f"Lot ID: {model_metadata['lot_id']}")

    # Show Total/Defect/Non-defect/unlabeled count
    st.text("Inference results")
    count_rate_data = api_helper.calculate_recipe_filtered_results(inference_result_dir, recipe=recipe)
    data_list = [[
        model_metadata['lot_id'],
        count_rate_data['as_is_defect_count'],
        count_rate_data['to_be_defect_count'],
        count_rate_data['filter_rate'],
        count_rate_data['as_is_true_defect_count'],
        count_rate_data['to_be_true_defect_count'],
        count_rate_data['capture_rate'],
        count_rate_data['as_is_non_defect_count'],
        count_rate_data['to_be_non_defect_count'],
        count_rate_data['false_filter_rate'],
        count_rate_data['unlabeled'],
        count_rate_data['filtered_unlabeled_defect_count']
    ]]
    multi_lot_stats.gen_stats_df_by_data_list(data_list, key="single_lot_df")

    # TODO: Get classtype grouping from backend
    with st.container():
        with st.expander(label="LRF ClassType count"):
            class_type_component.gen(inference_result_dir, [model_metadata])

    if recipe is not None and len(recipe['recipes']) == 1:
    # Columns for drawing distribution chart and ROC curve
        col_1d_chart, col_roc_curve = st.columns(2)
        # Draw 1D comparison chart
        with col_1d_chart:
            st.plotly_chart(prob_distribution_fig.generate_1D_plot(model_raw_data, recipe['recipes'][0]['threshold']))
        with col_roc_curve:
            # TODO: This should be done somewhere else
            if 1 not in set(model_raw_data[2]):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")
            else:
                model_roc_data = api_helper.get_roc_data(inference_result_dir, return_curve=True)[0]
                st.plotly_chart(
                    roc_fig.plot_roc(
                        [
                            (
                                "Model 1",
                                model_roc_data,
                                recipe['recipes'][0]['threshold'],
                                model_metadata.get("model_threshold_0", ""),
                                inference_result_dir,
                            )
                        ]
                    )
                )
    st.divider()
