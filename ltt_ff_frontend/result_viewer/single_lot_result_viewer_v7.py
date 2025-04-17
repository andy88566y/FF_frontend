import streamlit as st

from typing import Any
from loguru import logger
from ltt_ff_frontend.helpers import ui_helper, api_helper
from ltt_ff_frontend.helpers.api_helper import MultiLotModelData
from ltt_ff_frontend.shared_components import multi_lot_stats

def app(output_dir_default: str, inference_result_dir: str, recipe: Any) -> None:
    # Column for printing error message
    error_msg_container, _ = st.columns([3, 2])
    invalid_input = [output_dir_default, ""]

    if inference_result_dir in invalid_input:
        with error_msg_container:
            st.error(f"Inference Result Directory is invalid.")
        return

    model_metadata, model_raw_data = api_helper.get_model_data(inference_result_dir)
    if model_metadata is None:
        with error_msg_container:
            st.error(f"Error getting result data from {inference_result_dir}")
            return
    
    # Show result database details
    st.subheader(f"Lot ID: {model_metadata['lot_id']}")

    # Show Total/Defect/Non-defect/unlabeled count
    st.text("Inference results")
    multi_lot_model_data = api_helper.get_multilot_model_data(inference_result_dir)
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
            defects = api_helper.get_lrf_data_lists(
                output_dir=inference_result_dir, cols=["ClassType"], include_prob=False
            )[0]
            classtype_counter_df = ui_helper.get_classtype_count(defects)
            st.caption(f"LRF type: {model_metadata['input_lrf_type']}")
            st.dataframe(data=classtype_counter_df)

    if model_metadata.get('model_count', 0) == 1:
    # Columns for drawing distribution chart and ROC curve
        col_1d_chart, col_roc_curve = st.columns(2)
        # Draw 1D comparison chart
        with col_1d_chart:
            st.plotly_chart(ui_helper.generate_1D_plot(model_raw_data, st.session_state["input_model_threshold_0"]))
        with col_roc_curve:
            # TODO: This should be done somewhere else
            if 1 not in set(model_raw_data[2]):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")
            else:
                model_roc_data = api_helper.get_roc_data(inference_result_dir, return_curve=True)[0]
                st.plotly_chart(
                    ui_helper.plot_roc(
                        [
                            (
                                "Model 1",
                                model_roc_data,
                                st.session_state["input_model_threshold_0"],
                                model_metadata.get("model_threshold_0", ""),
                                inference_result_dir,
                            )
                        ]
                    )
                )
    st.divider()