import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import ui_helper, api_helper

# TODO: st_gen_lrf_type should be constraint by CONSTANT
def app(output_dir_default: str, inference_result_dir: str, st_gen_lrf_type: str) -> None:
    # Column for printing error message
    error_msg_container, _ = st.columns([3, 2])
    invalid_input = [output_dir_default, ""]

    if inference_result_dir in invalid_input:
        with error_msg_container:
            st.error(f"Inference Result Directory is invalid.")
        return

    model_metadata, model_raw_data = api_helper.get_model_data(inference_result_dir)
    model_count = model_metadata.get('model_count', 0)
    if model_metadata is None:
        with error_msg_container:
            st.error(f"Error getting result data from {inference_result_dir}")
            return
    
    for i in range(model_count):
        model_name = model_metadata.get(f"model_name_{i}", "")
        model_threshold = model_metadata.get(f"model_threshold_{i}", "")
        col1, col2, col3, col4, col5 = st.columns([1, 2, 4, 2, 2])
        # Show result database details
        with col1:
            # TODO: check what this "(Base)" really means
            st.text(f"Model {i+1} (Base)")
        with col2:
            st.text(f"Lot ID:\n{model_metadata['lot_id']}")
        with col3:
            st.text(f"Inference Model:\n{ui_helper.format_model_name(model_name)}")

        # Generate lrf by top k, and adjust threshold according to top k
        if st_gen_lrf_type == "top_k":
            with col4:
                rv_m1_topk = st.number_input(
                    "Top k",
                    0,
                    999,
                    150,
                    1,
                    help="Top-k defects ranked by Probabilities will be considered as defects.",
                    key="m1_topk",
                )
            with col5:
                api_helper.gen_lrf(i+1, inference_result_dir, st_gen_lrf_type, top_k=rv_m1_topk)

            model_threshold = api_helper.get_topk_model_threshold(inference_result_dir, rv_m1_topk)

        # Generate lrf by threshold
        elif st_gen_lrf_type == 'threshold':
            with col4:
                model_threshold = st.number_input(
                    label="Confidence threshold:",
                    value=model_threshold,
                    step=0.00001,
                    format="%.5f",
                    min_value=0.0,
                    max_value=1.0,
                    help="Probabilities above threshold will be considered as defects.",
                    key=f"input_model_threshold_{i}",
                )

            # Validate confidence threshold
            if model_threshold < 0.0 or model_threshold > 1.0:
                logger.error(
                    f"Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {model_threshold}"
                )
                st.error(
                    f"Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {model_threshold}"
                )
                return

            with col5:
                api_helper.gen_lrf(
                    i+1, 
                    inference_result_dir, 
                    st_gen_lrf_type, 
                    threshold=model_threshold, 
                    key_number=i+1)
        else:
            logger.error("unknown st_gen_lrf_type")
            st.error("error when getting st_gen_lrf_type")
            return
        # Show Total/Defect/Non-defect/unlabeled count
        with st.container():
            col1, col2, col3, col4 = st.columns([4, 3, 3, 2])
        
        logger.debug(len(model_raw_data))
        count_rate_data = ui_helper.calculate_filtered_results(model_raw_data, model_threshold)
        with col1:
            st.warning(f"""**Total defect count**: As-is: {count_rate_data['as_is_defect_count']}
                    → To-be: {count_rate_data['to_be_defect_count']}
                    \n(Filter Rate: {count_rate_data['filter_rate']:.4f})""")
        with col2:
            st.error(f"""**True defect count**: {count_rate_data['as_is_true_defect_count']}
                    → {count_rate_data['to_be_true_defect_count']}
                    \n(Capture Rate: {count_rate_data['capture_rate']:.4f})""")
        with col3:
            st.success(f"""**Non-defect count**: {count_rate_data['as_is_non_defect_count']}
                    → {count_rate_data['to_be_non_defect_count']}
                    \n(False Filter Rate: {count_rate_data['false_filter_rate']:.4f})""")
        with col4:
            st.info(f"""**Unlabeled count**: {count_rate_data['unlabeled']}
                    → {count_rate_data['filtered_unlabeled_defect_count']}""")

    
    # Defining columns to display filter results (capture rate, filter rate, etc.)
    # with st.container():
    #     classtype_count_container = st.empty()
    # TODO: Get classtype grouping from backend
    with st.container():
        with st.expander(label="LRF ClassType count"):
            defects = api_helper.get_lrf_data_lists(
                output_dir=inference_result_dir, cols=["ClassType"], include_prob=False
            )[0]
            classtype_counter_df = ui_helper.get_classtype_count(defects)
            st.caption(f"LRF type: {model_metadata['input_lrf_type']}")
            st.dataframe(data=classtype_counter_df)

    if model_count == 1:
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