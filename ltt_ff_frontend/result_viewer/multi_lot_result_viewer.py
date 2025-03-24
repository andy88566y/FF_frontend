import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper
from ltt_ff_frontend.result_viewer import result_viewer


def app(output_dir_default: str, rv_m1_output_dir: str, rv_m2_output_dir: str, st_gen_lrf_type: str) -> None:

    # Column for printing error message
    r2_col1, _r2_col2 = st.columns([3, 2])

    # Columns for printing Model info for 1 or 2 models (lot ID, model name, threshold, etc)
    vr1_col1, vr1_col2, vr1_col3, vr1_col4, vr1_col5 = st.columns([1, 2, 3, 2, 2])
    vr2_col1, vr2_col2, vr2_col3, vr2_col4, vr2_col5 = st.columns([1, 2, 3, 2, 2])

    st.divider()

    # Defining columns to display filter results (capture rate, filter rate, etc.)
    with st.container():
        r3_header = st.empty()
        model_1_statistics = st.empty()
    with st.container():
        r4_header = st.empty()
        model_2_statistics = st.empty()
    with st.container():
        classtype_count = st.empty()

    st.divider()

    # Columns for drawing distribution chart and ROC curve
    vr3_col1, vr3_col2 = st.columns(2)

    invalid_input = [output_dir_default, '']

    if rv_m1_output_dir not in invalid_input and rv_m2_output_dir not in invalid_input:
        model_1_metadata, model_1_raw_data = result_viewer.get_multilot_model_data(rv_m1_output_dir)
        model_2_metadata, model_2_raw_data = result_viewer.get_multilot_model_data(rv_m2_output_dir)

        if model_1_metadata is None:
            with r2_col1:
                st.error(f"Error getting result data from {rv_m1_output_dir}")
                return

        if model_2_metadata is None:
            with r2_col1:
                st.error(f"Error getting result data from {rv_m2_output_dir}")
                return

        # Check both have the same list of lot id
        model_1_lot_id_set = {meta["lot_id"] for meta in model_1_metadata}
        model_2_lot_id_set = {meta["lot_id"] for meta in model_2_metadata}
        if model_1_lot_id_set != model_2_lot_id_set:
            with r2_col1:
                st.error(f"""Lot IDs do not match!
                         \nModel 1 lot IDs: {model_1_lot_id_set}
                         \nModel 2 lot IDs: {model_2_lot_id_set}""")
                return
        else:
            logger.info("All lot IDs matched between model 1 and model 2!")

        # Check if results for both models were calculated using the same labels
        for raw_data_1, meta_1, raw_data_2, meta_2 in zip(model_1_raw_data, model_1_metadata, model_2_raw_data, model_2_metadata):
            if raw_data_1[2] != raw_data_2[2]:
                with r2_col1:
                    st.error(f"""Results were not calculated using the same labels. Check if the same lrf file was used.
                            \nModel 1 LRF: {meta_1['input_lrf_path']}
                            \nModel 2 LRF: {meta_2['input_lrf_path']}""")
                    return

        # Show result database details
        with vr1_col1:
            st.text("Model 1 (Base)")
        with vr2_col1:
            st.text("Model 2 (Candidate)")
        with vr1_col2:
            for meta in model_1_metadata:
                st.text(f"Lot ID:\n{meta['lot_id']}")
        with vr2_col2:
            for meta in model_2_metadata:
                st.text(f"Lot ID:\n{meta['lot_id']}")
        with vr1_col3:
            st.text(f"Inference Model:\n{helper.format_model_name(model_1_metadata[0]['model_name'])}")
        with vr2_col3:
            st.text(f"Inference Model:\n{helper.format_model_name(model_2_metadata[0]['model_name'])}")

        # Generate lrf by top k, and adjust threshold according to top k
        rv_m1_threshold = rv_m2_threshold = 1.0 # init as max value
        for index, (meta_1, meta_2) in enumerate(zip(model_1_metadata, model_2_metadata)):
            if st_gen_lrf_type == "top_k":
                with vr1_col4:
                    # Only show one top k number selector
                    if index == 0:
                        st.number_input(label="Top k", min_value=0, max_value=999, value=150, step=1,
                                        help="Top-k defects ranked by Probabilities will be considered as defects.",
                                        key='m1_topk')
                with vr2_col4:
                    if index == 0:
                        st.number_input(label="Top k", min_value=0, max_value=999, value=150, step=1,
                                        help="Top-k defects ranked by Probabilities will be considered as defects.",
                                        key='m2_topk')
                with vr1_col5:
                    result_viewer.gen_lrf(model_id="1",
                                          output_dir=rv_m1_output_dir,
                                          gen_lrf_type=st_gen_lrf_type,
                                          top_k=st.session_state['m1_topk'],
                                          key_number=index, # unique key for each gen lrf button
                                          lot_id=meta_1["lot_id"])

                with vr2_col5:
                    result_viewer.gen_lrf(model_id="2",
                                          output_dir=rv_m2_output_dir,
                                          gen_lrf_type=st_gen_lrf_type,
                                          top_k=st.session_state['m2_topk'],
                                          key_number=index + len(model_1_metadata), # unique key for each gen lrf button
                                          lot_id=meta_2["lot_id"])

                m1_current_threshold = helper.get_topk_model_threshold(output_dir=rv_m1_output_dir,
                                                                       top_k=st.session_state['m1_topk'],
                                                                       lot_id=meta_1["lot_id"])
                if m1_current_threshold < rv_m1_threshold:
                    rv_m1_threshold = m1_current_threshold

                m2_current_threshold = helper.get_topk_model_threshold(output_dir=rv_m2_output_dir,
                                                                       top_k=st.session_state['m2_topk'],
                                                                       lot_id=meta_2["lot_id"])
                if m2_current_threshold < rv_m2_threshold:
                    rv_m2_threshold = m2_current_threshold

            # Generate lrf by threshold
            else:
                with vr1_col4:
                    if index == 0:
                        rv_m1_threshold = st.number_input(label="Confidence threshold:",
                                                          value=meta_1['model_threshold'],
                                                          step=0.00001,
                                                          format="%.5f",
                                                          help="Probabilities above threshold will be considered as defects.",
                                                          key='m1_threshold')
                with vr2_col4:
                    if index == 0:
                        rv_m2_threshold = st.number_input(label="Confidence threshold:",
                                                          value=meta_2['model_threshold'],
                                                          step=0.00001,
                                                          format="%.5f",
                                                          help="Probabilities above threshold will be considered as defects.",
                                                          key='m2_threshold')

                # Validate confidence thresholds
                if rv_m1_threshold < 0.0 or rv_m1_threshold > 1.0:
                    logger.error(f'Confidence threshold must be between 0.0 and 1.0! Model 1 selected confidence threshold: {rv_m1_threshold}')
                    st.error(f'Confidence threshold must be between 0.0 and 1.0! Model 1 selected confidence threshold: {rv_m1_threshold}')
                    return
                elif rv_m2_threshold < 0.0 or rv_m2_threshold > 1.0:
                    logger.error(f'Confidence threshold must be between 0.0 and 1.0! Model 2 selected confidence threshold: {rv_m2_threshold}')
                    st.error(f'Confidence threshold must be between 0.0 and 1.0! Model 2 selected confidence threshold: {rv_m2_threshold}')
                    return

                with vr1_col5:
                    result_viewer.gen_lrf(model_id="1",
                                          output_dir=rv_m1_output_dir,
                                          gen_lrf_type=st_gen_lrf_type,
                                          threshold=rv_m1_threshold,
                                          key_number=index, # unique key for each gen lrf button
                                          lot_id=meta_1["lot_id"])
                with vr2_col5:
                    result_viewer.gen_lrf(model_id="2",
                                          output_dir=rv_m2_output_dir,
                                          gen_lrf_type=st_gen_lrf_type,
                                          threshold=rv_m2_threshold,
                                          key_number=index + len(model_1_metadata),
                                          lot_id=meta_2["lot_id"])

        # Show Total/Defect/Non-defect/unlabeled count
        with r3_header:
            st.text("Model 1 results")

        with model_1_statistics.container():
            defect_id_lists, probability_lists, answer_lists = model_1_raw_data

            for id_list, prob_list, ans_list, meta in zip(defect_id_lists, probability_lists, answer_lists, model_1_metadata):
                st.text(f"{meta['lot_id']}")
                count_rate_data = result_viewer.calculate_filtered_results((id_list, prob_list, ans_list), rv_m1_threshold)
                r3_col1, r3_col2, r3_col3, r3_col4 = st.columns([4, 3, 3, 2])
                with r3_col1:
                    st.warning(f"""**Total defect count**: As-is {count_rate_data['as_is_defect_count']}
                            → To-be: {count_rate_data['to_be_defect_count']}
                            (Filter Rate: {count_rate_data['filter_rate']:.4f})""")
                with r3_col2:
                    st.error(f"""**True defect count**: {count_rate_data['as_is_true_defect_count']}
                            → {count_rate_data['to_be_true_defect_count']}
                            (Capture Rate: {count_rate_data['capture_rate']:.4f})""")
                with r3_col3:
                    st.success(f"""**Non-defect count**: {count_rate_data['as_is_non_defect_count']}
                            → {count_rate_data['to_be_non_defect_count']}
                            (False Filter Rate: {count_rate_data['false_filter_rate']:.4f})""")
                with r3_col4:
                    st.info(f"""**Unlabeled count**: {count_rate_data['unlabeled']}
                            → {count_rate_data['filtered_unlabeled_defect_count']}""")

        with r4_header:
            st.text("Model 2 results")
        with model_2_statistics.container():
            defect_id_lists, probability_lists, answer_lists = model_2_raw_data

            for id_list, prob_list, ans_list, meta in zip(defect_id_lists, probability_lists, answer_lists, model_2_metadata):
                st.text(f"{meta['lot_id']}")
                count_rate_data = result_viewer.calculate_filtered_results((id_list, prob_list, ans_list), rv_m2_threshold)
                r4_col1, r4_col2, r4_col3, r4_col4 = st.columns([4, 3, 3, 2])
                with r4_col1:
                    st.warning(f"""**Total defect count**: As-is: {count_rate_data['as_is_defect_count']}
                            → To-be: {count_rate_data['to_be_defect_count']}
                            (Filter Rate: {count_rate_data['filter_rate']:.4f})""")
                with r4_col2:
                    st.error(f"""**True defect count**: {count_rate_data['as_is_true_defect_count']}
                            → {count_rate_data['to_be_true_defect_count']}
                            (Capture Rate: {count_rate_data['capture_rate']:.4f})""")
                with r4_col3:
                    st.success(f"""**Non-defect count**: {count_rate_data['as_is_non_defect_count']}
                            → {count_rate_data['to_be_non_defect_count']}
                            (False Filter Rate: {count_rate_data['false_filter_rate']:.4f})""")
                with r4_col4:
                    st.info(f"""**Unlabeled count**: {count_rate_data['unlabeled']}
                            → {count_rate_data['filtered_unlabeled_defect_count']}""")

        # TODO: Get classtype grouping from backend
        with classtype_count:
            with st.expander(label="LRF ClassType count"):
                defect_lists_1 = helper.get_lrf_data(output_dir=rv_m1_output_dir,
                                                   cols=["ClassType"],
                                                   include_prob=False)
                for defect_list, meta in zip(defect_lists_1, model_1_metadata):
                    classtype_counter_df = result_viewer.get_classtype_count(defect_list)
                    st.text(f"Lot ID: {meta['lot_id']}")
                    st.caption(f"LRF type: {meta['input_lrf_type']}")
                    st.dataframe(data=classtype_counter_df)
                    st.divider()

        # TODO: Wrap into a function
        # Aggregate raw data lists
        aggregate_id_list_1, aggregate_prob_list_1, aggregate_ans_list_1, aggregate_lot_id_list_1 =  [], [], [], []
        m1_defect_id_lists, m1_prob_lists, m1_ans_lists = model_1_raw_data
        for defect_id_list, prob_list, ans_list, meta in zip(m1_defect_id_lists, m1_prob_lists, m1_ans_lists, model_1_metadata):
            aggregate_id_list_1.extend(defect_id_list)
            aggregate_prob_list_1.extend(prob_list)
            aggregate_ans_list_1.extend(ans_list)
            aggregate_lot_id_list_1.extend([meta['lot_id']] * len(defect_id_list))
        aggregate_id_list_2, aggregate_prob_list_2, aggregate_ans_list_2, aggregate_lot_id_list_2 =  [], [], [], []
        m2_defect_id_lists, m2_prob_lists, m2_ans_lists = model_2_raw_data
        for defect_id_list, prob_list, ans_list, meta in zip(m2_defect_id_lists, m2_prob_lists, m2_ans_lists, model_2_metadata):
            aggregate_id_list_2.extend(defect_id_list)
            aggregate_prob_list_2.extend(prob_list)
            aggregate_ans_list_2.extend(ans_list)
            aggregate_lot_id_list_2.extend([meta['lot_id']] * len(defect_id_list))

        # Draw 2D comparison chart
        with vr3_col1:
            st.plotly_chart(result_viewer.generate_2D_plot((aggregate_id_list_1, aggregate_prob_list_1, aggregate_ans_list_1, rv_m1_threshold, aggregate_lot_id_list_1),
                                                           (aggregate_id_list_2, aggregate_prob_list_2, aggregate_ans_list_2, rv_m2_threshold, aggregate_lot_id_list_2)))

        with vr3_col2:
            # TODO: This should be done somewhere else
            if 1 not in set(aggregate_ans_list_1) or 1 not in set(aggregate_ans_list_2):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")

            else:
                model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, return_curve=True)
                model_2_roc_data = helper.get_roc_data(rv_m2_output_dir, return_curve=True)
                st.plotly_chart(result_viewer.plot_multilot_roc([
                    ("Model 1", model_1_roc_data, rv_m1_threshold, model_1_metadata),
                    ("Model 2", model_2_roc_data, rv_m2_threshold, model_2_metadata),
                ]))

    elif rv_m1_output_dir not in invalid_input:
        model_1_metadata, model_1_raw_data = result_viewer.get_multilot_model_data(rv_m1_output_dir)

        if model_1_metadata is None:
            with r2_col1:
                st.error(f"Error getting result data from {rv_m1_output_dir}")
                return

        # Show result database details
        with vr1_col1:
            st.text("Model 1 (Base)")
        with vr1_col2:
            for meta in model_1_metadata:
                st.text(f"Lot ID: {meta['lot_id']}")
        with vr1_col3:
            st.text(f"Inference Model:\n{helper.format_model_name(model_1_metadata[0]['model_name'])}")

        # Generate lrf by top k, and adjust threshold according to top k
        rv_m1_threshold = 1.0 # init as max value
        for index, meta in enumerate(model_1_metadata):
            if st_gen_lrf_type == "top_k":
                with vr1_col4:
                    # Only show one top k number selector
                    if index == 0:
                        st.number_input(label="Top k", min_value=0, max_value=999, value=150, step=1,
                                        help="Top-k defects ranked by Probabilities will be considered as defects.",
                                        key='m1_topk')
                with vr1_col5:
                    result_viewer.gen_lrf(model_id="1",
                                          output_dir=rv_m1_output_dir,
                                          gen_lrf_type=st_gen_lrf_type,
                                          top_k=st.session_state['m1_topk'],
                                          key_number=index, # unique key for each gen lrf button
                                          lot_id=meta["lot_id"])

                current_threshold = helper.get_topk_model_threshold(output_dir=rv_m1_output_dir,
                                                                          top_k=st.session_state['m1_topk'],
                                                                          lot_id=meta["lot_id"])

                # Select lowest calculated threshold to draw dotted line when in top_k mode
                if current_threshold < rv_m1_threshold:
                    rv_m1_threshold = current_threshold

            # Generate lrf by threshold
            else:
                with vr1_col4:
                    if index == 0:
                        rv_m1_threshold = st.number_input(label="Confidence threshold:",
                                                          value=meta['model_threshold'],
                                                          step=0.00001,
                                                          format="%.5f",
                                                          help="Probabilities above threshold will be considered as defects.",
                                                          key=f'm1_threshold_{index}')

                # Validate confidence threshold
                if rv_m1_threshold < 0.0 or rv_m1_threshold > 1.0:
                    logger.error(f'Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {rv_m1_threshold}')
                    st.error(f'Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {rv_m1_threshold}')
                    return

                with vr1_col5:
                    result_viewer.gen_lrf(model_id="1",
                                          output_dir=rv_m1_output_dir,
                                          gen_lrf_type=st_gen_lrf_type,
                                          threshold=rv_m1_threshold,
                                          key_number=index, # unique key for each gen lrf button
                                          lot_id=meta["lot_id"])

        # Show Total/Defect/Non-defect/unlabeled count
        with r3_header.container():
            st.text("Model 1 results")

        with model_1_statistics.container():
            defect_id_lists, probability_lists, answer_lists = model_1_raw_data

            for id_list, prob_list, ans_list, meta in zip(defect_id_lists, probability_lists, answer_lists, model_1_metadata):
                st.text(f"{meta['lot_id']}")
                count_rate_data = result_viewer.calculate_filtered_results((id_list, prob_list, ans_list), rv_m1_threshold)

                r3_col1, r3_col2, r3_col3, r3_col4 = st.columns([4, 3, 3, 2])
                with r3_col1:
                    st.warning(f"""**Total defect count**: As-is: {count_rate_data['as_is_defect_count']}
                            → To-be: {count_rate_data['to_be_defect_count']}
                            (Filter Rate: {count_rate_data['filter_rate']:.4f})""")
                with r3_col2:
                    st.error(f"""**True defect count**: {count_rate_data['as_is_true_defect_count']}
                            → {count_rate_data['to_be_true_defect_count']}
                            (Capture Rate: {count_rate_data['capture_rate']:.4f})""")
                with r3_col3:
                    st.success(f"""**Non-defect count**: {count_rate_data['as_is_non_defect_count']}
                            → {count_rate_data['to_be_non_defect_count']}
                            (False Filter Rate: {count_rate_data['false_filter_rate']:.4f})""")
                with r3_col4:
                    st.info(f"""**Unlabeled count**: {count_rate_data['unlabeled']}
                            → {count_rate_data['filtered_unlabeled_defect_count']}""")

        # TODO: Get classtype grouping from backend
        with classtype_count:
            with st.expander(label="LRF ClassType count"):
                defect_lists = helper.get_lrf_data(output_dir=rv_m1_output_dir,
                                                   cols=["ClassType"],
                                                   include_prob=False)
                for defect_list, meta in zip(defect_lists, model_1_metadata):
                    classtype_counter_df = result_viewer.get_classtype_count(defect_list)
                    st.text(f"Lot ID: {meta['lot_id']}")
                    st.caption(f"LRF type: {meta['input_lrf_type']}")
                    st.dataframe(data=classtype_counter_df)
                    st.divider()

        # Aggregate raw data lists
        aggregate_id_list, aggregate_prob_list, aggregate_ans_list, aggregate_lot_id_list =  [], [], [], []
        m1_defect_id_lists, m1_prob_lists, m1_ans_lists = model_1_raw_data
        for defect_id_list, prob_list, ans_list, meta in zip(m1_defect_id_lists, m1_prob_lists, m1_ans_lists, model_1_metadata):
            aggregate_id_list.extend(defect_id_list)
            aggregate_prob_list.extend(prob_list)
            aggregate_ans_list.extend(ans_list)
            # aggregate_threshold_list.extend([rv_m1_threshold] * len(defect_id_list))
            aggregate_lot_id_list.extend([meta['lot_id']] * len(defect_id_list))

        # Draw 1D comparison chart
        with vr3_col1:
            st.plotly_chart(result_viewer.generate_multilot_1D_plot(aggregate_id_list,
                                                                    aggregate_prob_list,
                                                                    aggregate_ans_list,
                                                                    rv_m1_threshold,
                                                                    aggregate_lot_id_list))

        with vr3_col2:
            # TODO: This should be done somewhere else
            if 1 not in set(aggregate_ans_list):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")

            else:
                model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, return_curve=True)
                st.plotly_chart(result_viewer.plot_multilot_roc([("Model 1", model_1_roc_data, rv_m1_threshold, model_1_metadata)]))

    else:
        pass
