import streamlit as st
import pandas as pd
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper
from ltt_ff_frontend.result_viewer import result_viewer

def highlight_cols(s):
    colors = {
        "red": "background-color: #ffcccb",
        "yellow": "background-color: #ffeb3b",
        "green": "background-color: #d4edda",
        "blue": "background-color: #d1ecf1",
        "default": "background-color: #d3d3d3"
    }
    col_to_colors = {
        'Total Defect Count': colors['yellow'],
        'True Defect Count': colors['red'],
        'Non Defect Count': colors['green'],
        'Unlabeled Count': colors['blue'],
    }
    return [col_to_colors.get(first_index, colors['default']) for first_index in s.index.get_level_values(0)]

def highlight_capture_rate(column):
    styles = []
    for val in column:
        numeric_val = float(val)
        color = 'red' if numeric_val != 1.0 and numeric_val != -1.0 else ''
        font_weight = 'bold' if numeric_val != 1.0 and numeric_val != -1.0 else ''
        styles.append(f'color: {color}; font-weight: {font_weight};')
    return styles

def highlight_to_be_total_defect_count(row):
    numeric_total_to_be_count = float(row[('Total Defect Count', 'To-be')])
    numeric_true_defect_count = float(row[('True Defect Count', 'Before')])

    row_styles = [''] * len(row)
    if numeric_total_to_be_count > 150 and numeric_true_defect_count <= 150:
        index = row.index.get_loc(('Total Defect Count', 'To-be'))
        row_styles[index] = "color: red; font-weight: bold;"
    else:
        pass
    return row_styles
def app(output_dir_default: str, rv_m1_output_dir: str, rv_m2_output_dir: str, st_gen_lrf_type: str) -> None:
    # Column for printing error message
    r2_col1, _r2_col2 = st.columns([3, 2])

    # Columns for printing Model info for 1 or 2 models (Model #1/2, model name, threshold, lot ID + gen lrf button)
    vr1_col1, vr1_col2, vr1_col3, vr1_col4 = st.columns([1, 3, 3, 4])
    with st.container():
        model_info_divider = st.empty()
    vr2_col1, vr2_col2, vr2_col3, vr2_col4 = st.columns([1, 3, 3, 4])

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
        for raw_data_1, meta_1, raw_data_2, meta_2 in zip(sorted(model_1_raw_data[2]), model_1_metadata, sorted(model_2_raw_data[2]), model_2_metadata):
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
            st.text(f"Inference Model:\n{helper.format_model_name(model_1_metadata[0]['model_name'])}")
        with vr2_col2:
            st.text(f"Inference Model:\n{helper.format_model_name(model_2_metadata[0]['model_name'])}")
        with model_info_divider:
            st.divider()

        # Generate lrf by top k, and adjust threshold according to top k
        rv_m1_threshold = rv_m2_threshold = 1.0 # init as max value
        for index, (meta_1, meta_2) in enumerate(zip(model_1_metadata, model_2_metadata)):
            if st_gen_lrf_type == "top_k":
                with vr1_col3:
                    # Only show one top k number selector
                    if index == 0:
                        st.number_input(label="Top k", min_value=0, max_value=999, value=150, step=1,
                                        help="Top-k defects ranked by Probabilities will be considered as defects.",
                                        key='m1_topk')
                with vr2_col3:
                    if index == 0:
                        st.number_input(label="Top k", min_value=0, max_value=999, value=150, step=1,
                                        help="Top-k defects ranked by Probabilities will be considered as defects.",
                                        key='m2_topk')
                with vr1_col4:
                    gen_lrf_col, lot_id_col = st.columns([2, 3])
                    with gen_lrf_col:
                        result_viewer.gen_lrf(model_id="1",
                                            output_dir=rv_m1_output_dir,
                                            gen_lrf_type=st_gen_lrf_type,
                                            top_k=st.session_state['m1_topk'],
                                            key_number=index, # unique key for each gen lrf button
                                            lot_id=meta_1["lot_id"])
                    with lot_id_col:
                        st.text(f"Lot ID: {meta_1['lot_id']}")

                with vr2_col4:
                    gen_lrf_col, lot_id_col = st.columns([2, 3])
                    with gen_lrf_col:
                        result_viewer.gen_lrf(model_id="2",
                                          output_dir=rv_m2_output_dir,
                                          gen_lrf_type=st_gen_lrf_type,
                                          top_k=st.session_state['m2_topk'],
                                          key_number=index + len(model_1_metadata), # unique key for each gen lrf button
                                          lot_id=meta_2["lot_id"])
                    with lot_id_col:
                        st.text(f"Lot ID: {meta_2['lot_id']}")

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
                with vr1_col3:
                    if index == 0:
                        rv_m1_threshold = st.number_input(label="Confidence threshold:",
                                                          value=meta_1['model_threshold'],
                                                          step=0.00001,
                                                          format="%.5f",
                                                          help="Probabilities above threshold will be considered as defects.",
                                                          key='m1_threshold')
                with vr2_col3:
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

                with vr1_col4:
                    gen_lrf_col, lot_id_col = st.columns([2, 3])
                    with gen_lrf_col:
                        result_viewer.gen_lrf(model_id="1",
                                            output_dir=rv_m1_output_dir,
                                            gen_lrf_type=st_gen_lrf_type,
                                            threshold=rv_m1_threshold,
                                            key_number=index, # unique key for each gen lrf button
                                            lot_id=meta_1["lot_id"])
                    with lot_id_col:
                        st.text(f"Lot ID: {meta_1['lot_id']}")

                with vr2_col4:
                    gen_lrf_col, lot_id_col = st.columns([2, 3])
                    with gen_lrf_col:
                        result_viewer.gen_lrf(model_id="2",
                                            output_dir=rv_m2_output_dir,
                                            gen_lrf_type=st_gen_lrf_type,
                                            threshold=rv_m2_threshold,
                                            key_number=index + len(model_1_metadata),
                                            lot_id=meta_2["lot_id"])
                    with lot_id_col:
                        st.text(f"Lot ID: {meta_2['lot_id']}")

        # Show Total/Defect/Non-defect/unlabeled count
        with r3_header:
            st.subheader("Model 1 results")

        with model_1_statistics.container():
            model_1_selected_lot_id_list = result_viewer.show_multilot_statistics(model_1_raw_data, model_1_metadata, rv_m1_threshold)

        with r4_header:
            st.subheader("Model 2 results")
        with model_2_statistics.container():
            result_viewer.show_multilot_statistics(model_2_raw_data, model_2_metadata, rv_m2_threshold)

        # TODO: Get classtype grouping from backend
        with classtype_count:
            with st.expander(label="LRF ClassType count"):
                defect_lists = helper.get_lrf_data_lists(output_dir=rv_m1_output_dir,
                                                   cols=["ClassType"],
                                                   include_prob=False)
                for defect_list, meta in zip(defect_lists, model_1_metadata):
                    classtype_counter_df = result_viewer.get_classtype_count(defect_list)
                    st.text(f"Lot ID: {meta['lot_id']}")
                    st.caption(f"LRF type: {meta['input_lrf_type']}")
                    st.dataframe(data=classtype_counter_df)
                    st.divider()

        # Combine defect_ids, probabilities, answers, lot_id into one list each
        m1_aggregated_data_lists = result_viewer.aggregate_lists(model_1_raw_data, model_1_metadata)
        m2_aggregated_data_lists = result_viewer.aggregate_lists(model_2_raw_data, model_2_metadata)

        # Draw 2D comparison chart
        with vr3_col1:
            st.plotly_chart(result_viewer.generate_2D_plot(m1_aggregated_data_lists,
                                                           m2_aggregated_data_lists,
                                                           rv_m1_threshold,
                                                           rv_m2_threshold))

        with vr3_col2:
            # TODO: This should be done somewhere else
            if 1 not in set(m1_aggregated_data_lists[2]) or 1 not in set(m1_aggregated_data_lists[2]):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")

            else:
                model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, return_curve=True)
                model_2_roc_data = helper.get_roc_data(rv_m2_output_dir, return_curve=True)
                st.plotly_chart(result_viewer.plot_multilot_roc([
                    ("Model 1", model_1_roc_data, rv_m1_threshold, model_1_metadata, rv_m1_output_dir),
                    ("Model 2", model_2_roc_data, rv_m2_threshold, model_2_metadata, rv_m2_output_dir),
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
            st.text(f"Inference Model:\n{helper.format_model_name(model_1_metadata[0]['model_name'])}")

        # Generate lrf by top k, and adjust threshold according to top k
        rv_m1_threshold = 1.0 # init as max value
        for index, meta in enumerate(model_1_metadata):
            if st_gen_lrf_type == "top_k":
                with vr1_col3:
                    # Only show one top k number selector
                    if index == 0:
                        st.number_input(label="Top k", min_value=0, max_value=999, value=150, step=1,
                                        help="Top-k defects ranked by Probabilities will be considered as defects.",
                                        key='m1_topk')
                with vr1_col4:
                    gen_lrf_col, lot_id_col = st.columns([2, 3])
                    with gen_lrf_col:
                        result_viewer.gen_lrf(model_id="1",
                                            output_dir=rv_m1_output_dir,
                                            gen_lrf_type=st_gen_lrf_type,
                                            top_k=st.session_state['m1_topk'],
                                            key_number=index, # unique key for each gen lrf button
                                            lot_id=meta["lot_id"])
                    with lot_id_col:
                        st.text(f"Lot ID: {meta['lot_id']}")

                current_threshold = helper.get_topk_model_threshold(output_dir=rv_m1_output_dir,
                                                                          top_k=st.session_state['m1_topk'],
                                                                          lot_id=meta["lot_id"])

                # Select lowest calculated threshold to draw dotted line when in top_k mode
                if current_threshold < rv_m1_threshold:
                    rv_m1_threshold = current_threshold

            # Generate lrf by threshold
            else:
                with vr1_col3:
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

                with vr1_col4:
                    gen_lrf_col, lot_id_col = st.columns([2, 3])
                    with gen_lrf_col:
                        result_viewer.gen_lrf(model_id="1",
                                            output_dir=rv_m1_output_dir,
                                            gen_lrf_type=st_gen_lrf_type,
                                            threshold=rv_m1_threshold,
                                            key_number=index, # unique key for each gen lrf button
                                            lot_id=meta["lot_id"])

                    with lot_id_col:
                        st.text(f"Lot ID: {meta['lot_id']}")

        # Show Total/Defect/Non-defect/unlabeled count
        with r3_header.container():
            st.subheader("Model 1 results")

        with model_1_statistics.container():
            model_1_selected_lot_id_list = result_viewer.show_multilot_statistics(model_1_raw_data, model_1_metadata, rv_m1_threshold)

        # TODO: Get classtype grouping from backend
        with classtype_count:
            with st.expander(label="LRF ClassType count"):
                defect_lists = helper.get_lrf_data_lists(output_dir=rv_m1_output_dir,
                                                   cols=["ClassType"],
                                                   include_prob=False)
                for defect_list, meta in zip(defect_lists, model_1_metadata):
                    classtype_counter_df = result_viewer.get_classtype_count(defect_list)
                    st.text(f"Lot ID: {meta['lot_id']}")
                    st.caption(f"LRF type: {meta['input_lrf_type']}")
                    st.dataframe(data=classtype_counter_df)
                    st.divider()

        # Combine defect_ids, probabilities, answers, lot_id into one list each
        m1_aggregated_data_lists = result_viewer.aggregate_lists(model_1_raw_data, model_1_metadata)

        # Draw 1D comparison chart
        with vr3_col1:
            st.plotly_chart(
                result_viewer.generate_multilot_1D_plot(
                    m1_aggregated_data_lists,
                    rv_m1_threshold,
                    model_1_selected_lot_id_list))

        with vr3_col2:
            # TODO: This should be done somewhere else
            if 1 not in set(m1_aggregated_data_lists[2]):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")

            else:
                model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, return_curve=True)
                params = [("Model 1", model_1_roc_data, rv_m1_threshold, model_1_metadata, rv_m1_output_dir)]
                st.plotly_chart(result_viewer.plot_multilot_roc(params, model_1_selected_lot_id_list))

    else:
        pass
