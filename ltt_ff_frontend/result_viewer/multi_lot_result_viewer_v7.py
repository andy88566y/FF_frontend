import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper
from ltt_ff_frontend.result_viewer import result_viewer

def highlight_cols(s):
    colors = {
        "red": "background-color: #ffcccb",
        "yellow": "background-color: #ffeb3b",
        "green": "background-color: #d4edda",
        "blue": "background-color: #d1ecf1",
        "default": "background-color: #d3d3d3",
    }
    col_to_colors = {
        "Total Defect Count": colors["yellow"],
        "True Defect Count": colors["red"],
        "Non Defect Count": colors["green"],
        "Unlabeled Count": colors["blue"],
    }
    return [col_to_colors.get(first_index, colors["default"]) for first_index in s.index.get_level_values(0)]

def highlight_capture_rate(column):
    styles = []
    for val in column:
        numeric_val = float(val)
        color = "red" if numeric_val != 1.0 and numeric_val != -1.0 else ""
        font_weight = "bold" if numeric_val != 1.0 and numeric_val != -1.0 else ""
        styles.append(f"color: {color}; font-weight: {font_weight};")
    return styles

def highlight_to_be_total_defect_count(row):
    numeric_total_to_be_count = float(row[("Total Defect Count", "To-be")])
    numeric_true_defect_count = float(row[("True Defect Count", "Before")])

    row_styles = [""] * len(row)
    if numeric_total_to_be_count > 150 and numeric_true_defect_count <= 150:
        index = row.index.get_loc(("Total Defect Count", "To-be"))
        row_styles[index] = "color: red; font-weight: bold;"
    else:
        pass
    return row_styles

def app(
    output_dir_default: str,
    output_dir: str,
    st_gen_lrf_type: str
) -> None:
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

    invalid_input = [output_dir_default, ""]

    if output_dir not in invalid_input:
        model_1_metadata, model_1_raw_data = result_viewer.get_multilot_model_data(output_dir)

        if model_1_metadata is None:
            with r2_col1:
                st.error(f"Error getting result data from {output_dir}")
                return

        model_1_name = model_1_metadata[0].get("model_name", model_1_metadata[0].get("model_name_0", ""))
        model_1_threshold = model_1_metadata[0].get("model_threshold", model_1_metadata[0].get("model_threshold_0", ""))

        # Show result database details
        with vr1_col1:
            st.text("Model 1 (Base)")
        with vr1_col2:
            st.text(f"Inference Model:\n{helper.format_model_name(model_1_name)}")

        # Generate lrf by top k, and adjust threshold according to top k
        rv_m1_threshold = 1.0  # init as max value
        for index, meta in enumerate(model_1_metadata):
            if st_gen_lrf_type == "top_k":
                with vr1_col3:
                    # Only show one top k number selector
                    if index == 0:
                        st.number_input(
                            label="Top k",
                            min_value=0,
                            max_value=999,
                            value=150,
                            step=1,
                            help="Top-k defects ranked by Probabilities will be considered as defects.",
                            key="m1_topk",
                        )
                with vr1_col4:
                    gen_lrf_col, lot_id_col = st.columns([2, 3])
                    with gen_lrf_col:
                        result_viewer.gen_lrf(
                            model_id="1",
                            output_dir=output_dir,
                            gen_lrf_type=st_gen_lrf_type,
                            top_k=st.session_state["m1_topk"],
                            key_number=index,  # unique key for each gen lrf button
                            lot_id=meta["lot_id"],
                        )
                    with lot_id_col:
                        st.text(f"Lot ID: {meta['lot_id']}")

                current_threshold = helper.get_topk_model_threshold(
                    output_dir=output_dir, top_k=st.session_state["m1_topk"], lot_id=meta["lot_id"]
                )

                # Select lowest calculated threshold to draw dotted line when in top_k mode
                if current_threshold < rv_m1_threshold:
                    rv_m1_threshold = current_threshold

            # Generate lrf by threshold
            else:
                with vr1_col3:
                    if index == 0:
                        rv_m1_threshold = st.number_input(
                            label="Confidence threshold:",
                            value=model_1_threshold,
                            step=0.00001,
                            format="%.5f",
                            help="Probabilities above threshold will be considered as defects.",
                            key=f"m1_threshold_{index}",
                        )

                # Validate confidence threshold
                if rv_m1_threshold < 0.0 or rv_m1_threshold > 1.0:
                    logger.error(
                        f"Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {rv_m1_threshold}"
                    )
                    st.error(
                        f"Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {rv_m1_threshold}"
                    )
                    return

                with vr1_col4:
                    gen_lrf_col, lot_id_col = st.columns([2, 3])
                    with gen_lrf_col:
                        result_viewer.gen_lrf(
                            model_id="1",
                            output_dir=output_dir,
                            gen_lrf_type=st_gen_lrf_type,
                            threshold=rv_m1_threshold,
                            key_number=index,  # unique key for each gen lrf button
                            lot_id=meta["lot_id"],
                        )

                    with lot_id_col:
                        st.text(f"Lot ID: {meta['lot_id']}")

        # Show Total/Defect/Non-defect/unlabeled count
        with r3_header.container():
            st.subheader("Model 1 results")

        with model_1_statistics.container():
            model_1_selected_lot_id_list = result_viewer.show_multilot_statistics(
                model_1_raw_data, model_1_metadata, rv_m1_threshold, "model_1_stats"
            )

        # TODO: Get classtype grouping from backend
        with classtype_count:
            with st.expander(label="LRF ClassType count"):
                defect_lists = helper.get_lrf_data_lists(
                    output_dir=output_dir, cols=["ClassType"], include_prob=False
                )
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
                    m1_aggregated_data_lists, rv_m1_threshold, model_1_selected_lot_id_list
                )
            )

        with vr3_col2:
            # TODO: This should be done somewhere else
            if 1 not in set(m1_aggregated_data_lists[2]):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")

            else:
                model_1_roc_data = helper.get_roc_data(output_dir, return_curve=True)
                params = [("Model 1", model_1_roc_data, rv_m1_threshold, model_1_metadata, output_dir)]
                st.plotly_chart(result_viewer.plot_multilot_roc(params, model_1_selected_lot_id_list))

    else:
        st.caption('Inference Result Directory is invalid.')
        pass
