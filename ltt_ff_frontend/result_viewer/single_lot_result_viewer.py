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
        r3_col1, r3_col2, r3_col3, r3_col4 = st.columns([4, 3, 3, 2])
    with st.container():
        r4_header = st.empty()
        r4_col1, r4_col2, r4_col3, r4_col4 = st.columns([4, 3, 3, 2])
    with st.container():
        classtype_count = st.empty()

    st.divider()

    # Columns for drawing distribution chart and ROC curve
    vr3_col1, vr3_col2 = st.columns(2)

    invalid_input = [output_dir_default, ""]

    if rv_m1_output_dir not in invalid_input and rv_m2_output_dir not in invalid_input:
        model_1_metadata, model_1_raw_data = result_viewer.get_model_data(rv_m1_output_dir)
        model_2_metadata, model_2_raw_data = result_viewer.get_model_data(rv_m2_output_dir)

        if model_1_metadata is None:
            with r2_col1:
                st.error(f"Error getting result data from {rv_m1_output_dir}")
                return

        if model_2_metadata is None:
            with r2_col1:
                st.error(f"Error getting result data from {rv_m2_output_dir}")
                return

        if model_1_metadata["lot_id"] != model_2_metadata["lot_id"]:
            with r2_col1:
                st.error(f"""Lot IDs do not match!
                         \nModel 1 lot ID: {model_1_metadata['lot_id']}
                         \nModel 2 lot ID: {model_2_metadata['lot_id']}""")
                return

        # Check if results for both models were calculated using the same labels
        if model_1_raw_data[2] != model_2_raw_data[2]:
            with r2_col1:
                st.error(f"""Results were not calculated using the same labels. Check if the same lrf file was used.
                         \nModel 1 LRF: {model_1_metadata['input_lrf_path']}
                         \nModel 2 LRF: {model_2_metadata['input_lrf_path']}""")
                return

        # Check if the same lrf was used for inference
        if model_1_metadata["input_lrf_path"] != model_2_metadata["input_lrf_path"]:
            with r2_col1:
                st.error(f"""Different LRF files were used during inference!
                         \nModel 1 LRF: {model_1_metadata['input_lrf_path']}
                         \nModel 2 LRF: {model_2_metadata['input_lrf_path']}""")
                return

        model_1_name = model_1_metadata.get("model_name", model_1_metadata.get("model_name_0", ""))
        model_1_threshold = model_1_metadata.get("model_threshold", model_1_metadata.get("model_threshold_0", ""))
        model_2_name = model_2_metadata.get("model_name", model_2_metadata.get("model_name_0", ""))
        model_2_threshold = model_2_metadata.get("model_threshold", model_2_metadata.get("model_threshold_0", ""))

        # Show result database details
        with vr1_col1:
            st.text("Model 1 (Base)")
        with vr2_col1:
            st.text("Model 2 (Candidate)")
        with vr1_col2:
            st.text(f"Lot ID:\n{model_1_metadata['lot_id']}")
        with vr2_col2:
            st.text(f"Lot ID:\n{model_2_metadata['lot_id']}")
        with vr1_col3:
            st.text(f"Inference Model:\n{helper.format_model_name(model_1_name)}")
        with vr2_col3:
            st.text(f"Inference Model:\n{helper.format_model_name(model_2_name)}")

        # Generate lrf by top k, and adjust threshold according to top k
        if st_gen_lrf_type == "top_k":
            with vr1_col4:
                rv_m1_topk = st.number_input(
                    "Top k",
                    0,
                    999,
                    150,
                    1,
                    help="Top-k defects ranked by Probabilities will be considered as defects.",
                    key="m1_topk",
                )
            with vr2_col4:
                rv_m2_topk = st.number_input(
                    "Top k",
                    0,
                    999,
                    150,
                    1,
                    help="Top-k defects ranked by Probabilities will be considered as defects.",
                    key="m2_topk",
                )
            with vr1_col5:
                result_viewer.gen_lrf(
                    model_id="1",
                    output_dir=rv_m1_output_dir,
                    gen_lrf_type=st_gen_lrf_type,
                    top_k=rv_m1_topk,
                    key_number=1,
                )
            with vr2_col5:
                result_viewer.gen_lrf(
                    model_id="2",
                    output_dir=rv_m2_output_dir,
                    gen_lrf_type=st_gen_lrf_type,
                    top_k=rv_m2_topk,
                    key_number=2,
                )

            rv_m1_threshold = helper.get_topk_model_threshold(rv_m1_output_dir, rv_m1_topk)
            rv_m2_threshold = helper.get_topk_model_threshold(rv_m2_output_dir, rv_m2_topk)

        # Generate lrf by threshold
        else:
            with vr1_col4:
                rv_m1_threshold = st.number_input(
                    label="Confidence threshold:",
                    value=model_1_threshold,
                    step=0.00001,
                    format="%.5f",
                    help="Probabilities above thershold will be considered as defects.",
                    key="m1_threshold",
                )
            with vr2_col4:
                rv_m2_threshold = st.number_input(
                    label="Confidence threshold:",
                    value=model_2_threshold,
                    step=0.00001,
                    format="%.5f",
                    help="Probabilities above thershold will be considered as defects.",
                    key="m2_threshold",
                )

            # Validate confidence thresholds
            if rv_m1_threshold < 0.0 or rv_m1_threshold > 1.0:
                logger.error(
                    f"Confidence threshold must be between 0.0 and 1.0! Model 1 selected confidence threshold: {rv_m1_threshold}"
                )
                st.error(
                    f"Confidence threshold must be between 0.0 and 1.0! Model 1 selected confidence threshold: {rv_m1_threshold}"
                )
                return
            elif rv_m2_threshold < 0.0 or rv_m2_threshold > 1.0:
                logger.error(
                    f"Confidence threshold must be between 0.0 and 1.0! Model 2 selected confidence threshold: {rv_m2_threshold}"
                )
                st.error(
                    f"Confidence threshold must be between 0.0 and 1.0! Model 2 selected confidence threshold: {rv_m2_threshold}"
                )
                return

            with vr1_col5:
                result_viewer.gen_lrf(
                    model_id="1",
                    output_dir=rv_m1_output_dir,
                    gen_lrf_type=st_gen_lrf_type,
                    threshold=rv_m1_threshold,
                    key_number=0,
                    lot_id=model_1_metadata["lot_id"],
                )
            with vr2_col5:
                result_viewer.gen_lrf(
                    model_id="2",
                    output_dir=rv_m2_output_dir,
                    gen_lrf_type=st_gen_lrf_type,
                    threshold=rv_m2_threshold,
                    key_number=1,
                    lot_id=model_2_metadata["lot_id"],
                )

        # Show Total/Defect/Non-defect/unlabeled count
        with r3_header:
            st.text("Model 1 results")
        count_rate_data = result_viewer.calculate_filtered_results(model_1_raw_data, rv_m1_threshold)
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
        count_rate_data = result_viewer.calculate_filtered_results(model_2_raw_data, rv_m2_threshold)
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
                defects = helper.get_lrf_data_lists(
                    output_dir=rv_m1_output_dir, cols=["ClassType"], include_prob=False
                )[0]
                classtype_counter_df = result_viewer.get_classtype_count(defects)
                st.caption(f"LRF type: {model_1_metadata['input_lrf_type']}")
                st.dataframe(data=classtype_counter_df)

        # Draw 2D comparison chart
        with vr3_col1:
            defect_id_list_1, prob_list_1, ans_list_1 = model_1_raw_data
            defect_id_list_2, prob_list_2, ans_list_2 = model_2_raw_data
            st.plotly_chart(
                result_viewer.generate_2D_plot(
                    (defect_id_list_1, prob_list_1, ans_list_1, [""] * len(defect_id_list_1)),
                    (defect_id_list_2, prob_list_2, ans_list_2, [""] * len(defect_id_list_2)),
                    rv_m1_threshold,
                    rv_m2_threshold,
                )
            )

        with vr3_col2:
            # TODO: This should be done somewhere else
            if 1 not in set(model_1_raw_data[2]) or 1 not in set(model_2_raw_data[2]):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")

            else:
                model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, return_curve=True)[0]
                model_2_roc_data = helper.get_roc_data(rv_m2_output_dir, return_curve=True)[0]
                st.plotly_chart(
                    result_viewer.plot_roc(
                        [
                            (
                                "Model 1",
                                model_1_roc_data,
                                rv_m1_threshold,
                                model_1_threshold,
                                rv_m1_output_dir,
                            ),
                            (
                                "Model 2",
                                model_2_roc_data,
                                rv_m2_threshold,
                                model_2_threshold,
                                rv_m2_output_dir,
                            ),
                        ]
                    )
                )

    elif rv_m1_output_dir not in invalid_input:
        model_1_metadata, model_1_raw_data = result_viewer.get_model_data(rv_m1_output_dir)

        if model_1_metadata is None:
            with r2_col1:
                st.error(f"Error getting result data from {rv_m1_output_dir}")
                return

        model_1_name = model_1_metadata.get("model_name", model_1_metadata.get("model_name_0", ""))
        model_1_threshold = model_1_metadata.get("model_threshold", model_1_metadata.get("model_threshold_0", ""))

        # Show result database details
        with vr1_col1:
            st.text("Model 1 (Base)")
        with vr1_col2:
            st.text(f"Lot ID:\n{model_1_metadata['lot_id']}")
        with vr1_col3:
            st.text(f"Inference Model:\n{helper.format_model_name(model_1_name)}")

        # Generate lrf by top k, and adjust threshold according to top k
        if st_gen_lrf_type == "top_k":
            with vr1_col4:
                rv_m1_topk = st.number_input(
                    "Top k",
                    0,
                    999,
                    150,
                    1,
                    help="Top-k defects ranked by Probabilities will be considered as defects.",
                    key="m1_topk",
                )
            with vr1_col5:
                result_viewer.gen_lrf("1", rv_m1_output_dir, st_gen_lrf_type, top_k=rv_m1_topk)

            rv_m1_threshold = helper.get_topk_model_threshold(rv_m1_output_dir, rv_m1_topk)

        # Generate lrf by threshold
        else:
            with vr1_col4:
                rv_m1_threshold = st.number_input(
                    label="Confidence threshold:",
                    value=model_1_threshold,
                    step=0.00001,
                    format="%.5f",
                    help="Probabilities above threshold will be considered as defects.",
                    key="m1_threshold",
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

            with vr1_col5:
                result_viewer.gen_lrf("1", rv_m1_output_dir, st_gen_lrf_type, threshold=rv_m1_threshold)

        # Show Total/Defect/Non-defect/unlabeled count
        with r3_header:
            st.text("Model 1 results")
        count_rate_data = result_viewer.calculate_filtered_results(model_1_raw_data, rv_m1_threshold)
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
                defects = helper.get_lrf_data_lists(
                    output_dir=rv_m1_output_dir, cols=["ClassType"], include_prob=False
                )[0]
                classtype_counter_df = result_viewer.get_classtype_count(defects)
                st.caption(f"LRF type: {model_1_metadata['input_lrf_type']}")
                st.dataframe(data=classtype_counter_df)

        # Draw 1D comparison chart
        with vr3_col1:
            st.plotly_chart(result_viewer.generate_1D_plot(model_1_raw_data, rv_m1_threshold))

        with vr3_col2:
            # TODO: This should be done somewhere else
            if 1 not in set(model_1_raw_data[2]):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")

            else:
                model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, return_curve=True)[0]
                st.plotly_chart(
                    result_viewer.plot_roc(
                        [
                            (
                                "Model 1",
                                model_1_roc_data,
                                rv_m1_threshold,
                                model_1_threshold,
                                rv_m1_output_dir,
                            )
                        ]
                    )
                )

    else:
        pass
