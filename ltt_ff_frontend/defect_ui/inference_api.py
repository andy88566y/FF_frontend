import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def app() -> None:
    logger.debug("Loading Inference Dashboard...")
    st.title("False Filter Inference")
    st.caption("Inference lot data with selected model")

    r1_col1, r1_col2, r1_col3 = st.columns([3, 2, 2])
    with r1_col1:
        inf_base_model = st.selectbox(
            "Base model", options=helper.get_base_models(), index=0, format_func=helper.format_model_name
        )
    with r1_col2:
        model_threshold = helper.get_model_threshold(model_name=inf_base_model)
        inf_filter_threshold = st.number_input(
            label="Confidence threshold:",
            value=model_threshold,
            step=0.00001,
            format="%.5f",
            help="Probabilities above threshold will be considered as defects.",
        )
    with r1_col3:
        inf_overwrite = st.toggle(label="Overwrite files in output directory", value=False)
        st.caption(":red[If Overwrite is set to true, all existing files in Result Directory will be removed.]")

    r2_col1, r2_col2 = st.columns([1, 1])
    with r2_col1:
        inf_lot_id = st.text_input(label="Lot ID", value="", help="Name of the lot of defect images.")
    with r2_col2:
        inf_output_dir = st.text_input(
            label="Result directory", value="/mnt/dbpc/xxx", help="The directory to store generated .lrf and .db files."
        )

    r3_col1, r3_col2 = st.columns([1, 1])
    with r3_col1:
        inf_image_dir = st.text_input(
            label="Image directory", value="/mnt/dbpc/xxx", help="The directory that contains the Images folder."
        )
    with r3_col2:
        inf_lrf_path = st.text_input(
            label=".lrf path", value="/mnt/dbpc/xxx", help="Absolute path to the selected .lrf file."
        )

    if st.button("Start Inference Job", type="primary"):
        # Validate user input first
        required_input = [inf_lot_id, inf_output_dir, inf_image_dir, inf_lrf_path]
        for item in required_input:
            if not item:
                logger.error(
                    "Missing user input detected. Please enter Lot ID/Result Directory/Image Directory/.lrf path."
                )
                st.error("Missing user input detected. Please enter Lot ID/Result Directory/Image Directory/.lrf path.")
                return

        # Validate confidence threshold
        if inf_filter_threshold < 0.0 or inf_filter_threshold > 1.0:
            logger.error(
                f"Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {inf_filter_threshold}"
            )
            st.error(
                f"Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {inf_filter_threshold}"
            )
            return

        request = helper.request_inference(
            base_model=inf_base_model,
            lot_id=inf_lot_id,
            output_dir=inf_output_dir,
            image_dir=inf_image_dir,
            lrf_path=inf_lrf_path,
            confidence_threshold=inf_filter_threshold,
            overwrite=inf_overwrite,
        )

        if request.json().get("status") == "error":
            code = request.json().get("code")
            message = request.json().get("message")
            st.text(f"Error code: {code}\nError message: {message}")
        else:
            inference_id = request.json().get("inference_id")
            st.text(f"Inference Job ID: {inference_id}")

    st.divider()

    if "status_df_inf" not in st.session_state:
        st.session_state.status_df_inf = pd.DataFrame()
    if "detailed_df_inf" not in st.session_state:
        st.session_state.detailed_df_inf = pd.DataFrame()

    col1, col2 = st.columns(2, vertical_alignment="bottom")

    with col1:
        if st.button("Check all inference jobs"):
            page_size = 10
            current_page = 1
            st.session_state.status_df_inf = helper.request_paginated_inference_status(page_size, current_page)

    progress_column = st.column_config.ProgressColumn(label="progress_bar", min_value=0, max_value=100)

    # Pagination settings
    with col2:
        page_size = 10
        current_page = st.number_input("Page number", min_value=1, value=1, step=1)
        st.session_state.status_df_inf = helper.request_paginated_inference_status(page_size, current_page)

    st.header("All inference jobs") if not st.session_state.status_df_inf.empty else st.write("")

    # Selection to find more detail
    event_inf = (
        st.dataframe(
            st.session_state.status_df_inf,
            key="statuses_inference",
            on_select="rerun",
            selection_mode="multi-row",
            use_container_width=True,
            column_config={"progress": progress_column},
        )
        if not st.session_state.status_df_inf.empty
        else st.write("")
    )

    if event_inf and event_inf.selection:
        # Check if the 'row' value's list is not empty
        if event_inf.selection["rows"]:
            # Get list of inference_id for all selected inference jobs
            selected_inference_id = [
                st.session_state.status_df_inf.iloc[i]["inference_id"] for i in event_inf.selection["rows"]
            ]

            # Get detailed statuses for each inference job and combine into one df
            raw_df_inf = helper.request_inference_statuses(selected_inference_id)
            st.session_state.detailed_df_inf = raw_df_inf
            st.dataframe(st.session_state.detailed_df_inf, use_container_width=True)
            # Stop job button
            helper.add_stop_job_button(raw_df_inf)
