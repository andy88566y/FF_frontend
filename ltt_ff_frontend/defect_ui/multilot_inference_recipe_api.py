import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def app() -> None:
    #####################################################################################################
    # Running inference                                                                                 #
    #####################################################################################################
    logger.debug("Loading Multilot Inference Dashboard...")
    st.title("False Filter Multilot Inference Recipe")
    st.caption("Inference multiple lot data with recipe")

    r1_col1, _r1_col2, r1_col3 = st.columns([10, 1, 10])
    with r1_col1:
        yaml_help_text = """
        **Example of a valid recipe:**\n
        recipes:\n
        \- model_name: base/model_1.encrypted.pth\n
        &nbsp;&nbsp;threshold: 0.5\n
        \- model_name: base/model_2.encrypted.pth\n
        &nbsp;&nbsp;threshold: 0.9\n
        """
        recipe_file = st.file_uploader("Upload Inference Recipe (.yaml)", type=".yaml", help=yaml_help_text)
    with r1_col3:
        yaml_help_text = """
        **Example of a valid .yaml config file:**\n
        data_paths:\n
        \- lot_id: N0_M0-0_20240101_000000\n
        &nbsp;&nbsp;lrf_path: /mnt/dbpc/xxx/N0_M0-0_20240101_000000_classified.lrf\n
        &nbsp;&nbsp;image_dir: /mnt/dbpc/xxx/N0_M0-0_20240101_000000/N0_M0-0_20240101_000000\n
        \- lot_id: N0_M0-0_20240101_000000\n
        &nbsp;&nbsp;lrf_path: /mnt/dbpc/xxx/N0_M0-0_20240101_000000_classified.lrf\n
        &nbsp;&nbsp;image_dir: /mnt/dbpc/xxx/N0_M0-0_20240101_000000/N0_M0-0_20240101_000000
        """
        inf_configfile = st.file_uploader(
            "Upload Multi-lot Inference Config (.yaml)", type=".yaml", help=yaml_help_text
        )

    helper.gap(1)
    r2_col1, _r2_col2, r2_col3, _r2_col4, r2_col5 = st.columns([10, 1, 4, 2, 4])
    with r2_col1:
        inf_output_dir = st.text_input(
            label="Result directory",
            value="/mnt/dbpc/xxx",
            help="The directory to store generated .lrf and .db files.",
        )
    with r2_col3:
        inf_overwrite = st.toggle(
            label="Overwrite files in output directory",
            value=False,
        )
        st.caption(":red[If Overwrite is set to true, all existing files in Result Directory will be removed.]")
    with r2_col5:
        inf_gen_optimized_recipe = st.toggle(
            label="Generate optimized recipe",
            value=False,
            key="inf_gen_optimized_recipe",
        )
        st.caption(":grey[Generate a recipe with optimized thresholds in the Result Directory.]")

    # Show recipe and multilot config previews
    r3_col1, r3_col2 = st.columns([1, 1])
    if recipe_file is not None:
        with r3_col1:
            st.subheader("Recipe preview")
            recipe = yaml.load(recipe_file, Loader=yaml.Loader)
            st.json(recipe)
    if inf_configfile is not None:
        with r3_col2:
            st.subheader("Multilot config preview")
            inf_config = yaml.load(inf_configfile, Loader=yaml.Loader)
            # TODO: Validate yaml file format from backend and pass error message
            st.json(inf_config)

    if st.button("Start Multilot Inference Job", type="primary"):
        # Validate user input first
        required_input = [inf_configfile, inf_output_dir]
        for item in required_input:
            if not item:
                logger.error("Missing input detected. Please upload .yaml config file and enter the Result Directory.")
                st.error("Missing input detected. Please upload .yaml config file and enter the Result Directory.")
                return

        # Validate confidence threshold
        for batch in recipe["recipes"]:
            if batch["threshold"] < 0.0 or batch["threshold"] > 1.0:
                logger.error(
                    f"Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {batch['threshold']}"
                )
                st.error(
                    f"Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {batch['threshold']}"
                )
                return

        request = helper.request_multilot_inference(
            output_dir=inf_output_dir,
            multilot_config=inf_config,
            recipe=recipe,
            overwrite=inf_overwrite,
            gen_optimized_recipe=inf_gen_optimized_recipe,
        )

        if request.json().get("status") == "error":
            code = request.json().get("code")
            message = request.json().get("message")
            st.text(f"Error code: {code}\nError message: {message}")
        else:
            inference_id = request.json().get("inference_id")
            st.text(f"Multilot Inference Job ID: {inference_id}")

    st.divider()

    #####################################################################################################
    # Multilot job status                                                                               #
    #####################################################################################################
    if "status_df_multi_inf" not in st.session_state:
        st.session_state.status_df_multi_inf = pd.DataFrame()
    if "detailed_df_multi_inf" not in st.session_state:
        st.session_state.detailed_df_multi_inf = pd.DataFrame()

    col1, col2 = st.columns(2, vertical_alignment="bottom")

    with col1:
        if st.button("Check all multilot inference jobs"):
            page_size = 10
            current_page = 1
            st.session_state.status_df_multi_inf = helper.request_paginated_multilot_inference_status(
                page_size, current_page
            )

    progress_column = st.column_config.ProgressColumn(label="progress_bar", min_value=0, max_value=100)

    # Pagination settings
    with col2:
        page_size = 10
        current_page = st.number_input("Page number", min_value=1, value=1, step=1, key="multilot_page")
        st.session_state.status_df_multi_inf = helper.request_paginated_multilot_inference_status(
            page_size, current_page
        )

    st.header("All multilot inference jobs") if not st.session_state.status_df_multi_inf.empty else st.write("")

    # Draw status overview table
    event_multilot_inf = (
        st.dataframe(
            st.session_state.status_df_multi_inf,
            key="statuses_multilot_inference",
            on_select="rerun",
            selection_mode="multi-row",
            use_container_width=True,
            column_config={"progress": progress_column},
        )
        if not st.session_state.status_df_multi_inf.empty
        else st.write("")
    )

    if event_multilot_inf and event_multilot_inf.selection:
        # Check if the 'row' value's list is not empty
        if event_multilot_inf.selection["rows"]:
            # Get list of inference_id for all selected inference jobs
            selected_multilot_inference_id = [
                st.session_state.status_df_multi_inf.iloc[i]["multilot_inference_id"]
                for i in event_multilot_inf.selection["rows"]
            ]

            # Get detailed statuses for each inference job and combine into one df
            st.session_state.detailed_df_multi_inf = helper.request_multilot_inference_statuses(
                selected_multilot_inference_id
            )
            st.dataframe(st.session_state.detailed_df_multi_inf, use_container_width=True)

    helper.gap(1)
    #####################################################################################################
    # Per lot job status                                                                                #
    #####################################################################################################
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
        current_page = st.number_input("Page number", min_value=1, value=1, step=1, key="per_lot_page")
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
