# ruff: noqa: F841
import json
import os

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import helper, stop_job_button

from .regression_constant import LAYERS, SITES
from .regression_utils import gen_lots_stats, get_detailed_stats, update_config_by_txt, update_config_by_yaml


DEFAULT_DATA_YAML = "/mnt/dbpc/FalseFilterDataSet/WeeklyYaml/WXXX_data_XXX.yaml"
DEFAULT_HOLDOUT_DATA_YAML = "/mnt/dbpc/FalseFilterDataSet/WeeklyYaml/WXXX_data_XXX_holdout.yaml"
DEFAULT_OUTPUT_PATH = "/mnt/fs0/MLE/ff_docker_output/mle_regression_test/"


def app():
    st.title("Regression Test Dashboard")
    st.caption("Configure and run regression tests")

    # Upload section
    r1_col1, _, r1_col3 = st.columns([10, 1, 10])
    with r1_col1:
        recipe_file = st.file_uploader("Upload Recipe Config (.json)")
    with r1_col3:
        yaml_help_text = f"**Example:**\n```yaml\n{DEFAULT_DATA_YAML}\n```"
        data_yaml = st.file_uploader("Upload Test Data (.yaml)", type="yaml", help=yaml_help_text)

    # Recipe config section
    r2_col1, _, r2_col2 = st.columns([10, 1, 10])
    with r2_col1:
        recipe_config = {}
        if recipe_file:
            st.subheader("📄 Recipe Preview")
            recipe_config = json.load(recipe_file)
            with st.expander("Original Recipe Config", expanded=False):
                st.json(recipe_config)

            update_recipe_file = st.file_uploader("Upload Update Recipe File (.txt or .yaml)", type=["txt", "yaml"])
            if update_recipe_file:
                if update_recipe_file.name.endswith(".txt"):
                    update_config_by_txt(recipe_config, update_recipe_file)
                else:
                    update_config_by_yaml(recipe_config, update_recipe_file)
                st.success("Recipe config updated successfully")
                with st.expander("🆕 Updated Recipe Config", expanded=True):
                    st.json(recipe_config)

                updated_json_str = json.dumps(recipe_config, indent=2)
                st.download_button(
                    label="💾 Download Updated Recipe Config",
                    data=updated_json_str,
                    file_name="updated_recipe_config.json",
                    mime="application/json",
                )

    # Test data section
    valid_data_lots = None
    if data_yaml:
        with r2_col2:
            show_lot_stats = st.toggle("Show Lot Statistics", value=True)
            show_detaild_stats = st.toggle("Show Detailed Lot Statistics", value=False)
            test_data = yaml.safe_load(data_yaml)
            valid_data_lots = api_helper.fetch_valid_lots(test_data, show_detaild_stats)
            if valid_data_lots["status"] != "completed":
                st.error(valid_data_lots["message"])
                logger.error(valid_data_lots["message"])
                return
            if show_lot_stats:
                with st.expander("Lots Statistics", expanded=False):
                    st.dataframe(gen_lots_stats(valid_data_lots["valid_lots"]))
            if show_detaild_stats:
                with st.expander("Detailed Lots Statistics", expanded=False):
                    st.dataframe(get_detailed_stats(valid_data_lots["stats"]))

    # Filters
    r3_col1, _, r3_col3 = st.columns([10, 1, 10])
    with r3_col1:
        layer_filter = st.multiselect("Run Layers", LAYERS, default=LAYERS)
        output_dir = st.text_input("Output Directory", value=DEFAULT_OUTPUT_PATH)
    with r3_col3:
        site_filter = st.multiselect("Run Sites", SITES, default=SITES)
        run_holdout = st.toggle("Run additional holdout datasets", value=True)
        holdout_valid_data_lots = None

        if run_holdout:
            holdout_data_yaml = st.file_uploader("Upload Holdout Test Data (.yaml)", type="yaml", help=yaml_help_text)
            if holdout_data_yaml:
                show_h_lot_stats = st.toggle("Show Lot Statistics", value=True)
                show_h_detaild_stats = st.toggle("Show Detailed Lot Statistics", value=False)
                st.subheader("📊 Holdout Lot Statistics")
                holdout_test_data = yaml.safe_load(holdout_data_yaml)
                holdout_valid_data_lots = api_helper.get_valid_lots(holdout_test_data, show_h_detaild_stats)

                if holdout_valid_data_lots["status"] != "completed":
                    st.error(f"Holdout validation failed: {holdout_valid_data_lots['message']}")
                    logger.error(f"[Holdout Error] {holdout_valid_data_lots['message']}")
                    return

                if show_h_lot_stats:
                    with st.expander("Lots Statistics", expanded=False):
                        st.dataframe(gen_lots_stats(holdout_valid_data_lots["valid_lots"]))
                if show_h_detaild_stats:
                    with st.expander("Detailed Lots Statistics", expanded=False):
                        st.dataframe(get_detailed_stats(holdout_valid_data_lots["stats"]))

    # Run button
    if st.button("Run Regression Test", type="primary"):
        if not helper.is_valid_output_dir(output_dir):
            st.error("Please enter a valid output directory.")
            return
        reg_output_dir = os.path.normpath(output_dir)

        if not recipe_config or not valid_data_lots:
            st.error("Missing required inputs.")
            return

        logger.info("[Regression Test] Starting main test...")
        request = api_helper.request_regression_test(
            reg_output_dir, valid_data_lots["valid_lots"], recipe_config, layer_filter, site_filter
        )

        if request.json().get("status") == "error":
            code = request.json().get("code")
            message = request.json().get("message")
            st.error(f"Error code: {code}\nMessage: {message}")
            logger.error(f"[Regression Test Error] Code: {code}, Message: {message}")
        else:
            st.success("Regression test started successfully!")
            st.session_state.testcase_inference_map = request.json().get("testcase_inference_map")

        if holdout_valid_data_lots:
            logger.info("[Regression Test] Starting holdout test...")
            holdout_request = api_helper.request_regression_test(
                reg_output_dir + "_holdout",
                holdout_valid_data_lots["valid_lots"],
                recipe_config,
                layer_filter,
                site_filter,
            )
            if holdout_request.json().get("status") == "error":
                code = holdout_request.json().get("code")
                message = holdout_request.json().get("message")
                st.error(f"Holdout Error code: {code}\nMessage: {message}")
                logger.error(f"[Holdout Test Error] Code: {code}, Message: {message}")
            else:
                st.success("Holdout regression test started successfully!")

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
            st.session_state.status_df_multi_inf = api_helper.request_paginated_multilot_inference_status(
                page_size, current_page
            )

    progress_column = st.column_config.ProgressColumn(label="progress_bar", min_value=0, max_value=100)

    # Pagination settings
    with col2:
        page_size = 10
        current_page = st.number_input("Page number", min_value=1, value=1, step=1, key="multilot_page")
        st.session_state.status_df_multi_inf = api_helper.request_paginated_multilot_inference_status(
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
            st.session_state.detailed_df_multi_inf = api_helper.request_multilot_inference_statuses(
                selected_multilot_inference_id
            )
            st.dataframe(st.session_state.detailed_df_multi_inf, use_container_width=True)
            stop_job_button.gen(st.session_state.detailed_df_multi_inf)
