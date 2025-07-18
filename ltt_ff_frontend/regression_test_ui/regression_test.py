# ruff: noqa: F841
import os

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.regression_test_ui.test_utils import (
    gen_lots_stats,
    get_detailed_stats,
    get_valid_lg_from_recipe,
    get_valid_site_from_recipe,
    get_valid_week_from_recipe,
    update_config_by_txt,
    update_config_by_yaml,
)
from ltt_ff_frontend.shared_components import helper, stop_job_button


DEFAULT_DATA_YAML = "/mnt/dbpc/FalseFilterDataSet/WeeklyYaml/WXXX_data_XXX.yaml"
DEFAULT_HOLDOUT_DATA_YAML = "/mnt/dbpc/FalseFilterDataSet/WeeklyYaml/WXXX_data_XXX_holdout.yaml"
DEFAULT_OUTPUT_PATH = "/mnt/output/mle_regression_test/"
UPLOAD_MODE = "Upload"
EDIT_MODE = "Edit"
UPDATE_MODE = [UPLOAD_MODE, EDIT_MODE]
TEXT_EXAMPLE = """ODPO
                    [OD][F20] *new* 0710-34764f49 (TH: 0, THC: 0.2) 0710-cd2a4c02 (TH: 0, THC: 0.5)\u200b
                    [OD][F12] 0424-a4058515 (TH: 0.14) 0522-009dc97e (TH: 0.3)\u200b
                    """


def app() -> None:
    st.title("Regression Test Dashboard")
    st.caption("Configure and run regression tests")

    # Upload section
    r1_col1, _, r1_col3 = st.columns([10, 1, 10])
    with r1_col1:
        recipe_file = st.file_uploader("Upload Regression Config (.yaml)")
    with r1_col3:
        yaml_help_text = f"**Example:**\n```yaml\n{DEFAULT_DATA_YAML}\n```"
        data_yaml = st.file_uploader("Upload Test Data (.yaml)", type="yaml", help=yaml_help_text)

    # Recipe config section
    r2_col1, _, r2_col2 = st.columns([10, 1, 10])
    if recipe_file:
        recipe_config = {}
        with r2_col1:
            recipe_config = yaml.safe_load(recipe_file)
            update_mode = st.segmented_control("Update Mode", UPDATE_MODE, default=UPLOAD_MODE)
            updated = False
            if update_mode == UPLOAD_MODE:
                update_recipe_file = st.file_uploader("Upload Update Recipe File (.txt or .yaml)", type=["txt", "yaml"])
                if update_recipe_file:
                    if update_recipe_file.name.endswith(".txt"):
                        update_config_by_txt(recipe_config, update_recipe_file)
                    else:
                        update_config_by_yaml(recipe_config, update_recipe_file)
                    updated = True
            else:
                code_input = st.text_area(
                    label="Please paste  the newer Regression config...", height=300, placeholder=TEXT_EXAMPLE
                )
                if code_input:
                    update_config_by_txt(recipe_config, code_input)
                    updated = True
            if updated:
                st.success("Regression recipe config updated successfully")
                updated_yaml_str = yaml.dump(recipe_config, sort_keys=False)
                st.download_button(
                    label="📄 Download Updated Regression Config (YAML)",
                    data=updated_yaml_str,
                    file_name="updated_regression_config.yaml",
                    mime="application/x-yaml",
                )
            with st.expander("Regression Config", expanded=False):
                st.json(recipe_config)
            layers = get_valid_lg_from_recipe(recipe_config)
            sites = get_valid_site_from_recipe(recipe_config)
            weeks = get_valid_week_from_recipe(recipe_config)

            layer_filter = st.multiselect("Run Layers", layers, default=layers)
            site_filter = st.multiselect("Run Sites", sites, default=sites)
            week_filter = st.multiselect("Run Weeks", weeks, default=weeks)

    # Test data section
    valid_data_lots = None
    if data_yaml:
        with r2_col2:
            show_lot_stats = st.toggle("Show Lot Statistics", value=False)
            show_detaild_stats = st.toggle("Show Detailed Lot Statistics", value=False)
            test_data = yaml.safe_load(data_yaml)
            valid_data_lots = api_helper.fetch_valid_lots(test_data, show_detaild_stats)
            if valid_data_lots["status"] != "completed":
                st.error(valid_data_lots["message"])
                return

            if show_lot_stats:
                with st.expander("Lots Statistics", expanded=False):
                    st.dataframe(gen_lots_stats(valid_data_lots["valid_lots"]))
            if show_detaild_stats:
                with st.expander("Detailed Lots Statistics", expanded=False):
                    st.dataframe(get_detailed_stats(valid_data_lots["stats"]))

    # Filters
    r4_col1, _, r4_col3 = st.columns([10, 1, 10])
    with r4_col1:
        output_dir = st.text_input("Output Directory", value=DEFAULT_OUTPUT_PATH)
    with r4_col3:
        run_holdout = st.toggle("Run additional holdout datasets", value=False)
        holdout_valid_data_lots = None

        if run_holdout:
            holdout_data_yaml = st.file_uploader("Upload Holdout Test Data (.yaml)", type="yaml", help=yaml_help_text)
            if holdout_data_yaml:
                show_h_lot_stats = st.toggle("Show Holdout Lot Statistics", value=False)
                show_h_detaild_stats = st.toggle("Show Holdout Detailed Lot Statistics", value=False)
                st.subheader("📊 Holdout Lot Statistics")
                holdout_test_data = yaml.safe_load(holdout_data_yaml)
                holdout_valid_data_lots = api_helper.fetch_valid_lots(holdout_test_data, show_h_detaild_stats)

                if holdout_valid_data_lots["status"] != "completed":
                    st.error(f"Holdout validation failed: {holdout_valid_data_lots['message']}")
                    logger.error(f"[Holdout Error] {holdout_valid_data_lots['message']}")
                    return

                if show_h_lot_stats:
                    with st.expander("Holdout Lots Statistics", expanded=False):
                        st.dataframe(gen_lots_stats(holdout_valid_data_lots["valid_lots"]))
                if show_h_detaild_stats:
                    with st.expander("Holdout Detailed Lots Statistics", expanded=False):
                        st.dataframe(get_detailed_stats(holdout_valid_data_lots["stats"]))

    generate_yaml = st.toggle("download regression test settings", value=True)

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
            reg_output_dir, valid_data_lots["valid_lots"], recipe_config, layer_filter, site_filter, week_filter
        )

        if request["status"] == "error":
            st.error(f"Error \nMessage: {request['message']}")
        else:
            st.success("Regression test started successfully!")
            st.session_state.testcase_inference_map = request["testcase_inference_map"]

            if generate_yaml:
                regression_config = {
                    "data_yaml_path": data_yaml.name,
                    "holdout_data_yaml_path": holdout_data_yaml.name if run_holdout and holdout_data_yaml else None,
                    "layers": layer_filter,
                    "sites": site_filter,
                    "weeks": week_filter,
                }
                os.makedirs(reg_output_dir, exist_ok=True)
                yaml_path = os.path.join(reg_output_dir, "regression_config.yaml")

                with open(yaml_path, "w") as f:
                    yaml.dump(regression_config, f)
                st.success(f"Test config saved to: {yaml_path}")

        if holdout_valid_data_lots:
            logger.info("[Regression Test] Starting holdout test...")
            holdout_request = api_helper.request_regression_test(
                reg_output_dir + "_holdout",
                holdout_valid_data_lots["valid_lots"],
                recipe_config,
                layer_filter,
                site_filter,
                week_filter,
            )
            if holdout_request["status"] == "error":
                st.error(f"Holdout test Error \nMessage: {request['message']}")
            else:
                st.success("Holdout regression test started successfully!")
        elif run_holdout:
            st.error("Holdout test Error \nMissing required inputs.")

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
