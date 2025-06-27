# ruff: noqa: F841
import json

import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.helpers import api_helper

from .regression_utils import gen_lots_stats, update_config_by_txt, update_config_by_yaml


DEFAULT_API_SERVER = "http://192.168.201.11:7500"
DEFAULT_DATA_YAML = "/mnt/dbpc/FalseFilterDataSet/WeeklyYaml/WXXX_data_XXX.yaml"
DEFAULT_OUTPUT_DIR_ROOT = "/mnt/output/mle_regression_test"


def app():
    st.title("Regression Test Dashboard")
    st.caption("Configure and run regression tests")

    api_server = st.text_input("API Server", value=DEFAULT_API_SERVER)
    r1_col1, _r1_col2, r1_col3 = st.columns([10, 1, 10])
    with r1_col1:
        # Load config
        recipe_file = st.file_uploader("Upload Recipe Config (.json)")

    with r1_col3:
        yaml_help_text = f"**Example: {DEFAULT_DATA_YAML}**\n"
        data_yaml = st.file_uploader("Upload Test Data (.yaml)", type=".yaml", help=yaml_help_text)

    r2_col1, r2_col2 = st.columns([1, 1])
    with r2_col1:
        recipe_config = {}
        if recipe_file:
            # recipe_config = load_recipe_config(recipe_file)
            st.subheader("Recipe preview")
            recipe_config = json.load(recipe_file)
            with st.expander("📄 Original Recipe Config", expanded=False):
                st.json(recipe_config)

            update_recipe_file = st.file_uploader("Upload update recipe file (.txt or .yaml)", type=[".txt", ".yaml"])
            if update_recipe_file:
                if update_recipe_file.name.endswith(".txt"):
                    st.text(update_recipe_file)
                    update_config_by_txt(recipe_config, update_recipe_file)
                else:
                    st.json(update_recipe_file)
                    update_config_by_yaml(recipe_config, update_recipe_file)
                st.success("✅ Recipe config updated successfully")
                with st.expander("🆕 Updated Recipe Config", expanded=True):
                    st.json(recipe_config)

                updated_json_str = json.dumps(recipe_config, indent=2)
                st.download_button(
                    label="💾 Download Updated Recipe Config",
                    data=updated_json_str,
                    file_name="updated_recipe_config.json",
                    mime="application/json",
                )

    if data_yaml:
        with r2_col2:
            st.subheader("Lot Statistics")
            test_data = yaml.safe_load(data_yaml)
            valid_data_lots = api_helper.get_valid_lots(test_data)
            if valid_data_lots["status"] != "completed":
                st.error(valid_data_lots["message"])
                logger.error(valid_data_lots["message"])
                return
            st.dataframe(gen_lots_stats(valid_data_lots["valid_lots"]))

    # Layer and site filters
    layers = ["OD", "PO", "CUT", "M0M2", "M1", "VIA"]
    sites = ["F20", "F12", "F18A", "F18B", "F18EBO", "F15EBO"]
    r3_col1, _r3_col2, r3_col3 = st.columns([10, 1, 10])
    with r3_col1:
        layer_filter = st.multiselect("Layer Filter", layers, default=layers)
        output_dir = st.text_input("Output Directory", value=f"{DEFAULT_OUTPUT_DIR_ROOT}/2025-06-27")
    with r3_col3:
        site_filter = st.multiselect("Site Filter", sites, default=sites)
        run_modes = st.multiselect(
            "Run modes", ["Normal", "with Rule", "Rule only"], ["Normal", "with Rule", "Rule only"]
        )

    # Run button
    if st.button("Run Regression Test", type="primary"):
        st.success("Regression test triggered (mock).")  # Replace with actual API call

    st.divider()

    # Job status
    st.subheader("Multilot Job Status")
    if st.button("Check Multilot Jobs"):
        jobs = api_helper.request_paginated_multilot_inference_status(page_size=10, current_page=1)
        st.dataframe(jobs)

    st.subheader("Inference Job Status")
    if st.button("Check Inference Jobs"):
        jobs = api_helper.request_paginated_inference_status(page_size=10, current_page=1)
        st.dataframe(jobs)
