import json
import time
from datetime import datetime
from decimal import Decimal

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.constant import BLANK_MODEL, INFERENCE_DEFAULT_RESULT_DIR, ResultViewerComponents
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import (
    class_type_component,
    helper,
    missed_defects_component,
    multi_lot_stats,
    new_lrf_button,
    oos_summary_component,
    particle_mode_defects_component,
    prob_distribution_fig,
    roc_fig,
)


YAML_MODE = "Yaml"
DB_MODE = "Database"
CREATOR_MODE = "Creator"
RECIPE_INPUT_MODES = [YAML_MODE, DB_MODE, CREATOR_MODE]


def app() -> None:
    logger.debug("Loading Result Viewer...")
    st.title("False Filter Result Viewer")
    st.caption("Visualize False Filter Result")

    r1_col1, r1_col2, r1_col3 = st.columns([3, 3, 1])

    recipe = None

    with r1_col1:
        inference_result_dir = st.text_input("Inference Result Directory", value=INFERENCE_DEFAULT_RESULT_DIR)
    with r1_col2:
        st_recipe_type = st.segmented_control("Recipe UI", RECIPE_INPUT_MODES, default=DB_MODE)
        if st_recipe_type is None:
            st.error("Recipe UI Option can not be None!")
            return

    if helper.is_valid_output_dir(inference_result_dir):
        multi_lot_model_data = api_helper.get_multilot_model_data(inference_result_dir)
        with r1_col2:
            if multi_lot_model_data is None:
                st.error(f"Error getting result data from {inference_result_dir}")
                return
        recipe_list = [metadata["recipe"] for metadata in multi_lot_model_data.model_metadata_list]
        if all(recipe == recipe_list[0] for recipe in recipe_list):
            db_recipe = json.loads(recipe_list[0])
            filtered_db_recipe = helper.filter_recipe_columns(db_recipe)
        else:
            st.error("Not all lots use same recipe.")
            return
    else:
        filtered_db_recipe = {"recipes": []}

    if st_recipe_type == YAML_MODE:
        col, _ = st.columns([3, 4])
        with col:
            yaml_help_text = """
            **Example of a valid recipe:**\n
            recipes:\n
            \\- model_name: base/model_1.encrypted.pth\n
            &nbsp;&nbsp;threshold: 0.5\n
            \\- model_name: base/model_2.encrypted.pth\n
            &nbsp;&nbsp;threshold: 0.9\n
            """
            recipe_file = st.file_uploader("Upload Recipe (.yaml)", type=".yaml", help=yaml_help_text)
            if recipe_file is not None:
                uploaded_recipe = yaml.load(recipe_file, Loader=yaml.Loader)
                recipe_to_preview = uploaded_recipe
                recipe = uploaded_recipe
    elif st_recipe_type == DB_MODE:
        if helper.is_valid_output_dir(inference_result_dir):
            recipe_to_preview = filtered_db_recipe
            recipe = db_recipe
    elif st_recipe_type == CREATOR_MODE:
        available_models = api_helper.get_base_models(include_blank=True)
        recipe_models = []
        recipe_model_thresholds = []
        for i in range(5):
            col1, col2, _ = st.columns([3, 2, 2])
            with col1:
                recipe_model = st.selectbox(
                    f"Model {i + 1}", options=available_models, format_func=helper.format_model_name
                )
                recipe_models.append(recipe_model)
            with col2:
                input_threshold = st.number_input(
                    label=f"Model {i + 1} threshold:",
                    value=api_helper.get_model_threshold(model_name=recipe_model),
                    step=1e-5,
                    format="%.5f",
                    help="Probabilities below threshold will be considered as non-defects.",
                )
                # Use Decimal for precise floating point arithmetic
                # Passing the threshold as a string ensures that it does not first get interpreted as a float, which
                # can introduce a precision error.
                rounded_threshold = int(Decimal(f"{input_threshold}") * Decimal("1e5")) / Decimal("1e5")
                recipe_model_thresholds.append(float(rounded_threshold))
            # with col3:
            #     recipe_model_shf = st.toggle("Model {i + 1} SHF")
        user_input_recipe = {"recipes": []}
        for recipe_model, threshold in zip(recipe_models, recipe_model_thresholds):
            if recipe_model != BLANK_MODEL:
                user_input_recipe["recipes"].append(
                    {
                        "model_name": recipe_model,
                        "threshold": threshold,
                    }
                )
        recipe = user_input_recipe
        recipe_to_preview = user_input_recipe
    st.divider()
    if recipe is not None and recipe["recipes"] != []:
        with st.expander(f"{st_recipe_type} Recipe preview:", expanded=True):
            st.code(yaml.dump(recipe_to_preview), language="yaml")
        st.divider()

        if filtered_db_recipe["recipes"] == []:
            # no data, early return
            return
        if st_recipe_type in [YAML_MODE, CREATOR_MODE]:
            new_lrf_button.gen(r1_col3, inference_result_dir, recipe, filtered_db_recipe)
    else:
        # recipe not ready, early return
        return

    # Show Total/Defect/Non-defect/unlabeled count
    with st.container():
        st.subheader("Inference Results")

    start = time.perf_counter()

    required_components = [
        ResultViewerComponents.OOS_SUMMARY.value,
        ResultViewerComponents.MISSED_DEFECT_LIST.value,
        ResultViewerComponents.PARTICLE_MODE_ONLY_DEFECT_LIST.value,
        ResultViewerComponents.CLASSTYPE_COUNT.value,
        ResultViewerComponents.INFERENCE_RESULT_TABLE.value,
    ]
    result_viewer_components = api_helper.get_result_viewer_components(
        inference_result_dir=inference_result_dir,
        recipe=recipe,
        required_components=required_components,
    )

    if "oos_summary" in required_components:
        with st.expander(label="OOS Summary"):
            oos_summary_component.gen(oos_calculation=result_viewer_components["oos_summary"])

    if "missed_defect_list" in required_components:
        with st.expander(label="Missed defects"):
            missed_defects_component.gen(missed_defects=result_viewer_components["missed_defect_list"])

    if "particle_mode_only_defect_list" in required_components:
        with st.expander(label="ParticleMode defects"):
            particle_mode_defects_component.gen(
                particle_mode_only_defects=result_viewer_components["particle_mode_only_defect_list"]
            )
    if "classtype_count" in required_components:
        with st.expander(label="LRF ClassType count"):
            class_type_component.gen(classtype_count_list=result_viewer_components["classtype_count"])

    end_summary = time.perf_counter()
    elapsed = end_summary - start
    logger.debug(f"Elapsed time (loading only summary components): {elapsed:.6f} seconds.")

    if "inference_result_table" in required_components:
        with st.expander(label="Inference Result Table"):
            selected_lot_id_list, df = multi_lot_stats.draw_stats_df(
                multi_lot_model_data, recipe, inference_result_dir, key="recipe_stats_df"
            )

        st.download_button(
            label="Download inference result table",
            data=df.to_csv(index=False),
            file_name=f"inference_result_{datetime.now().astimezone()}.csv",
            mime="text/csv",
        )
    else:
        selected_lot_id_list = None

    if len(recipe["recipes"]) == 1:
        # Columns for drawing distribution chart and ROC curve
        col_1d_chart_column, col_roc_curve_column = st.columns(2)
        recipe_model_name = recipe["recipes"][0]["model_name"]
        recipe_threshold = recipe["recipes"][0]["threshold"]
        model_raw_data = (
            multi_lot_model_data.defect_id_lists,
            multi_lot_model_data.probability_lists,
            multi_lot_model_data.answer_lists,
        )
        # Draw 1D comparison chart
        with col_1d_chart_column:
            with st.expander(label="1D Prob Distribution Chart"):
                st.plotly_chart(
                    prob_distribution_fig.generate_multilot_1D_plot(
                        model_raw_data, multi_lot_model_data.model_metadata_list, recipe_threshold, selected_lot_id_list
                    )
                )
        with col_roc_curve_column:
            with st.expander(label="Roc Curve Chart"):
                roc_fig.gen(
                    recipe_model_name,
                    inference_result_dir,
                    model_raw_data,
                    multi_lot_model_data.model_metadata_list,
                    recipe_threshold,
                    selected_lot_id_list,
                )

    end_all = time.perf_counter()
    elapsed = end_all - start
    logger.debug(f"Elapsed time (loading summary components + result table): {elapsed:.6f} seconds.")
