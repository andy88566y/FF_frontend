import json

import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.constant import BLANK_MODEL
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import (
    class_type_component,
    helper,
    multi_lot_stats,
    new_lrf_button,
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

    output_dir_default = "/mnt/dbpc/xxx"
    invalid_input = [output_dir_default, ""]
    recipe = None

    with r1_col1:
        inference_result_dir = st.text_input("Inference Result Directory", value=output_dir_default)
    with r1_col2:
        st_recipe_type = st.segmented_control("Recipe UI", RECIPE_INPUT_MODES, default=DB_MODE)
        if st_recipe_type is None:
            st.error("Recipe UI Option can not be None!")
            return

    if inference_result_dir not in invalid_input:
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
                recipe = yaml.load(recipe_file, Loader=yaml.Loader)
                multi_lot_model_data = api_helper.get_multilot_model_data(inference_result_dir)
    elif st_recipe_type == DB_MODE:
        if inference_result_dir in invalid_input:
            st.warning("Fill in Inference Result Directory")
            return
        recipe = filtered_db_recipe
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
                rounded_threshold = int(input_threshold * 1e5) / 1e5
                recipe_model_thresholds.append(rounded_threshold)
            # with col3:
            #     recipe_model_shf = st.toggle("Model {i + 1} SHF")
        recipe = {"recipes": []}
        for recipe_model, threshold in zip(recipe_models, recipe_model_thresholds):
            if recipe_model != BLANK_MODEL:
                recipe["recipes"].append(
                    {
                        "model_name": recipe_model,
                        "threshold": threshold,
                    }
                )
    st.divider()
    if recipe is not None:
        with st.expander(f"{st_recipe_type} Recipe preview:", expanded=True):
            st.code(yaml.dump(recipe), language="yaml")
        st.divider()

        if st_recipe_type in [YAML_MODE, CREATOR_MODE] and recipe["recipes"] is not []:
            new_lrf_button.gen(r1_col3, inference_result_dir, recipe, db_recipe)

    if st_recipe_type == YAML_MODE and recipe is None:
        st.warning("Empty Recipe in YAML mode, please upload valid YAML!")
        return

    if inference_result_dir in invalid_input:
        st.warning("Inference Result Directory invalid.")
        return
    helper.disallow_invalid_output_dir(inference_result_dir)

    multi_lot_model_data = api_helper.get_multilot_model_data(inference_result_dir)

    if recipe is None or recipe["recipes"] == []:
        return
    # Show Total/Defect/Non-defect/unlabeled count
    with st.container():
        st.subheader("Inference Results")

    with st.container():
        selected_lot_id_list = multi_lot_stats.draw_stats_df(
            multi_lot_model_data, recipe, inference_result_dir, key="recipe_stats_df"
        )

    # TODO: Get classtype grouping from backend
    with st.container():
        with st.expander(label="LRF ClassType count"):
            class_type_component.gen(inference_result_dir, multi_lot_model_data.model_metadata_list)
    if recipe is not None and len(recipe["recipes"]) == 1:
        recipe_model_name = recipe["recipes"][0]["model_name"]
        recipe_threshold = recipe["recipes"][0]["threshold"]
        # Columns for drawing distribution chart and ROC curve
        col_1d_chart_column, col_roc_curve_column = st.columns(2)
        model_raw_data = (
            multi_lot_model_data.defect_id_lists,
            multi_lot_model_data.probability_lists,
            multi_lot_model_data.answer_lists,
        )
        # Draw 1D comparison chart
        with col_1d_chart_column:
            st.plotly_chart(
                prob_distribution_fig.generate_multilot_1D_plot(
                    model_raw_data, multi_lot_model_data.model_metadata_list, recipe_threshold, selected_lot_id_list
                )
            )
        with col_roc_curve_column:
            roc_fig.gen(
                recipe_model_name,
                inference_result_dir,
                model_raw_data,
                multi_lot_model_data.model_metadata_list,
                recipe_threshold,
                selected_lot_id_list,
            )
        st.divider()
