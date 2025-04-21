import glob

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
    st.title("False Filter Result Viewer (Recipe)")
    st.caption("Visualize False Filter Result")

    r1_col1, r1_col2, r1_col3 = st.columns([3, 3, 1])

    output_dir_default = "/mnt/dbpc/xxx"
    invalid_input = [output_dir_default, ""]
    recipe = None

    with r1_col1:
        inference_result_dir = st.text_input("Inference Result Directory", value=output_dir_default)
    with r1_col2:
        if inference_result_dir in invalid_input:
            st.error("Inference Result Directory is invalid.")
            return

        multi_lot_model_data = api_helper.get_multilot_model_data(inference_result_dir)
        if multi_lot_model_data is None:
            st.error(f"Error getting result data from {inference_result_dir}")
            return

        st_recipe_type = st.segmented_control("Recipe UI", RECIPE_INPUT_MODES, default=DB_MODE)
        if st_recipe_type is None:
            st.error("Recipe UI Option can not be None!")
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
    elif st_recipe_type == DB_MODE:
        first_model_metadata = multi_lot_model_data.model_metadata_list[0]
        recipe = {"recipes": yaml.load(first_model_metadata["recipe"], Loader=yaml.Loader)}
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
                rounded_threshold = round(input_threshold, 5)
                recipe_model_thresholds.append(rounded_threshold)
            # with col3:
            #     recipe_model_suf = st.toggle("Model 1 SUF")
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

        new_lrf_button.gen(r1_col3, inference_result_dir, recipe)

    db_files = glob.glob(f"{inference_result_dir}/*.db")
    if st_recipe_type == YAML_MODE and recipe is None:
        st.warning("Yaml mode, wating for uploading yaml file.")
        return

    # Defining columns to display filter results (capture rate, filter rate, etc.)
    with st.container():
        r3_header = st.empty()
        model_statistics = st.empty()
    with st.container():
        classtype_count = st.empty()

    st.divider()

    # Show Total/Defect/Non-defect/unlabeled count
    with r3_header:
        st.subheader("Recipe Results")

    with model_statistics.container():
        selected_lot_id_list = multi_lot_stats.draw_stats_df(
            multi_lot_model_data, recipe, inference_result_dir, key="recipe_stats_df"
        )

    # TODO: Get classtype grouping from backend
    with classtype_count:
        with st.expander(label="LRF ClassType count"):
            class_type_component.gen(inference_result_dir, multi_lot_model_data.model_metadata_list)
    if recipe is not None and len(recipe["recipes"]) == 1:
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
            roc_fig.gen_fig(
                inference_result_dir,
                model_raw_data,
                multi_lot_model_data.model_metadata_list,
                recipe_threshold,
                selected_lot_id_list,
            )
        st.divider()
