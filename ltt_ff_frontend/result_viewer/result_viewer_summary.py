import json

import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import INFERENCE_DEFAULT_RESULT_DIR, ResultViewerComponents
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import (
    class_type_component,
    helper,
    missed_defects_component,
    new_lrf_button,
    oos_summary_component,
    particle_mode_defects_component,
    recipe_preview,
)


YAML_MODE = "Yaml"
DB_MODE = "Database"
CREATOR_MODE = "Creator"
CUSTOM_MODE = "Custom"
RECIPE_INPUT_MODES = [YAML_MODE, DB_MODE, CREATOR_MODE, CUSTOM_MODE]


def app() -> None:
    logger.debug("Loading Result Viewer Summary...")
    st.title("False Filter Result Viewer Summary")
    st.caption("Quick overview of False Filter Result")

    result_dir_col, gen_lrf_button_col = st.columns([6, 2])

    recipe_mode_col, read_children_dirs_toggle_col, _ = st.columns([3, 3, 2])

    with result_dir_col:
        inference_result_dir = st.text_input("Inference Result Directory", value=INFERENCE_DEFAULT_RESULT_DIR)
    with recipe_mode_col:
        st_recipe_type = st.segmented_control("Recipe UI", RECIPE_INPUT_MODES, default=DB_MODE)
        if st_recipe_type is None:
            st.error("Recipe UI Option can not be None!")
            return
    with read_children_dirs_toggle_col:
        read_children_dirs = st.toggle(label="Read .db in children directories", value=False)

    if helper.is_valid_output_dir(inference_result_dir):
        try:
            db_metadata_list = api_helper.get_db_metadata_lists(
                output_dir=inference_result_dir, read_children_dirs=read_children_dirs
            )
        except ValueError as e:
            st.error(f"{e}\n\nError getting result data from {inference_result_dir}")
            return

        recipe_list = [metadata["recipe"] for metadata in db_metadata_list]
        if all(recipe == recipe_list[0] for recipe in recipe_list):
            db_recipe = json.loads(recipe_list[0])
        else:
            st.error("Not all lots use same recipe.")
            return
    else:
        db_recipe = {}
        if inference_result_dir != INFERENCE_DEFAULT_RESULT_DIR and inference_result_dir != "":
            st.error(f"Output directory is invalid: {inference_result_dir}")

    recipe = recipe_preview.gen(recipe_type=st_recipe_type, db_recipe=db_recipe)
    if not recipe or recipe["recipes"] == []:
        return

    if st_recipe_type != DB_MODE and recipe and db_recipe:
        new_lrf_button.gen(gen_lrf_button_col, inference_result_dir, recipe, helper.filter_recipe_columns(db_recipe))

    # Show Total/Defect/Non-defect/unlabeled count
    with st.container():
        st.subheader("Inference Results")

    required_components = {
        ResultViewerComponents.OOS_SUMMARY.value: True,
        ResultViewerComponents.MISSED_DEFECT_LIST.value: True,
        ResultViewerComponents.PARTICLE_MODE_LIST.value: True,
        ResultViewerComponents.CLASSTYPE_COUNT.value: True,
    }

    try:
        result_viewer_components = api_helper.get_result_viewer_components(
            inference_result_dir=inference_result_dir,
            recipe=recipe,
            required_components=required_components,
            required_input={
                "recipe_threshold": recipe["recipes"][0]["threshold"],
            },
            read_children_dirs=read_children_dirs,
        )
    except ValueError as e:
        st.error(e)
        return

    try:
        if ResultViewerComponents.OOS_SUMMARY.value in required_components:
            with st.expander(label="OOS Summary"):
                oos_summary_component.gen(oos_calculation=result_viewer_components["oos_summary"])

        if ResultViewerComponents.MISSED_DEFECT_LIST.value in required_components:
            with st.expander(label="Missed defects"):
                missed_defects_component.gen(missed_defect_info=result_viewer_components["missed_defect_info"])

        if ResultViewerComponents.PARTICLE_MODE_LIST.value in required_components:
            with st.expander(label="ParticleMode defects"):
                particle_mode_defects_component.gen(particle_mode_info=result_viewer_components["particle_mode_info"])

        if ResultViewerComponents.CLASSTYPE_COUNT.value in required_components:
            with st.expander(label="LRF ClassType count"):
                class_type_component.gen(classtype_count_list=result_viewer_components["classtype_count"])
    except KeyError as e:
        st.error(e)
        return
