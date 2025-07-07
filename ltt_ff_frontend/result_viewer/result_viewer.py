import json
import time

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
RECIPE_INPUT_MODES = [YAML_MODE, DB_MODE, CREATOR_MODE]


def app() -> None:
    logger.debug("Loading Result Viewer...")
    st.title("False Filter Result Viewer")
    st.caption("Visualize False Filter Result")

    r1_col1, r1_col2, r1_col3 = st.columns([3, 3, 1])

    with r1_col1:
        inference_result_dir = st.text_input("Inference Result Directory", value=INFERENCE_DEFAULT_RESULT_DIR)
    with r1_col2:
        st_recipe_type = st.segmented_control("Recipe UI", RECIPE_INPUT_MODES, default=DB_MODE)
        if st_recipe_type is None:
            st.error("Recipe UI Option can not be None!")
            return

    if helper.is_valid_output_dir(inference_result_dir):
        try:
            get_metadata_start_time = time.perf_counter()
            db_metadata_list = api_helper.get_db_metadata_lists(output_dir=inference_result_dir)
            get_metadata_time_taken = time.perf_counter() - get_metadata_start_time
            logger.warning(f"Elapsed time (get metadata for each DB): {get_metadata_time_taken:.6f} seconds.")
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

    recipe_preview_start_time = time.perf_counter()
    recipe = recipe_preview.gen(recipe_type=st_recipe_type, db_recipe=db_recipe)
    if not recipe or recipe["recipes"] == []:
        return
    recipe_preview_time_taken = time.perf_counter() - recipe_preview_start_time
    logger.warning(f"Elapsed time (recipe_preview_time_taken): {recipe_preview_time_taken:.6f} seconds.")

    if st_recipe_type in [YAML_MODE, CREATOR_MODE] and recipe and db_recipe:
        new_lrf_button.gen(r1_col3, inference_result_dir, recipe, helper.filter_recipe_columns(db_recipe))

    # Show Total/Defect/Non-defect/unlabeled count
    with st.container():
        st.subheader("Inference Results")

    required_components = [
        ResultViewerComponents.OOS_SUMMARY.value,
        ResultViewerComponents.MISSED_DEFECT_LIST.value,
        ResultViewerComponents.PARTICLE_MODE_ONLY_DEFECT_LIST.value,
        ResultViewerComponents.CLASSTYPE_COUNT.value,
    ]

    generate_summary_components_start = time.perf_counter()
    try:
        result_viewer_components = api_helper.get_result_viewer_components(
            inference_result_dir=inference_result_dir,
            recipe=recipe,
            required_components=required_components,
            required_input={
                "recipe_threshold": recipe["recipes"][0]["threshold"],
            },
        )
    except ValueError as e:
        st.error(e)
        return

    try:
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
    except KeyError as e:
        st.error(e)
        return

    generate_summary_components_time_taken = time.perf_counter() - generate_summary_components_start
    logger.warning(f"Elapsed time (generate_summary_components): {generate_summary_components_time_taken:.6f} seconds.")
