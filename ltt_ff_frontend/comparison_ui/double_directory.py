import json

import streamlit as st

from ltt_ff_frontend.constant import INFERENCE_DEFAULT_RESULT_DIR, ResultViewerComponents
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import helper, prob_2d_distribution_fig


NAME_SPACE = ["First Inference (Base) ", "Second Inference (Candidate)"]


def app() -> None:
    result_dirs = ["", ""]
    selected_models = [None, None]
    selected_thresholds = [0.0, 0.0]
    for i in range(2):
        result_dir_col, recipe_choice_col, th_choice_col = st.columns([2, 1, 1])
        db_recipe = None
        with result_dir_col:
            result_dirs[i] = st.text_input(f"{NAME_SPACE[i]} Result Directory", value=INFERENCE_DEFAULT_RESULT_DIR)
        if helper.is_valid_output_dir(result_dirs[i]):
            try:
                db_metadata_list = api_helper.get_db_metadata_lists(output_dir=result_dirs[i])
            except ValueError as e:
                st.error(f"{e}\n\nError getting result data from {result_dirs[i]}")
                return

            recipe_list = [metadata["recipe"] for metadata in db_metadata_list]
            if all(recipe == recipe_list[0] for recipe in recipe_list):
                db_recipe = json.loads(recipe_list[0])
            else:
                st.error("Not all lots use same recipe.")
                return
        else:
            if result_dirs[i] != INFERENCE_DEFAULT_RESULT_DIR and result_dirs[i] != "":
                st.error(f"{NAME_SPACE[i]} Result Directory is invalid: {result_dirs[i]}")
                return

        if not db_recipe:
            return
        model_map = helper.get_model_map(db_recipe)

        with recipe_choice_col:
            selected_models[i] = model_map[st.selectbox(f"Select for Model {i + 1}", list(model_map.keys()))]
        with th_choice_col:
            selected_thresholds[i] = st.number_input(
                label=f"Model {i + 1} threshold:",
                value=selected_models[i]["threshold"],
                step=1e-5,
                format="%.5f",
                help="Probabilities below threshold will be considered as non-defects.",
            )

    if "" in result_dirs or any(not model for model in selected_models):
        return

    try:
        result_viewer_components = api_helper.get_result_viewer_components(
            inference_result_dir=result_dirs[0],
            secondary_inference_result_dir=result_dirs[1],
            recipe=None,
            required_components=[ResultViewerComponents.TWO_D_DEFECT_DISTRIBUTION_CHART.value],
            required_input={
                "model_ids": [selected_models[0]["id"]],
                "model_ids_2": [selected_models[1]["id"]],
            },
        )
    except ValueError as e:
        st.error(e)
        return
    aggregated_model_data = list(result_viewer_components["two_d_defect_distribution_chart"].values())

    split_lot = st.toggle(label="Results split by lots", value=False)
    _, plot_container, _ = st.columns([1, 8, 1])
    with plot_container:
        st.plotly_chart(
            prob_2d_distribution_fig.gen(
                aggregated_model_data=aggregated_model_data,
                m1_name=selected_models[0]["model_hash"],
                m2_name=selected_models[1]["model_hash"],
                m1_threshold=selected_thresholds[0],
                m2_threshold=selected_thresholds[1],
                split_lot=split_lot,
            )
        )
