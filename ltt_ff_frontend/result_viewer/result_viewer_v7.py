import glob

import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.constant import BLANK_MODEL
from ltt_ff_frontend.helpers import api_helper, ui_helper
from ltt_ff_frontend.result_viewer import multi_lot_result_viewer_v7, single_lot_result_viewer_v7

DB_OR_YAML_MODE = "DB_OR_YAML"
CREATOR_MODE = "CREATOR"
RECIPE_INPUT_MODES = [DB_OR_YAML_MODE, CREATOR_MODE]

def app() -> None:
    logger.debug("Loading V7 Result Viewer...")
    st.title("False Filter Result Viewer (Recipe)")
    st.caption("Visualize False Filter Result")

    r1_col1, r1_col2, r1_col3 = st.columns([3, 3, 1], border=True)

    output_dir_default = "/mnt/dbpc/xxx"
    single_lot_single_model_dir = '/home/ronyauw/0411/single_lot_single_model/'
    single_lot_two_models_recipe_dir = '/home/ronyauw/0411/single_lot_two_model/'
    single_lot_three_models_recipe_dir = '/home/ronyauw/0411/single_lot_three_model/'
    multi_lot_single_model_dir = '/home/ronyauw/0411/multi_lot_single_model/'
    multi_lot_two_model_dir = '/home/ronyauw/0411/multi_lot_two_model/'
    multi_lot_three_model_dir = '/home/ronyauw/0411/multi_lot_three_model/'

    recipe_of_two_model_path = '/home/ronyauw/0411/recipe_two_model.yaml'
    recipe_of_three_model_path = '/home/ronyauw/0411/recipe_three_model.yaml'
    user_upload_recipe = None
    with r1_col1:
        rv_output_dir = st.text_input("Inference Result Directory", value=multi_lot_single_model_dir)
    with r1_col2:
        st_recipe_type = st.segmented_control("Recipe UI", RECIPE_INPUT_MODES, default=DB_OR_YAML_MODE)
        if st_recipe_type is None:
            st.error("Recipe UI Option can not be None!")
            return
    with r1_col3:
        if st.button("Generate new lrf with Recipe"):
            request = api_helper.request_recipe_lrf(output_dir=rv_output_dir, recipe=user_upload_recipe, lot_id="")
            if request.json().get("status") == "error":
                code = request.json().get("code")
                message = request.json().get("message")
                st.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                logger.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
            else:
                st.success(f"New .lrf file using recipe generated at {rv_output_dir}!")
                logger.info(f"New .lrf file using recipe generated at {rv_output_dir}!")

    if st_recipe_type == DB_OR_YAML_MODE:
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
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
                user_upload_recipe = yaml.load(recipe_file, Loader=yaml.Loader)
    elif st_recipe_type == CREATOR_MODE:
        available_models = api_helper.get_base_models(include_blank=True)
        recipe_models = []
        recipe_model_thresholds = []
        for i in range(5):
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                recipe_model = st.selectbox(
                    f"Model {i+1}",
                    options=available_models,
                    format_func=ui_helper.format_model_name
                )
                recipe_models.append(recipe_model)
            with col2:
                input_threshold = st.number_input(
                    label=f"Model {i+1} threshold:",
                    value=api_helper.get_model_threshold(model_name=recipe_model),
                    step=0.00001,
                    format="%.5f",
                    help="Probabilities below threshold will be considered as non-defects.",
                )
                recipe_model_thresholds.append(input_threshold)
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
        if recipe is not None:
            st.subheader("Recipe preview:")
            st.code(yaml.dump(recipe), language="yaml")

            st.divider()
    if user_upload_recipe is not None:
        with st.expander("Recipe preview:"):
            st.code(yaml.dump(user_upload_recipe), language="yaml")
        st.divider()

    db_files = glob.glob(f"{rv_output_dir}/*.db")
    if len(db_files) > 1:
        multi_lot_result_viewer_v7.app(output_dir_default, rv_output_dir, user_upload_recipe=user_upload_recipe)
    else:
        single_lot_result_viewer_v7.app(output_dir_default, rv_output_dir, user_upload_recipe=user_upload_recipe)
