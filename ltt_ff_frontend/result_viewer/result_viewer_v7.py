import glob

import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.result_viewer import multi_lot_result_viewer_v7, single_lot_result_viewer_v7


def app() -> None:
    logger.debug("Loading V7 Result Viewer...")
    st.title("False Filter Result Viewer (Recipe)")
    st.caption("Visualize False Filter Result")

    r1_col1, r1_col2, r1_col3 = st.columns([3, 3, 1])
    # TODO: Switch to list
    # TODO: figure out why these containers are here
    # rc1_col1, rc1_col2, rc1_col3 = st.columns([3, 2, 2])
    # rc2_col1, rc2_col2, rc2_col3 = st.columns([3, 2, 2])
    # rc3_col1, rc3_col2, rc3_col3 = st.columns([3, 2, 2])
    # rc4_col1, rc4_col2, rc4_col3 = st.columns([3, 2, 2])
    # rc5_col1, rc5_col2, rc5_col3 = st.columns([3, 2, 2])

    output_dir_default = "/mnt/dbpc/xxx"
    single_lot_single_model_dir = '/home/ronyauw/0411/single_lot_single_model/'
    multi_lot_single_model_dir = '/home/ronyauw/0411/multi_lot_single_model/'
    single_lot_two_models_recipe_dir = '/home/ronyauw/0411/single_lot_two_model/'
    recipe_of_two_model_path = '/home/ronyauw/0411/recipe_two_model.yaml'
    single_lot_three_models_recipe_dir = '/home/ronyauw/0411/single_lot_three_model/'
    recipe_of_three_model_path = '/home/ronyauw/0411/recipe_three_model.yaml'
    st_recipe_type = 'yaml'
    recipe = None
    with r1_col1:
        rv_output_dir = st.text_input("Inference Result Directory", value=single_lot_two_models_recipe_dir)
    with r1_col2:
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
        # TODO: develop use, remove this when merge request is ready
        with open(recipe_of_two_model_path) as recipe_file:
            recipe = yaml.load(recipe_file, Loader=yaml.FullLoader)
        # TODO END
    with r1_col3:
            if st.button("Generate new lrf with Recipe"):
                request = api_helper.request_recipe_lrf(output_dir=rv_output_dir, recipe=recipe, lot_id="")

                if request.json().get("status") == "error":
                    code = request.json().get("code")
                    message = request.json().get("message")
                    st.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                    logger.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                else:
                    st.success(f"New .lrf file using recipe generated at {rv_output_dir}!")
                    logger.info(f"New .lrf file using recipe generated at {rv_output_dir}!")

    if recipe is not None:
        with st.expander("Recipe preview:"):
            st.code(yaml.dump(recipe), language="yaml")
        st.divider()

    db_files = glob.glob(f"{rv_output_dir}/*.db")
    if len(db_files) > 1:
        multi_lot_result_viewer_v7.app(output_dir_default, rv_output_dir)
    else:
        single_lot_result_viewer_v7.app(output_dir_default, rv_output_dir, recipe=recipe)

    invalid_input = [output_dir_default, ""]
