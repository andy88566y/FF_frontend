import streamlit as st
import yaml

from ltt_ff_frontend.regression_test_ui.help_text import RECIPE_HELP_TEXT
from ltt_ff_frontend.regression_test_ui.test_utils import Regression_Recipe
from ltt_ff_frontend.shared_components.helper import is_valid_output_dir


def app() -> None:
    st.title("Regression Recipe Converter")
    st.caption("Convert regression recipe config to multiple inference recipes")
    recipe_file = st.file_uploader("Upload Regression Recipe Config (.yaml)", help=RECIPE_HELP_TEXT)
    if recipe_file:
        recipe_config = yaml.safe_load(recipe_file)
        try:
            regression_recipe = Regression_Recipe(recipe_config)
        except ValueError as e:
            st.error(f"Invalid Regression Recipe: {e}")
            return
    with st.expander("Regression Recipe Config", expanded=False):
        st.json(regression_recipe.recipe)
    recipe_output_dir = st.text_input("Output Directory")

    if st.button("Convert", type="primary"):
        if not is_valid_output_dir(recipe_output_dir):
            st.error("Please enter a valid output directory.")
        try:
            regression_recipe.make_recipe_dir(recipe_output_dir)
            st.success("Recipe Convert Completed")
        except:
            st.error("Recipe Convert Failed")
