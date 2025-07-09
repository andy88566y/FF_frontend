import json
from decimal import Decimal
from typing import Any

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


def gen(recipe_type: str, db_recipe: dict[str, Any]) -> dict:
    logger.debug("Generating recipe preview...")

    # In Yaml Mode, preview + use the uploaded recipe
    if recipe_type == YAML_MODE:
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
            else:
                return {}

    # In DB Mode, filter DB recipe before preview and use DB recipe for calculation
    elif recipe_type == DB_MODE:
        if not db_recipe:
            return {}
        recipe_to_preview = helper.filter_recipe_columns(db_recipe)
        recipe = db_recipe

    #  In Creator Mode, preview + use the recipe using selected models and thresholds
    elif recipe_type == CREATOR_MODE:
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

        user_input_recipe: dict[str, Any] = {"recipes": []}
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
    else:
        recipe, recipe_to_preview = {}, {}

    if recipe and recipe["recipes"] != []:
        with st.expander(f"{recipe_type} Recipe preview:", expanded=True):
            st.code(yaml.dump(recipe_to_preview), language="yaml")

    return recipe
