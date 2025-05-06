from typing import Any

import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


def recipe_model_name_matches(new_recipe: dict[str, Any], db_recipe: dict[str, Any]):
    new_recipe_models = [r["model_name"] for r in new_recipe["recipes"]]
    db_recipe_models = [r["model_name"] for r in db_recipe["recipes"]]
    return new_recipe_models == db_recipe_models

def gen(container: st.container, inference_result_dir: str, new_recipe: dict[str, Any], db_recipe: dict[str, Any]) -> None:
    with container:
        if recipe_model_name_matches(new_recipe, db_recipe):
            if st.button("Generate new lrf with Recipe"):
                request = api_helper.request_recipe_lrf(output_dir=inference_result_dir, recipe=new_recipe, lot_id="")
                if request.json().get("status") == "error":
                    code = request.json().get("code")
                    message = request.json().get("message")
                    st.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                    logger.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                else:
                    st.success(f"New .lrf file using recipe generated at {inference_result_dir}!")
                    logger.info(f"New .lrf file using recipe generated at {inference_result_dir}!")
        else:
            st.error("recipe models name does not match db's recipe.")
            logger.error(f"new_recipe: {new_recipe}. db_recipe: {db_recipe}")
