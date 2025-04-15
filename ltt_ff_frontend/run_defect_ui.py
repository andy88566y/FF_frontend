import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_review_ui import defect_review_gui
from ltt_ff_frontend.defect_ui import (
    finetuning_api,
    multilot_inference_recipe_api,
)
from ltt_ff_frontend.result_viewer import result_viewer, result_viewer_recipe


if __name__ == "__main__":
    logger.debug("Loading main UI ...")

    page_result_viewer = st.Page(
        result_viewer.app, url_path="result_viewer", title="Result Viewer", icon=":material/search_check_2:"
    )
    page_result_viewer_recipe = st.Page(
        result_viewer_recipe.app,
        url_path="result_viewer_recipe",
        title="Result Viewer Recipe",
        icon=":material/search_check_2:",
    )
    page_multilot_inference_recipe_api = st.Page(
        multilot_inference_recipe_api.app,
        url_path="multilot_inference_recipe_api",
        title="Multilot Inference Recipe",
        icon=":material/action_key:",
    )
    page_defect_review_gui = st.Page(
        defect_review_gui.app, url_path="defect_review", title="Defect Review", icon=":material/image_search:"
    )
    page_finetuning_api = st.Page(
        finetuning_api.app, url_path="training_api", title="Fine-tuning", icon=":material/build:"
    )

    pg = st.navigation(
        [
            page_result_viewer,
            page_result_viewer_recipe,
            page_multilot_inference_recipe_api,
            page_defect_review_gui,
            page_finetuning_api,
        ]
    )

    st.set_page_config(
        page_title="Lasertec Defect Filter UI",
        page_icon=":material/manufacturing:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.sidebar.markdown("###### FF-FE v0.7.0")

    pg.run()
