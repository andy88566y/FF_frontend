import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_review_ui import defect_review_gui
from ltt_ff_frontend.defect_ui import finetuning_api, inference_api, result_viewer


if __name__ == "__main__":
    logger.debug("Loading main UI ...")

    result_viewer = st.Page(result_viewer.app, url_path="result_viewer",
                            title="Result Viewer", icon=":material/search_check_2:")
    inference_api = st.Page(inference_api.app, url_path="inference_api",
                            title="Inference", icon=":material/content_paste_search:")
    finetuning_api = st.Page(finetuning_api.app, url_path="training_api", title="Fine-tuning", icon=":material/build:")
    defect_review_gui = st.Page(defect_review_gui.app, url_path="defect_review", title="Defect Review", icon=":material/image_search:")

    pg = st.navigation([defect_review_gui, result_viewer, inference_api, finetuning_api])

    st.set_page_config(page_title="Lasertec Defect Filter UI",
                       page_icon=":material/manufacturing:",
                       layout="wide", initial_sidebar_state="collapsed")

    st.sidebar.markdown("###### FF-FE v0.2.0")

    pg.run()
