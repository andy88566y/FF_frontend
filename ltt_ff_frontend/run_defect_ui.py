import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import model_comparison, inference_api, finetuning_api, defect_viz


if __name__ == "__main__":
    logger.debug("Loading main UI ...")

    model_comparison = st.Page(model_comparison.app, url_path="model_comparison",
                               title="Defect Probability Visualization Dashboard", icon=":material/search_check_2:")
    inference_api = st.Page(inference_api.app, url_path="inference_api",
                            title="Inference", icon=":material/search_check_2:")
    finetuning_api = st.Page(finetuning_api.app, url_path="training_api", title="Fine-tuning", icon=":material/build:")
    defect_viz = st.Page(defect_viz.app, url_path="defect_viz",
                         title="Defect visualization", icon=":material/browse_activity:")

    pg = st.navigation([model_comparison, inference_api, finetuning_api, defect_viz])

    st.set_page_config(page_title="Lasertec Defect Filter UI",
                       page_icon=":material/manufacturing:",
                       layout="wide", initial_sidebar_state="collapsed")

    pg.run()
