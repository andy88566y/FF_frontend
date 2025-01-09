import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_viz, defect_dashboard, inference_api, finetuning_api


if __name__ == "__main__":
    logger.debug("Loading main UI ...")

    inference_api = st.Page(inference_api.app, url_path="inference_api",
                            title="Inference", icon=":material/search_check_2:")
    finetuning_api = st.Page(finetuning_api.app, url_path="training_api", title="Fine-tuning", icon=":material/build:")
    defect_dashboard = st.Page(defect_dashboard.app, url_path="defect_dashboard",
                        title="Defect Probability Visualization Dashboard", icon=":material/search_check_2:")
    defect_viz = st.Page(defect_viz.app, url_path="defect_viz",
                         title="Defect visualization", icon=":material/browse_activity:")

    pg = st.navigation([defect_dashboard, inference_api, finetuning_api, defect_viz])

    st.set_page_config(page_title="Lasertec Defect Filter UI", page_icon=":material/manufacturing:")

    pg.run()
