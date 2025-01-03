import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_viz, inference_api, defect_dshbrd


if __name__ == "__main__":
    logger.debug("Loading main UI ...")

    defect_viz = st.Page(defect_viz.app, url_path="defect_viz",
                         title="Defect visualization", icon=":material/browse_activity:")
    inference_api = st.Page(inference_api.app, url_path="inference_api",
                            title="Inference", icon=":material/search_check_2:")
    defect_dshbrd = st.Page(defect_dshbrd.app, url_path="defect_dshbrd", 
                        title="Defect Probability Visualization Dashboard", icon=":material/search_check_2:")

    pg = st.navigation([defect_viz, inference_api, defect_dshbrd])

    st.set_page_config(page_title="Lasertec Defect Filter UI", page_icon=":material/manufacturing:")

    pg.run()
