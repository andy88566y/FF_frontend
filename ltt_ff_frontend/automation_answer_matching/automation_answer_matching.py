import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import LRFType
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import (
    class_type_component,
    missed_defects_component,
    multi_lot_stats,
    oos_summary_component,
    particle_mode_defects_component,
)


def app() -> None:
    logger.debug("Loading Automation Answer Matching UI...")
    st.title("Automation Answer Matching")
    st.caption("Assess unlabeled FF results by comparing with labeled LRF")

    labeled_lrf_dir_col, ff_result_dir_col = st.columns(2)

    with labeled_lrf_dir_col:
        labeled_lrf_dir = st.text_input(
            label="Directory with labeled LRF",
            value="",
            help="YYYY/MM/DD subdirectories will be searched.",
        )
    with ff_result_dir_col:
        ff_result_dir = st.text_input(
            label="Directory with unlabeled FF results",
            value="",
            help="YYYY/MM/DD subdirectories will be searched.",
        )

    labeled_lrf_type_col, _ = st.columns(2)
    with labeled_lrf_type_col:
        labeled_lrf_type = st.selectbox(label="Select labeled LRF type", options=[lrftype.value for lrftype in LRFType])

    date_range = st.date_input(label="Select period (start & end dates are inclusive)", value=())

    if not labeled_lrf_dir or not ff_result_dir or not date_range or len(date_range) < 2:
        return

    st.divider()

    ui_components = api_helper.match_automation_results(
        labeled_lrf_dir=labeled_lrf_dir,
        ff_result_dir=ff_result_dir,
        labeled_lrf_type=labeled_lrf_type,
        date_range=date_range,
    )

    with st.expander(label="Incomplete lots"):
        st.dataframe(ui_components["incomplete_lots"])

    with st.expander(label="OOS Summary"):
        oos_summary_component.gen(oos_calculation=ui_components["oos_summary"])

    with st.expander(label="Missed defects"):
        missed_defects_component.gen(missed_defect_info=ui_components["missed_defect_info"])

    with st.expander(label="ParticleMode defects"):
        particle_mode_defects_component.gen(particle_mode_info=ui_components["particle_mode_info"])

    with st.expander(label="LRF ClassType count"):
        class_type_component.gen(classtype_count_list=ui_components["classtype_count"])

    with st.expander(label="Inference Result Table"):
        _, _ = multi_lot_stats.gen(inference_data=ui_components["inference_result_table"], key="recipe_stats_df")
