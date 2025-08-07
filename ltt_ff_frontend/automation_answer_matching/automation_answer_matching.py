import json
import os

import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import INFERENCE_DEFAULT_RESULT_DIR, ResultViewerComponents
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import (
    class_type_component,
    helper,
    missed_defects_component,
    multi_lot_stats,
    new_lrf_button,
    oos_summary_component,
    particle_mode_defects_component,
    recipe_preview,
)


def app() -> None:
    logger.debug("Loading Automation Answer Matching UI...")
    st.title("Automation Answer Matching")
    st.caption("Assess unlabeled FF results by comparing with labeled LRF")

    labeled_lrf_dir_col, ff_result_dir_col = st.columns(2)

    with labeled_lrf_dir_col:
        labeled_lrf_dir = st.text_input(
            label="Directory with labeled LRF",
            value="/mnt/fs0/MLE/ff_docker_output/minye/perm/automation_answer_matching/labeled_lrf_dir",
            help="YYYY/MM/DD subdirectories will be searched.",
        )
    with ff_result_dir_col:
        ff_result_dir = st.text_input(
            label="Directory with unlabeled FF results",
            value="/mnt/fs0/MLE/ff_docker_output/minye/perm/automation_answer_matching/unlabeled_result_dir",
            help="YYYY/MM/DD subdirectories will be searched.",
        )

    # # Parse YYYY/MM/DD directories to be presented as options
    # dir_structure: dict[str, dict[str, list[str]]] = {}
    # years = os.listdir(labeled_lrf_dir)
    # for year in years:
    #     dir_structure[year] = {}
    #     months = os.listdir(os.path.join(labeled_lrf_dir, year))
    #     for month in months:
    #         days = os.listdir(os.path.join(labeled_lrf_dir, year, month))
    #         dir_structure[year][month] = days

    # logger.warning(dir_structure)

    # year_col, month_col, day_col = st.columns(3)
    # with year_col:
    #     selected_years = st.multiselect(label="Years", options=dir_structure.keys())
    # with month_col:
    #     selected_months = st.multiselect(label="Years", options=dir_structure.keys())

    ui_components = api_helper.match_automation_results(add_lrf_dir=labeled_lrf_dir, ff_result_dir=ff_result_dir)

    with st.expander(label="OOS Summary"):
        oos_summary_component.gen(oos_calculation=ui_components["oos_summary"])

    with st.expander(label="Missed defects"):
        missed_defects_component.gen(missed_defect_info=ui_components["missed_defect_info"])

    with st.expander(label="ParticleMode defects"):
        particle_mode_defects_component.gen(particle_mode_info=ui_components["particle_mode_info"])

    with st.expander(label="LRF ClassType count"):
        class_type_component.gen(classtype_count_list=ui_components["classtype_count"])

    with st.expander(label="Inference Result Table"):
        selected_lot_id_list, df = multi_lot_stats.gen(
            inference_data=ui_components["inference_result_table"], key="recipe_stats_df"
        )
