import glob
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.helpers import api_helper, ui_helper
from ltt_ff_frontend.result_viewer import single_lot_result_viewer_v7, multi_lot_result_viewer_v7

def app() -> None:
    logger.debug("Loading V7 Result Viewer...")
    st.title("False Filter Result Viewer (Recipe)")
    st.caption("Visualize False Filter Result")

    gen_lrf_type_button_container, _ = st.columns([1, 1])
    r1_col1, r1_col2, r1_col3 = st.columns([3, 3, 1])
    # TODO: Switch to list
    rc1_col1, rc1_col2, rc1_col3 = st.columns([3, 2, 2])
    rc2_col1, rc2_col2, rc2_col3 = st.columns([3, 2, 2])
    rc3_col1, rc3_col2, rc3_col3 = st.columns([3, 2, 2])
    rc4_col1, rc4_col2, rc4_col3 = st.columns([3, 2, 2])
    rc5_col1, rc5_col2, rc5_col3 = st.columns([3, 2, 2])

    output_dir_default = "/mnt/dbpc/xxx"
    single_lot_single_model_dir = '/home/ronyauw/0411/single_lot_single_model/'
    multi_lot_single_model_dir = '/home/ronyauw/0411/multi_lot_single_model/'
    st_recipe_type = 'yaml'
    recipe = None
    with r1_col1:
        rv_output_dir = st.text_input("Inference Result Directory", value=multi_lot_single_model_dir)
    with r1_col2:
        yaml_help_text = """
        **Example of a valid recipe:**\n
        recipes:\n
        \- model_name: base/model_1.encrypted.pth\n
        &nbsp;&nbsp;threshold: 0.5\n
        \- model_name: base/model_2.encrypted.pth\n
        &nbsp;&nbsp;threshold: 0.9\n
        """
        recipe_file = st.file_uploader("Upload Recipe (.yaml)", type=".yaml", help=yaml_help_text)
        if recipe_file is not None:
            recipe = yaml.load(recipe_file, Loader=yaml.Loader)

    # with gen_lrf_type_button_container:
    #     st_gen_lrf_type = st.segmented_control("lrf Generation Option", ["threshold", "top_k"], default="threshold")
    # TODO: develop use, remove this when merge request is ready
    st_gen_lrf_type = "threshold"
    if st_gen_lrf_type is None:
        st.error("lrf Generation Option can not be None!")
        return

    db_files = glob.glob(f"{rv_output_dir}/*.db")
    if len(db_files) > 1:
        multi_lot_result_viewer_v7.app(output_dir_default, rv_output_dir, st_gen_lrf_type)
    else:
        single_lot_result_viewer_v7.app(output_dir_default, rv_output_dir, st_gen_lrf_type)
    st.divider()

    if recipe is not None:
        st.subheader("Recipe preview:")
        st.code(yaml.dump(recipe), language="yaml")
        st.divider()

    invalid_input = [output_dir_default, ""]

    if rv_output_dir not in invalid_input and recipe is not None:
        model_metadata_list = api_helper.get_db_metadata_lists(rv_output_dir)[0]
        if model_metadata_list is None:
            with r1_col3:
                st.error(f"Error getting result data from {rv_output_dir}")
                return

        with r1_col3:
            if st.button("Generate new lrf with Recipe"):
                request = api_helper.request_recipe_lrf(output_dir=rv_output_dir, recipe=recipe, lot_id="")

                if request.json().get("status") == "error":
                    code = request.json().get("code")
                    message = request.json().get("message")
                    st.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                    logger.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                else:
                    st.success(f"New .lrf file using recipe generated at {rv_output_dir}!")
                    logger.info(f"New .lrf file using recipe generated at {rv_output_dir}!")

        # Show result database details
        st.subheader(f"Lot ID: {model_metadata_list['lot_id']}")

        # Show Total/Defect/Non-defect/unlabeled count
        st.text("Inference results")
        count_rate_data = api_helper.calculate_recipe_filtered_results(rv_output_dir, recipe=recipe)
        r3_col1, r3_col2, r3_col3, r3_col4 = st.columns([4, 3, 3, 2])
        with r3_col1:
            st.warning(f"""**Total defect count**: As-is: {count_rate_data['as_is_defect_count']}
                    → To-be: {count_rate_data['to_be_defect_count']}
                    (Filter Rate: {count_rate_data['filter_rate']:.4f})""")
        with r3_col2:
            st.error(f"""**True defect count**: {count_rate_data['as_is_true_defect_count']}
                    → {count_rate_data['to_be_true_defect_count']}
                    (Capture Rate: {count_rate_data['capture_rate']:.4f})""")
        with r3_col3:
            st.success(f"""**Non-defect count**: {count_rate_data['as_is_non_defect_count']}
                    → {count_rate_data['to_be_non_defect_count']}
                    (False Filter Rate: {count_rate_data['false_filter_rate']:.4f})""")
        with r3_col4:
            st.info(f"""**Unlabeled count**: {count_rate_data['unlabeled']}
                    → {count_rate_data['filtered_unlabeled_defect_count']}""")

        # TODO: Get classtype grouping from backend
        with st.expander(label="LRF ClassType count"):
            defects = api_helper.get_lrf_data_lists(output_dir=rv_output_dir, cols=["ClassType"], include_prob=False)[0]
            classtype_counter_df = ui_helper.get_classtype_count(defects)
            st.caption(f"LRF type: {model_metadata_list['input_lrf_type']}")
            st.dataframe(data=classtype_counter_df)
    else:
        pass