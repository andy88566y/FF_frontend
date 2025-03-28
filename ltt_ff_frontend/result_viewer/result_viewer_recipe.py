import glob
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def calculate_recipe_filtered_results(output_dir: str, recipe: dict[str, Any]) -> dict[str, Any]:
    defect_id_list = helper.get_defect_id_lists(output_dir)
    answer_list = helper.get_answer(output_dir=output_dir, defect_id=defect_id_list)[0]
    prediction_list = helper.get_predictions(output_dir=output_dir, recipe=recipe, defect_list=defect_id_list)[0]

    positive = answer_list.count(1)
    negative = answer_list.count(0)
    unlabeled = answer_list.count(-1)
    true_positive = sum(1 for pred, ans in zip(prediction_list, answer_list) if pred == 1 and ans == 1)
    false_positive = sum(1 for pred, ans in zip(prediction_list, answer_list) if pred == 1 and ans == 0)
    true_negative = sum(1 for pred, ans in zip(prediction_list, answer_list) if pred == 0 and ans == 0)
    false_negative = sum(1 for pred, ans in zip(prediction_list, answer_list) if pred == 0 and ans == 1)
    filtered_unlabeled_defect_count = sum(
        1 for pred, ans in zip(prediction_list, answer_list) if pred == 1 and ans == -1
    )

    as_is_defect_count = positive + negative + unlabeled
    to_be_defect_count = true_positive + false_positive + filtered_unlabeled_defect_count

    capture_rate = true_positive / positive if positive > 0 else -1
    false_filter_rate = true_negative / negative if negative > 0 else -1
    filter_rate = 1 - (to_be_defect_count / as_is_defect_count) if as_is_defect_count > 0 else -1

    return {
        "as_is_defect_count": as_is_defect_count,
        "to_be_defect_count": to_be_defect_count,
        "filter_rate": filter_rate,
        "as_is_true_defect_count": positive,
        "to_be_true_defect_count": true_positive,
        "capture_rate": capture_rate,
        "as_is_non_defect_count": negative,
        "to_be_non_defect_count": false_positive,
        "false_filter_rate": false_filter_rate,
        "unlabeled": unlabeled,
        "filtered_unlabeled_defect_count": filtered_unlabeled_defect_count,
    }


def get_classtype_count(defect_list: list[dict[str, Any]]) -> pd.DataFrame:
    classtype_counter: dict[str, int] = {}

    # Leave this as dict, move to backend in the future
    for defect in defect_list:
        defect_string = "Defect" if defect["Ans"] == 1 else "Non-defect" if defect["Ans"] == 0 else "Unlabeled"
        key = f"[{defect_string}] {defect['ClassType']}"
        classtype_counter[key] = classtype_counter.get(key, 0) + 1

    classtype_counter_list = []
    for key, value in classtype_counter.items():
        classification, classtype = key.split(" ")
        classtype_counter_list.append({"Classification": classification, "ClassType": classtype, "Count": value})

    # Convert to DF and rename columns (this will appear on streamlit DF)
    classtype_counter_df = pd.DataFrame.from_records(data=classtype_counter_list).sort_values(
        by="ClassType", ascending=True
    )

    return classtype_counter_df


def app() -> None:
    logger.debug("Loading Result Viewer...")
    st.title("False Filter Result Viewer (Recipe)")
    st.caption("Visualize False Filter Result")

    r1_col1, r1_col2 = st.columns([1, 1])

    output_dir_default = "/mnt/dbpc/xxx"
    with r1_col1:
        rv_output_dir = st.text_input("Inference (Recipe) Result Directory", value=output_dir_default)
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

    db_files = glob.glob(f"{rv_output_dir}/*.db")

    if len(db_files) > 1:
        st.error(f"Error: multiple ({len(db_files)}) .db files found in {rv_output_dir}")
        return

    if recipe_file is not None:
        st.subheader("Recipe preview:")
        recipe = yaml.load(recipe_file, Loader=yaml.Loader)
        st.json(recipe)

    st.divider()

    invalid_input = [output_dir_default, ""]

    if rv_output_dir not in invalid_input and recipe_file is not None:
        model_metadata = helper.get_db_metadata_lists(rv_output_dir)[0]

        if model_metadata is None:
            with r1_col2:
                st.error(f"Error getting result data from {rv_output_dir}")
                return

        # Show result database details
        st.subheader(f"Lot ID: {model_metadata['lot_id']}")

        # Show Total/Defect/Non-defect/unlabeled count
        st.text("Inference results")
        count_rate_data = calculate_recipe_filtered_results(rv_output_dir, recipe=recipe)
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
            defects = helper.get_lrf_data_lists(output_dir=rv_output_dir, cols=["ClassType"], include_prob=False)[0]
            classtype_counter_df = get_classtype_count(defects)
            st.caption(f"LRF type: {model_metadata['input_lrf_type']}")
            st.dataframe(data=classtype_counter_df)

    else:
        pass


def app_allow_multilot() -> None:
    pass
