from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ltt_ff_frontend.constant import BLANK_MODEL
from ltt_ff_frontend.helpers import api_helper

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

    # Convert to DF and sort by classtype
    classtype_counter_df = pd.DataFrame.from_records(data=classtype_counter_list).sort_values(
        by="ClassType", ascending=True, key=lambda classtype: classtype.astype(int)
    )

    # Reset index after sorting, and let it start from 1 instead of 0
    classtype_counter_df.reset_index(inplace=True, drop=True)
    classtype_counter_df.index = range(1, len(classtype_counter_df) + 1)

    return classtype_counter_df


def gen(inference_result_dir, model_metadata_list) -> None:
    defect_lists = api_helper.get_lrf_data_lists(
        output_dir=inference_result_dir, cols=["ClassType"], include_prob=False
    )
    for defect_list, meta in zip(defect_lists, model_metadata_list):
        classtype_counter_df = get_classtype_count(defect_list)
        st.text(f"Lot ID: {meta['lot_id']}")
        st.caption(f"LRF type: {meta['input_lrf_type']}")
        st.dataframe(data=classtype_counter_df)
    st.divider()