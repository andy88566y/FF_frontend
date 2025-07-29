import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ltt_ff_frontend.shared_components.upset_plot_helper import plotting
from upsetplot import UpSet, from_memberships
import matplotlib.pyplot as plt
import streamlit as st
from loguru import logger

def convert_to_set(model_names: list[str], corrections: list[list[int]], ans: list[int]) -> list[list[str]]:
    tp_model_sets = []
    tn_model_sets = []
    for correction, a in zip(corrections, ans):
        if a == 1:
            tp_model_sets.append([model_names[i] for i, c in enumerate(correction) if c])
        elif a == 0:
            tn_model_sets.append([model_names[i] for i, c in enumerate(correction) if c])
    return tp_model_sets, tn_model_sets


def get_upset_fig(data: list[list[str]], title: str, sort_by: str, exclude_zero: bool):
    memberships = from_memberships(data)
    # Generate and display the plot
    fig = plt.figure(figsize=(8, 6))
    UpSet(
        memberships, 
        show_counts=True, 
        subset_size="count", 
        sort_by=sort_by, 
        include_empty_subsets= not exclude_zero
    ).plot(fig)
    fig.tight_layout()
    fig.suptitle(title)

    return plt.gcf()


def gen(
    aggregated_model_data: tuple[list[str], list[float], list[int], list[str]],
    intersections: tuple[list[list[int]]],
    model_names: list[str],
    split_lot: bool,
    sort_by: str,
    exclude_zero: bool,
) -> tuple:
    defect_ids, prob_lists, ans, lot_ids = aggregated_model_data
    classifications = ["Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in ans]
    tp_datas, tn_datas, corretions = intersections
    tp_set, tn_set = convert_to_set(model_names, corretions, ans)
    tp_col, _, tn_col = st.columns([4.5, 1, 4.5])
    with tp_col:
        if any(len(tp) != 0 for tp in tp_set):
            st.pyplot(get_upset_fig(
                tp_set, 
                f"True Defects Upset Chart (total: {ans.count(1)})", 
                sort_by, 
                exclude_zero))
        else:
            st.markdown("No True Defects Were Predicted Correct!")
    with tn_col:
        if any(len(tn) != 0 for tn in tn_set):
            st.pyplot(
                get_upset_fig(
                    tn_set, 
                    f"None Defects Upset Chart (total: {ans.count(0)})", 
                    sort_by, 
                    exclude_zero))
        else:
            st.markdown("No None Defects Were Predicted Correct!")
    

