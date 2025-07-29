import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ltt_ff_frontend.shared_components.upset_plot_helper import plotting
from upsetplot import UpSet, from_memberships
import matplotlib.pyplot as plt
from loguru import logger


def convert_to_set(model_names: list[str], corrections: list[list[int]], ans: list[int]) -> list[list[str]]:
    tp_model_sets = []
    tn_model_sets = []
    for correction, a in zip(corrections, ans):
        if a:
            tp_model_sets.append([model_names[i] for i, c in enumerate(correction) if c])
        else:
            tn_model_sets.append([model_names[i] for i, c in enumerate(correction) if c])
    return tp_model_sets, tn_model_sets


def get_upset_fig(data: list[list[str]], title: str, sort_by: str):
    memberships = from_memberships(data)
    # Generate and display the plot
    fig = plt.figure(figsize=(8, 6))
    UpSet(memberships, show_counts=True, subset_size="count", sort_by=sort_by, include_empty_subsets=True).plot(fig)
    fig.tight_layout()
    fig.suptitle(title)

    return plt.gcf()


def gen(
    aggregated_model_data: tuple[list[str], list[float], list[int], list[str]],
    intersections: tuple[list[list[int]]],
    model_names: list[str],
    split_lot: bool,
    sort_by: str,
) -> tuple:
    defect_ids, prob_lists, ans, lot_ids = aggregated_model_data
    classifications = ["Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in ans]
    tp_datas, tn_datas, corretions = intersections
    tp_set, tn_set = convert_to_set(model_names, corretions, ans)

    fig1 = get_upset_fig(tp_set, f"TP Upset Chart (total: {ans.count(1)})", sort_by)
    fig2 = get_upset_fig(tn_set, f"TN Upset Chart (total: {ans.count(0)})", sort_by)

    return fig1, fig2
