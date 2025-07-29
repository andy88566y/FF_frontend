import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

COLORS = ("#0072B2", "#D55E00", "#009E73")
##################################################################
#                                                                #
# Shared Components                                              #
#                                                                #
# This file stores components or charts                          #
# That will be used in multiple viewers or components.           #
#                                                                #
##################################################################


def bin_key_to_set(bin_key: str, model_names: list[dict]):
    return set([model_names[i]["model_hash"] for i, b in enumerate(bin_key) if b == "1"])


def get_venn_fig(sets, labels, title, ax):
    ax.clear()
    venn3(sets, set_labels=labels, set_colors=COLORS, ax=ax)
    ax.set_title(title)
    ax.set_axis_off()


def gen(
    aggregated_model_data: tuple[list[str], list[float], list[int], list[str]],
    intersections: tuple[list[list[int]]],
    model_names: list[str],
    split_lot: bool,
) -> tuple:
    defect_ids, prob_lists, ans, lot_ids = aggregated_model_data
    classifications = ["Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in ans]
    tp_datas, tn_datas, _ = intersections
    tp_sets = [set(model_correction) for model_correction in tp_datas]
    tn_sets = [set(model_correction) for model_correction in tn_datas]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    get_venn_fig(tp_sets, model_names, f"TP Venn Diagram (Total: {ans.count(1)})", axs[0])
    get_venn_fig(tn_sets, model_names, f"TN Venn Diagram (Total: {ans.count(0)})", axs[1])

    # Adjust layout for better spacing
    fig.subplots_adjust(wspace=0.3)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
