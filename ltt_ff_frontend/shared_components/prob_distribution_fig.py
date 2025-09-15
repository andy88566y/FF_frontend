import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DEFECT_COLOR_MAPPING = {
    "D": "darkred",
    "ND": "olivedrab",
    "UNK": "blue",
}

TOTAL_CLASSTYPES = 32
CLASSTYPE_COLOR_MAPPING = px.colors.sample_colorscale(
    "rainbow", [(i + 1) / TOTAL_CLASSTYPES for i in range(TOTAL_CLASSTYPES - 2, -2, -1)]
)


def get_legend_color_map(legends_list: list[str]) -> dict[str, Any]:
    unique_legends = set(legends_list)
    color_map = {}

    for legend in unique_legends:
        if re.search("Non-defect", legend) is not None:
            color_map[legend] = DEFECT_COLOR_MAPPING["ND"]
        elif re.search("Unlabeled", legend) is not None:
            color_map[legend] = DEFECT_COLOR_MAPPING["UNK"]
        else:
            color_map[legend] = DEFECT_COLOR_MAPPING["D"]

    return color_map


def get_classtype_color_map(classtypes: list[str]) -> dict[str, str]:
    def extract_ct_number(unique_ct: str) -> int:
        return int(unique_ct.replace("-Non-defect", "").replace("-Defect", "").replace("-Unlabeled", ""))

    unique_classes = sorted(set(classtypes))
    return {cat: CLASSTYPE_COLOR_MAPPING[extract_ct_number(cat) + 1] for cat in unique_classes}


def gen(
    aggregated_lists: tuple[list[str], list[dict[str, Any]], list[list[float]], list[int], list[str]],
    selected_model: dict[str, Any],
    split_lot: bool,
) -> go.Figure:
    """
    Generates a 1D plot for defect probability distribution across multiple lots.

    Parameters:
    aggregated_lists:
        A tuple containing four lists of lists:
        - list[list[str]]: List of defect No or UniqueID.
        - list[list[float]]: List of probabilities per models.
        - list[list[int]]: List of labels (0 for non-defect, 1 for defect, other values for unlabeled).
        - list[str]: List of Lot IDs

    selected_threshold (float):
        The threshold value for classification.
        Used to draw red dot line.

    Returns:
    go.Figure:
        Plotly figure object with the defect probability distribution histogram.
    """

    id_list, defect_infos, prob_list, ans_list, lot_id_list = aggregated_lists
    classtypes = [str(defect_info["ClassType"]) for defect_info in defect_infos]
    df = pd.DataFrame(
        data={
            "Defect_ID": id_list,
            "Probability": np.array(prob_list)[:, 0],
            "LRF_Label": ans_list,
            "Lot ID": lot_id_list,
            "ClassType": classtypes,
        }
    )

    df["Classification"] = [
        "Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in df["LRF_Label"]
    ]

    histogram_params = {}
    if split_lot:
        df["Legends"] = [
            f"{classification} {lot_id}" for classification, lot_id in zip(df["Classification"], df["Lot ID"])
        ]
        histogram_params["pattern_shape"] = "Lot ID"
        histogram_params["pattern_shape_sequence"]=["", "/", "x", "+", "|", "-", "."]
    else:
        df["Legends"] = df["Classification"]

    df["ClassType"] = df["ClassType"] + "-" + df["Classification"]
    # Add histogram
    # hover_data defines which df columns will appear on the hover message
    # label changes the column name on the hover message

    fig = px.histogram(
        data_frame=df,
        x="Probability",
        range_x=[0.0, 1.0],
        nbins=100,
        color="ClassType",
        color_discrete_map=get_classtype_color_map(df["ClassType"]),
        marginal="rug",
        hover_name="Classification",
        hover_data={
            "Probability": True,
            "Defect_ID": True,
            "LRF_Label": False,
            "Classification": False,
            "Legends": False,
            "Lot ID": True,
        },
        labels={
            "LRF_Label": "Defect/non-defect",
        },
        **histogram_params,
    )

    # Add threshold line
    fig.add_shape(
        type="line",
        x0=selected_model["threshold"],
        x1=selected_model["threshold"],
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "Red", "width": 2, "dash": "dash"},
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Probabilities",
        yaxis_title="Frequency",
        title="Defect Probability Distribution",
    )

    return fig
