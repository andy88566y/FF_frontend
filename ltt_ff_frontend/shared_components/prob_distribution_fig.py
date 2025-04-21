from typing import Any
from ltt_ff_frontend.shared_components import helper

import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DEFECT_COLOR_MAPPING = {
    "D": "darkred",
    "ND": "olivedrab",
    "UNK": "blue",
}

def get_color_map(legends_list: list[str]) -> dict[str, Any]:
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

def generate_1D_plot(raw_data: tuple[list[int], list[float], list[int]], threshold: float) -> go.Figure:
    defect_ids, probs, ans = raw_data

    df = pd.DataFrame(data={"Defect_ID": defect_ids, "Probability": probs, "LRF_Label": ans})
    df["Classification"] = [
        "Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in df["LRF_Label"]
    ]

    # Add histogram
    fig = px.histogram(
        data_frame=df,
        x="Probability",
        range_x=[0.0, 1.0],
        nbins=100,
        color="Classification",
        color_discrete_map={
            "Non-defect": DEFECT_COLOR_MAPPING["ND"],
            "Defect": DEFECT_COLOR_MAPPING["D"],
            "Unlabeled": DEFECT_COLOR_MAPPING["UNK"],
        },
        marginal="rug",
        hover_name="Classification",
        hover_data={
            "Probability": True,
            "Defect_ID": True,
            "LRF_Label": False,
            "Classification": False,
        },
        labels={
            "LRF_Label": "Defect/non-defect",
        },
    )

    # Add threshold line
    fig.add_shape(
        type="line",
        x0=threshold,
        x1=threshold,
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


def generate_multilot_1D_plot(
    raw_data: tuple[list[int], list[float], list[int]],
    model_metadata_list: list[dict[str, Any]],
    selected_threshold: float,
    selected_lot_id_list: list[str] = []
) -> go.Figure:
    """
    Generates a 1D plot for defect probability distribution across multiple lots.

    Parameters:
    raw_data (tuple[list[int], list[float], list[int]]): 
        A tuple containing three lists:
        - list[int]: List of defect IDs.
        - list[float]: List of probabilities.
        - list[int]: List of labels (0 for non-defect, 1 for defect, other values for unlabeled).

    model_metadata_list (list[dict[str, Any]]): 
        A list of dictionaries containing metadata for each model.

    selected_threshold (float): 
        The threshold value for classification.
        Used to draw red dot line.

    selected_lot_id_list (list[str], optional): 
        A list of lot IDs to filter the data. Defaults to an empty list.

    Returns:
    go.Figure: 
        Plotly figure object with the defect probability distribution histogram.
    """

    id_list, prob_list, ans_list, lot_id_list = helper.aggregate_lists(raw_data, model_metadata_list)
    df = pd.DataFrame(
        data={"Defect_ID": id_list, "Probability": prob_list, "LRF_Label": ans_list, "Lot ID": lot_id_list}
    )
    df = df[df["Lot ID"].isin(selected_lot_id_list)] if selected_lot_id_list else df
    df["Classification"] = [
        "Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in df["LRF_Label"]
    ]
    df["Legends"] = [f"{classification} {lot_id}" for classification, lot_id in zip(df["Classification"], df["Lot ID"])]

    # Add histogram
    # hover_data defines which df columns will appear on the hover message
    # label changes the column name on the hover message
    fig = px.histogram(
        data_frame=df,
        x="Probability",
        range_x=[0.0, 1.0],
        nbins=100,
        color="Legends",
        color_discrete_map=get_color_map(df["Legends"]),
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
    )

    # Add threshold line
    fig.add_shape(
        type="line",
        x0=selected_threshold,
        x1=selected_threshold,
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