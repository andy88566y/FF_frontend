import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ltt_ff_frontend.shared_components import helper
from ltt_ff_frontend.shared_components.prob_distribution_fig import get_color_map
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.helpers.api_helper import MultiLotModelData
from loguru import logger


def generate(
    multi_lot_model_data_1: MultiLotModelData,
    multi_lot_model_data_2: MultiLotModelData,
    m1_threshold: float,
    m2_threshold: float,
) -> go.Figure:
    lot_id_lists_1 = [meta['lot_id'] for meta in multi_lot_model_data_1.model_metadata_list]
    logger.debug(lot_id_lists_1)
    lot_id_lists_2 = [meta['lot_id'] for meta in multi_lot_model_data_2.model_metadata_list]
    assert sorted(lot_id_lists_1) == sorted(lot_id_lists_2), "lot ids mismatch"

    idx = 0
    m1_defect_ids, m1_probs, m1_ans = (
        multi_lot_model_data_1.defect_id_lists[idx],
        multi_lot_model_data_1.probability_lists[idx],
        multi_lot_model_data_1.answer_lists[idx],
    )
    m2_defect_ids, m2_probs, m2_ans = (
        multi_lot_model_data_2.defect_id_lists[idx],
        multi_lot_model_data_2.probability_lists[idx],
        multi_lot_model_data_2.answer_lists[idx],
    )
    assert sorted(multi_lot_model_data_1.defect_id_lists[idx]) == sorted(multi_lot_model_data_2.defect_id_lists[idx]), "defect id lists mismatch"

    defect_ids = [f"Defect ID: {defect_id}" for defect_id in m1_defect_ids]
    classifications = [
        "Defect" if a1 == 1 and a2 == 1 else "Non-defect" if a1 == 0 and a2 == 0 else "No-Label"
        for a1, a2 in zip(m1_ans, m2_ans)
    ]
    marker_text = [f"{defect_id}<br>{classification}" for defect_id, classification in zip(defect_ids, classifications)]
    extended_lot_id_list = [lot_id_lists_1[idx]] * len(defect_ids)
    legends = [f"{classification} {lot_id}" for classification, lot_id in zip(classifications, extended_lot_id_list)]

    # logger.debug(len(m1_defect_ids))
    # logger.debug(len(m1_probs))
    # logger.debug(len(m2_probs))
    # logger.debug(len(classifications))
    # logger.debug(len(legends))
    df = pd.DataFrame(
        data={
            "Defect_ID": m1_defect_ids,
            "Probability_M1": m1_probs,
            "Probability_M2": m2_probs,
            "Classification": classifications,
            "Legends": legends,
        }
    )

    fig = px.scatter(
        df,
        x="Probability_M1",
        y="Probability_M2",
        range_x=[0.0, 1.0],
        range_y=[0.0, 1.0],
        marginal_x="histogram",
        marginal_y="histogram",
        color="Legends",
        color_discrete_map=get_color_map(df["Legends"]),
        hover_data={"Defect_ID": True},
    )

    # Workaround to set number of bins for the marginal histograms
    for _, trace in enumerate(fig.data):
        if trace.type == "histogram":
            trace.nbinsx = 100
            trace.nbinsy = 100

    # Add in threshold lines
    fig.add_shape(
        type="line",
        x0=m1_threshold,
        x1=m1_threshold,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "Red", "width": 2, "dash": "dash"},
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=m2_threshold,
        y1=m2_threshold,
        xref="paper",
        yref="y",
        line={"color": "Red", "width": 2, "dash": "dash"},
    )

    # add diagonal dotted line
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        xref="x",
        yref="y",
        line={"color": "Gray", "width": 1, "dash": "dash"},
    )

    # add performance hint (upper left: red, bottom right: green)
    fig.add_shape(type="path", path="M 0 0 L 0 1 L 1 1 Z", line_width=0, fillcolor="lightpink", opacity=0.3)
    fig.add_shape(type="path", path="M 0 0 L 1 0 L 1 1 Z", line_width=0, fillcolor="palegreen", opacity=0.3)

    fig.update_layout(
        title="Model Comparision Chart",
        xaxis={"zeroline": False, "showgrid": False, "title": "Model 1 (Base)"},
        yaxis={"zeroline": False, "showgrid": False, "title": "Model 2 (Candidate)"},
        xaxis2={"zeroline": False, "showgrid": False, "title": "Model 2 Histogram"},
        yaxis2={"zeroline": False, "showgrid": False},
        xaxis3={"zeroline": False, "showgrid": False},
        yaxis3={"zeroline": False, "showgrid": False, "title": "Model 1 Histogram"},
        height=600,
        width=800,
        bargap=0,
        barmode="stack",
        hovermode="closest",
        showlegend=True,
    )

    return fig
