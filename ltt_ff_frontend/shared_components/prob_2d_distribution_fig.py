from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ltt_ff_frontend.shared_components.prob_distribution_fig import get_classtype_color_map


def gen(
    aggregated_model_data: list[tuple[list[str], list[dict[str, Any]], list[list[float]], list[int], list[str]]],
    model_1: dict[str, Any],
    model_2: dict[str, Any],
    split_lot: bool,
    mode: str,
) -> go.Figure:
    m1_name = model_1["model_hash"]
    m2_name = model_2["model_hash"]
    if len(aggregated_model_data) == 1:
        defect_ids, defect_infos, model_probs, ans, lot_ids = aggregated_model_data[0]
        classifications = ["Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in ans]
        m1_probs = np.array(model_probs)[:, 0]
        m2_probs = np.array(model_probs)[:, 1]
    else:
        defect_ids, defect_infos, m1_probs, m1_ans, lot_ids = aggregated_model_data[0]
        m2_defect_ids, _, m2_probs, m2_ans, m2_lot_ids = aggregated_model_data[1]
        m1_probs = np.array(m1_probs)[:, 0]
        m2_probs = np.array(m2_probs)[:, 0]
        assert sorted(lot_ids) == sorted(m2_lot_ids), "Lot IDs Mismatch!"
        assert sorted(defect_ids) == sorted(m2_defect_ids), "Defect IDs Count Mismatch!"

        classifications = [
            "Defect" if (a1 * a2) == 1 else "Non-defect" if (a1 + a2) == 0 else "No-Label"
            for a1, a2 in zip(m1_ans, m2_ans)
        ]
    classtypes = [str(defect_info["ClassType"]) for defect_info in defect_infos]
    if split_lot:
        legends = [f"{classification} {lot_id}" for classification, lot_id in zip(classifications, lot_ids)]
    else:
        legends = classifications
    df = pd.DataFrame(
        data={
            "Defect_ID": defect_ids,
            f"Probability_{m1_name}": m1_probs,
            f"Probability_{m2_name}": m2_probs,
            "Classtype": classtypes,
            "Classification": classifications,
            "Legends": legends,
        }
    )

    available_symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "star",
    ]
    symbols = available_symbols[: len(df["Legends"])]
    fig = px.scatter(
        df,
        x=f"Probability_{m1_name}",
        y=f"Probability_{m2_name}",
        range_x=[0.0, 1.0],
        range_y=[0.0, 1.0],
        marginal_x="histogram",
        marginal_y="histogram",
        color="Classtype",
        color_discrete_map=get_classtype_color_map(df["Classtype"]),
        hover_data={"Defect_ID": True},
        symbol="Legends",
        symbol_sequence=symbols,
    )

    # Workaround to set number of bins for the marginal histograms
    for _, trace in enumerate(fig.data):
        if trace.type == "histogram":
            trace.nbinsx = 100
            trace.nbinsy = 100

    # Add in threshold lines
    fig.add_shape(
        type="line",
        x0=model_1["threshold"],
        x1=model_1["threshold"],
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "Red", "width": 2, "dash": "dash"},
    )

    fig.add_shape(
        type="line",
        x0=model_1["threshold_c"],
        x1=model_1["threshold_c"],
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "Red", "width": 2, "dash": "longdash"},
    )

    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=model_2["threshold"],
        y1=model_2["threshold"],
        xref="paper",
        yref="y",
        line={"color": "Red", "width": 2, "dash": "dash"},
    )

    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=model_2["threshold_c"],
        y1=model_2["threshold_c"],
        xref="paper",
        yref="y",
        line={"color": "Red", "width": 2, "dash": "longdash"},
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
    if mode == "Finetuned":
        # add performance hint (upper left: red, bottom right: green)
        fig.add_shape(type="path", path="M 0 0 L 0 1 L 1 1 Z", line_width=0, fillcolor="lightpink", opacity=0.3)
        fig.add_shape(type="path", path="M 0 0 L 1 0 L 1 1 Z", line_width=0, fillcolor="palegreen", opacity=0.3)
    else:
        fig.add_shape(
            type="rect",
            x0=model_1["threshold_c"],
            x1=1,
            y0=0
            if model_1["model_hash"].startswith("RULE")
            else max(0, model_2["threshold"]),  # if model 1 is rule model, force the result to be true defect
            y1=1,
            xref="x",
            yref="y",
            fillcolor="lightpink",
            opacity=0.3,
            line_width=0,
        )
        fig.add_shape(
            type="rect",
            x0=0
            if model_2["model_hash"].startswith("RULE")
            else max(0, model_1["threshold"]),  # if model 2 is rule model, force the result to be true defect
            x1=model_1["threshold_c"],
            y0=model_2["threshold_c"],
            y1=1,
            xref="x",
            yref="y",
            fillcolor="lightpink",
            opacity=0.3,
            line_width=0,
        )
        fig.add_shape(
            type="rect",
            x0=max(0, model_1["threshold"]),
            x1=model_1["threshold_c"],
            y0=max(0, model_2["threshold"]),
            y1=model_2["threshold_c"],
            xref="x",
            yref="y",
            fillcolor="palegreen",
            opacity=0.3,
            line_width=0,
        )
        fig.add_shape(
            type="rect",
            x0=0,
            x1=model_1["threshold"],
            y0=0,
            y1=model_2["threshold_c"] if model_2["model_hash"].startswith("RULE") else 1,
            xref="x",
            yref="y",
            fillcolor="palegreen",
            opacity=0.3,
            line_width=0,
        )
        fig.add_shape(
            type="rect",
            x0=model_1["threshold"],
            x1=model_1["threshold_c"] if model_1["model_hash"].startswith("RULE") else 1,
            y0=0,
            y1=model_2["threshold"],
            xref="x",
            yref="y",
            fillcolor="palegreen",
            opacity=0.3,
            line_width=0,
        )

    fig.update_layout(
        title=f"{m1_name} & {m2_name} Comparision Chart",
        xaxis={"zeroline": False, "showgrid": False, "title": f"{m1_name} (Base)"},
        yaxis={"zeroline": False, "showgrid": False, "title": f"{m2_name} (Candidate)"},
        xaxis2={"zeroline": False, "showgrid": False, "title": f"{m2_name}  Histogram"},
        yaxis2={"zeroline": False, "showgrid": False},
        xaxis3={"zeroline": False, "showgrid": False},
        yaxis3={"zeroline": False, "showgrid": False, "title": f"{m1_name} Histogram"},
        height=800,
        width=800,
        bargap=0,
        barmode="stack",
        hovermode="closest",
        showlegend=True,
    )

    return fig
