import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ltt_ff_frontend.shared_components.prob_distribution_fig import get_color_map


def gen(
    aggregated_model_data_1: tuple[list[str], list[float], list[int], list[str]],
    aggregated_model_data_2: tuple[list[str], list[float], list[int], list[str]],
    m1_threshold: float,
    m2_threshold: float,
) -> go.Figure:
    m1_defect_ids, m1_probs, m1_ans, m1_lot_ids = aggregated_model_data_1
    m2_defect_ids, m2_probs, m2_ans, m2_lot_ids = aggregated_model_data_2
    assert sorted(m1_lot_ids) == sorted(m2_lot_ids), "Lot IDs Mismatch!"
    assert sorted(m1_defect_ids) == sorted(m2_defect_ids), "Defect IDs Count Mismatch!"

    classifications = [
        "Defect" if a1 == 1 and a2 == 1 else "Non-defect" if a1 == 0 and a2 == 0 else "No-Label"
        for a1, a2 in zip(m1_ans, m2_ans)
    ]
    legends = [f"{classification} {lot_id}" for classification, lot_id in zip(classifications, m1_lot_ids)]

    df = pd.DataFrame(
        data={
            "Defect_ID": m1_defect_ids,
            "Probability_M1": m1_probs,
            "Probability_M2": m2_probs,
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
        x="Probability_M1",
        y="Probability_M2",
        range_x=[0.0, 1.0],
        range_y=[0.0, 1.0],
        marginal_x="histogram",
        marginal_y="histogram",
        color="Legends",
        color_discrete_map=get_color_map(df["Legends"]),
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
