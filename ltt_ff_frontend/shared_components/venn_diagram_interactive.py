import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import plotly.io as pio

COLORS = ("rgba(228,26,28,0.4)", "rgba(55,126,184,0.4)", "rgba(77,175,74,0.4)")
CIRCLES = [{"x": 0.4, "y": 0.5, "r": 0.3}, {"x": 0.6, "y": 0.5, "r": 0.3}, {"x": 0.5, "y": 0.7, "r": 0.3}]


def make_circles(model_names: list[dict]):
    shapes = []
    annotations = []
    for circle, color, model_name in zip(CIRCLES, COLORS, model_names):
        shapes.append(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=circle["x"] - circle["r"],
                y0=circle["y"] - circle["r"],
                x1=circle["x"] + circle["r"],
                y1=circle["y"] + circle["r"],
                fillcolor=color,
                line_color=color,
            )
        )
        annotations.append(
            dict(
                x=circle["x"],
                y=circle["y"] + circle["r"] + 0.05,
                text=model_name,
                showarrow=False,
                font=dict(size=20, color=color),
            )
        )
    return shapes, annotations


def bin_key_to_set(bin_key: str, model_names: list[dict]):
    return set([model_names[i]["model_hash"] for i, b in enumerate(bin_key) if b == "1"])


def get_venn_fig(sets, labels, title):
    fig, ax = plt.subplots()
    venn3(sets, set_labels=labels, set_colors=COLORS, ax=ax)
    ax.set_title(title)
    return fig


def gen(
    aggregated_model_data: tuple[list[str], list[float], list[int], list[str]],
    intersections: tuple[list[list[int]]],
    model_names: list[str],
    split_lot: bool,
) -> tuple:
    defect_ids, prob_lists, ans, lot_ids = aggregated_model_data
    classifications = ["Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in ans]
    tp_datas, tn_datas, bin_keys = intersections
    if split_lot:
        legends = [f"{classification} {lot_id}" for classification, lot_id in zip(classifications, lot_ids)]
    else:
        legends = classifications
    df = pd.DataFrame(
        data={
            "Defect_ID": defect_ids,
            "Probabilities": prob_lists,
            "Classification": classifications,
            "Legends": legends,
        }
    )
    tp_sets = [set(model_correction) for model_correction in tp_datas]
    tn_Sets = [set(model_correction) for model_correction in tn_datas]

    shapes, annotations = make_circles(model_names)
    fig = go.Figure()
    fig.update_layout(
        title=f"TP Venn Diagram",
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=800,
        width=800,
        bargap=0,
        barmode="stack",
        hovermode="closest",
        showlegend=True,
    )

    fig2 = go.Figure()
    fig2.update_layout(
        title=f"TN Venn Diagram",
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=800,
        width=800,
        bargap=0,
        barmode="stack",
        hovermode="closest",
        showlegend=True,
    )
    return fig, fig2
