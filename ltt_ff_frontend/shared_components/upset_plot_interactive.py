import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ltt_ff_frontend.shared_components.upset_plot_helper import plotting


def gen(
    aggregated_model_data: tuple[list[str], list[float], list[int], list[str]],
    intersections: tuple[list[list[int]]],
    model_names: list[str],
    split_lot: bool,
) -> tuple:
    defect_ids, prob_lists, ans, lot_ids = aggregated_model_data
    classifications = ["Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in ans]
    tp_datas, tn_datas, corretions = intersections

    metadata_df = pd.DataFrame(
        {
            "classification": classifications,
        }
    )
    df = pd.DataFrame(corretions, columns=model_names)
    df = pd.concat([metadata_df, df], axis=1)

    tp_df = df[df["classification"] == "Defect"]
    tn_df = df[df["classification"] == "Non-defect"]
    tp_df.pop("classification")
    tn_df.pop("classification")

    fig = plotting.plot_upset(
        dataframes=[tp_df],
        exclude_zeros=True,
        legendgroups=["test"],
        sorted_x="d",
        sorted_y="a",
        column_widths=[0.2, 0.8],
        horizontal_spacing=0.21,
        marker_size=10,
    )

    fig.update_layout(
        title=f"TP Upset Chart",
        width=800,
        font_family="Jetbrains Mono",
    )

    fig2 = plotting.plot_upset(
        dataframes=[tn_df],
        exclude_zeros=True,
        legendgroups=["test"],
        sorted_x="d",
        sorted_y="a",
        column_widths=[0.2, 0.8],
        horizontal_spacing=0.21,
        marker_size=10,
    )

    fig2.update_layout(
        title=f"TN Upset Chart",
        width=800,
        font_family="Jetbrains Mono",
    )

    return fig, fig2
