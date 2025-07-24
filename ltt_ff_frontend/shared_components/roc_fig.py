from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from loguru import logger


def gen(
    model: dict[str, Any],
    roc_data: dict[str, Any],
    split_lot: bool,
) -> None:
    # TODO: This should be done somewhere else
    if 1 not in set(roc_data["aggregated_answer_list"]):
        # All data is unlabeled or dataset consists of only non-defects
        st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")
    else:
        if split_lot:
            st.plotly_chart(
                plot_multilot_roc(
                    model["model_hash"],
                    model["threshold"],
                    roc_data["curve_data_list"],
                    roc_data["lot_id_list"],
                    roc_data["model_default_threshold_list"],
                    roc_data["selected_threshold_coord_list"],
                    roc_data["inference_threshold_coord_list"],
                )
            )
        else:
            st.plotly_chart(
                plot_aggregate_roc(
                    model["model_hash"],
                    model["threshold"],
                    roc_data["aggregate_roc_data"],
                    roc_data["model_default_threshold_list"][0],
                    roc_data["aggregate_selected_threshold_coord"],
                    roc_data["aggregate_inference_threshold_coord"],
                )
            )


def plot_aggregate_roc(
    model_name: str,
    selected_threshold: float,
    curve_data: tuple[list[float], list[float], list[float]],
    model_default_threshold: float,
    selected_threshold_coord: tuple[float, float],
    inference_threshold_coord: tuple[float, float],
) -> go.Figure:
    fig = go.Figure()
    fpr = np.array(curve_data[0])
    tpr = np.array(curve_data[1])
    threshold = np.array(curve_data[2])
    tnr = 1 - fpr
    tpr_contains_nan = any(np.isnan(value) for value in tpr)
    if tpr_contains_nan:
        logger.warnings("tpr_contains_nan")
    inference_threshold = model_default_threshold

    fig.add_trace(
        go.Scatter(
            x=tnr,
            y=tpr,
            mode="lines",
            name=f"{model_name}",
            hoverinfo="text+name",
            hovertext=[
                f"Capture rate: {x}<br>False Filter Rate: {y}<br>Threshold: {z}" for x, y, z in zip(tpr, tnr, threshold)
            ],
            legendgroup=f"{model_name}",
        )
    )

    ##################################################################
    # Highest FFR when CR = 100%                                     #
    ##################################################################
    # Search for index of highest FR when CR = 1
    highest_fr_idx = np.where(tpr == 1.0)[0][0]

    # Draw highest FR when CR = 1
    fig.add_trace(
        go.Scatter(
            x=[tnr[highest_fr_idx]],
            y=[tpr[highest_fr_idx]],
            legendgroup=f"{model_name}",
            text=f"Th: {threshold[highest_fr_idx]:.6f}<br>"
            + f"CR: {tpr[highest_fr_idx]:.4f}<br>"
            + f"FFR: {tnr[highest_fr_idx]:.4f}",
            mode="markers+text",
            textposition="bottom center",
            cliponaxis=False,  # ensures annotation does not get clipped when exceeding boundary
            marker={"color": "blue", "size": 10},
            name="Highest False Filter Rate at 100% Capture Rate",
            hoverinfo="text",
            hovertext=f"""Highest False Filter Rate at 100% Capture Rate<br>
Capture rate: {tpr[highest_fr_idx]}<br>
False Filter Rate: {tnr[highest_fr_idx]}<br>
Threshold: {threshold[highest_fr_idx]:.6f}""",
        )
    )

    ##################################################################
    # Selected threshold                                             #
    ##################################################################
    # Draw marker for current selected model threshold
    fig.add_trace(
        go.Scatter(
            x=[selected_threshold_coord[0]],
            y=[selected_threshold_coord[1]],
            legendgroup=f"{model_name}",
            text=f"TH: {selected_threshold:.6f}<br>"
            + f"CR: {selected_threshold_coord[1]:.4f}<br>"
            + f"FFR: {selected_threshold_coord[0]:.4f}",
            mode="markers+text",
            textposition="top center",
            cliponaxis=False,  # ensures annotation does not get clipped when exceeding boundary
            marker={"color": "red", "size": 10},
            name=f"Selected Threshold ({selected_threshold:.6f})",
            hoverinfo="text",
            hovertext=f"""Selected Threshold<br>
Capture rate: {selected_threshold_coord[1]}<br>
False Filter Rate: {selected_threshold_coord[0]}<br>
Threshold: {selected_threshold:.6f}""",
        )
    )

    ##################################################################
    # Default threshold                                              #
    ##################################################################
    # Draw inference threshold
    if selected_threshold != inference_threshold:
        fig.add_trace(
            go.Scatter(
                x=[inference_threshold_coord[0]],
                y=[inference_threshold_coord[1]],
                legendgroup=f"{model_name}",
                text=f"Inference threshold = {inference_threshold:.6f} <br>"
                + f"Capture Rate: {inference_threshold_coord[1]:.4f} <br>"
                + f"False Filter Rate: {inference_threshold_coord[0]:.4f}",
                mode="markers+text",
                textposition="top center",
                cliponaxis=False,  # ensures annotation does not get clipped when exceeding boundary
                marker={"color": "black", "size": 10},
                name=f"Inference ({inference_threshold:.6f})",
                hoverinfo="text",
                hovertext=f"""Inference Threshold<br>
    Capture rate: {inference_threshold_coord[1]}<br>
    False Filter Rate: {inference_threshold_coord[0]}<br>
    Threshold: {inference_threshold:.6f}""",
            )
        )

    fig.add_trace(go.Scatter(x=[1, 0], y=[0, 1], mode="lines", line={"dash": "dash", "color": "grey"}, name="Random"))

    fig.update_layout(
        title=f"{model_name} Capture Rate / False Filter Rate Curve",
        xaxis_title="False Filter Rate",
        yaxis_title="Capture Rate",
        legend_title="Legends",
        template="plotly_white",
        showlegend=True,
        xaxis={"range": [0.0, 1.05]},
        yaxis={"range": [0.0, 1.05]},
    )

    # TODO: Make it square and can show properly on wide screen
    # fig.update_yaxes(
    #     scaleanchor="x",
    #     scaleratio=1,
    # )

    return fig


def plot_multilot_roc(
    model_name: str,
    selected_threshold: float,
    curve_data_list: list[tuple[list[float], list[float], list[float]]],
    lot_id_list: list[str],
    model_default_threshold_list: list[float],
    selected_threshold_coord_list: list[tuple[float, float]],
    inference_threshold_coord_list: list[tuple[float, float]],
) -> go.Figure:
    fig = go.Figure()

    for lot_data, lot_id, model_threshold, selected_threshold_coord, inference_threshold_coord in zip(
        curve_data_list,
        lot_id_list,
        model_default_threshold_list,
        selected_threshold_coord_list,
        inference_threshold_coord_list,
    ):
        fpr = np.array(lot_data[0])
        tpr = np.array(lot_data[1])
        threshold = np.array(lot_data[2])

        tnr = 1 - fpr

        # Skip lots that do not have any true defects
        tpr_contains_nan = any(np.isnan(value) for value in tpr)
        if tpr_contains_nan:
            continue

        inference_threshold = model_threshold

        # Draw the main curve
        fig.add_trace(
            go.Scatter(
                x=tnr,
                y=tpr,
                mode="lines",
                name=f"{model_name}: {lot_id}",
                hoverinfo="text+name",
                hovertext=[
                    f"Capture rate: {x}<br>False Filter Rate: {y}<br>Threshold: {z}"
                    for x, y, z in zip(tpr, tnr, threshold)
                ],
                legendgroup=f"{model_name}: {lot_id}",
                legendgrouptitle_text=f"{lot_id}",
            )
        )

        ##################################################################
        # Highest FFR when CR = 100%                                     #
        ##################################################################
        # Search for index of highest FR when CR = 1
        highest_fr_idx = np.where(tpr == 1.0)[0][0]

        # Draw highest FR when CR = 1
        fig.add_trace(
            go.Scatter(
                x=[tnr[highest_fr_idx]],
                y=[tpr[highest_fr_idx]],
                legendgroup=f"{model_name}: {lot_id}",
                text=f"Th: {threshold[highest_fr_idx]:.6f}<br>"
                + f"CR: {tpr[highest_fr_idx]:.4f}<br>"
                + f"FFR: {tnr[highest_fr_idx]:.4f}",
                mode="markers+text",
                textposition="bottom center",
                cliponaxis=False,  # ensures annotation does not get clipped when exceeding boundary
                marker={"color": "blue", "size": 10},
                name="Highest False Filter Rate at 100% Capture Rate",
                hoverinfo="text",
                hovertext=f"""Highest False Filter Rate at 100% Capture Rate<br>
    Capture rate: {tpr[highest_fr_idx]}<br>
    False Filter Rate: {tnr[highest_fr_idx]}<br>
    Threshold: {threshold[highest_fr_idx]:.6f}""",
            )
        )

        ##################################################################
        # Selected threshold                                             #
        ##################################################################
        # Draw marker for current selected model threshold
        fig.add_trace(
            go.Scatter(
                x=[selected_threshold_coord[0]],
                y=[selected_threshold_coord[1]],
                legendgroup=f"{model_name}: {lot_id}",
                text=f"TH: {selected_threshold:.6f}<br>"
                + f"CR: {selected_threshold_coord[1]:.4f}<br>"
                + f"FFR: {selected_threshold_coord[0]:.4f}",
                mode="markers+text",
                textposition="top center",
                cliponaxis=False,  # ensures annotation does not get clipped when exceeding boundary
                marker={"color": "red", "size": 10},
                name=f"Selected Threshold ({selected_threshold:.6f})",
                hoverinfo="text",
                hovertext=f"""Selected Threshold<br>
    Capture rate: {selected_threshold_coord[1]}<br>
    False Filter Rate: {selected_threshold_coord[0]}<br>
    Threshold: {selected_threshold:.6f}""",
            )
        )

        ##################################################################
        # Default threshold                                              #
        ##################################################################
        # Draw inference threshold
        if selected_threshold != inference_threshold:
            fig.add_trace(
                go.Scatter(
                    x=[inference_threshold_coord[0]],
                    y=[inference_threshold_coord[1]],
                    legendgroup=f"{model_name}: {lot_id}",
                    text=f"Inference threshold = {inference_threshold:.6f} <br>"
                    + f"Capture Rate: {inference_threshold_coord[1]:.4f} <br>"
                    + f"False Filter Rate: {inference_threshold_coord[0]:.4f}",
                    mode="markers+text",
                    textposition="top center",
                    cliponaxis=False,  # ensures annotation does not get clipped when exceeding boundary
                    marker={"color": "black", "size": 10},
                    name=f"Inference ({inference_threshold:.6f})",
                    hoverinfo="text",
                    hovertext=f"""Inference Threshold<br>
        Capture rate: {inference_threshold_coord[1]}<br>
        False Filter Rate: {inference_threshold_coord[0]}<br>
        Threshold: {inference_threshold:.6f}""",
                )
            )

    # Add a diagonal grey dotted-line
    fig.add_trace(go.Scatter(x=[1, 0], y=[0, 1], mode="lines", line={"dash": "dash", "color": "grey"}, name="Random"))

    fig.update_layout(
        title=f"{model_name} Capture Rate / False Filter Rate Curve",
        xaxis_title="False Filter Rate",
        yaxis_title="Capture Rate",
        legend_title="Legends",
        template="plotly_white",
        showlegend=True,
        xaxis={"range": [0.0, 1.05]},
        yaxis={"range": [0.0, 1.05]},
    )

    # TODO: Make it square and can show properly on wide screen
    # fig.update_yaxes(
    #     scaleanchor="x",
    #     scaleratio=1,
    # )

    return fig
