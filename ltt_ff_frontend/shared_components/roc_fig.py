from typing import Any
from ltt_ff_frontend.shared_components import helper
from ltt_ff_frontend.helpers import api_helper

import plotly.graph_objects as go
import numpy as np
import streamlit as st

def gen_fig(
    inference_result_dir: str,
    raw_data: tuple[list[list[int]], list[list[float]], list[list[int]]],
    meta_list: list[dict[str, Any]],
    threshold: float,
    selected_lot_id_list: list[str] = []
) -> None:
    aggregated_data_lists = helper.aggregate_lists(raw_data, meta_list)
    # TODO: This should be done somewhere else
    if 1 not in set(aggregated_data_lists[2]):
        # All data is unlabeled or dataset consists of only non-defects
        st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")

    else:
        roc_data = api_helper.get_roc_data(inference_result_dir, return_curve=True)
        params = [("Recipe", roc_data, threshold, meta_list, inference_result_dir)]
        st.plotly_chart(plot_multilot_roc(params, selected_lot_id_list))

def plot_multilot_roc(
    # The outermost list is actually not needed; remove it and spread them into separate params.
    roc_data: list[tuple[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]], float, list[dict[str, Any], str]]],
    selected_lot_id_list: list[str] = [],
) -> go.Figure:
    fig = go.Figure()

    for curve_data in roc_data:
        model_name, data_list, selected_threshold, model_metadata_list, output_dir = curve_data

        selected_threshold_coord_list = api_helper.get_roc_threshold_marker_coordinates(
            output_dir=output_dir, selected_threshold=selected_threshold
        )

        inference_threshold_coord_list = api_helper.get_roc_threshold_marker_coordinates(
            output_dir=output_dir, selected_threshold=None
        )

        for lot_data, model_metadata, selected_threshold_coord, inference_threshold_coord in zip(
            data_list, model_metadata_list, selected_threshold_coord_list, inference_threshold_coord_list
        ):
            model_threshold = model_metadata.get("model_threshold", model_metadata.get("model_threshold_0", ""))

            fpr, tpr, threshold = lot_data
            tnr = 1 - fpr

            # Skip lots that do not have any true defects
            tpr_contains_nan = any(np.isnan(value) for value in tpr)
            if tpr_contains_nan:
                continue

            inference_threshold = model_threshold
            lot_id = model_metadata["lot_id"]

            # Skip lots if not selected
            if selected_lot_id_list and lot_id not in selected_lot_id_list:
                continue

            # Draw the main curve
            if len(roc_data) > 1 and model_name == "Model 1":
                fig.add_trace(
                    go.Scatter(
                        x=tnr,
                        y=tpr,
                        mode="lines",
                        line={"dash": "dash"},
                        name=f"{model_name}: {lot_id}",
                        hoverinfo="text+name",
                        hovertext=[
                            f"Capture rate: {x}<br>False Filter Rate: {y}<br>Threshold: {z}"
                            for x, y, z in zip(tpr, tnr, threshold)
                        ],
                    )
                )
            else:
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
                    mode="markers",
                    marker={"color": "blue", "size": 10},
                    name="Highest False Filter Rate at 100% Capture Rate",
                    hoverinfo="text",
                    hovertext=f"""Highest False Filter Rate at 100% Capture Rate<br>
        Capture rate: {tpr[highest_fr_idx]}<br>
        False Filter Rate: {tnr[highest_fr_idx]}<br>
        Threshold: {threshold[highest_fr_idx]:.6f}""",
                )
            )

            # Add annotation below the highest FR marker
            fig.add_annotation(
                x=tnr[highest_fr_idx],
                y=tpr[highest_fr_idx],
                text=f"""{model_name} Threshold = {threshold[highest_fr_idx]:.6f} <br>
        Capture Rate: {tpr[highest_fr_idx]:.4f} <br>
        False Filter Rate: {tnr[highest_fr_idx]:.4f}""",
                showarrow=False,
                yshift=-30,
            )

            ##################################################################
            # Selected threshold                                             #
            ##################################################################
            # Draw marker for current selected model threshold
            fig.add_trace(
                go.Scatter(
                    x=[selected_threshold_coord[0]],
                    y=[selected_threshold_coord[1]],
                    mode="markers",
                    marker={"color": "red", "size": 10},
                    name=f"Selected Threshold ({selected_threshold:.6f})",
                    hoverinfo="text",
                    hovertext=f"""Selected Threshold<br>
        Capture rate: {selected_threshold_coord[1]}<br>
        False Filter Rate: {selected_threshold_coord[0]}<br>
        Threshold: {selected_threshold:.6f}""",
                )
            )

            # Add annotation above current selected model threshold
            fig.add_annotation(
                x=selected_threshold_coord[0],
                y=selected_threshold_coord[1],
                text=f"""{model_name} Threshold = {selected_threshold:.6f} <br>
        Capture Rate: {selected_threshold_coord[1]:.4f} <br>
        False Filter Rate: {selected_threshold_coord[0]:.4f}""",
                showarrow=False,
                yshift=30,
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
                        mode="markers",
                        marker={"color": "black", "size": 10},
                        name=f"Inference ({inference_threshold:.6f})",
                        hoverinfo="text",
                        hovertext=f"""Inference Threshold<br>
            Capture rate: {inference_threshold_coord[1]}<br>
            False Filter Rate: {inference_threshold_coord[0]}<br>
            Threshold: {inference_threshold:.6f}""",
                    )
                )

                fig.add_annotation(
                    x=inference_threshold_coord[0],
                    y=inference_threshold_coord[1],
                    text=f"""{model_name} Inference threshold = {inference_threshold:.6f} <br>
            Capture Rate: {inference_threshold_coord[1]:.4f} <br>
            False Filter Rate: {inference_threshold_coord[0]:.4f}""",
                    showarrow=False,
                    yshift=-30,
                )

    # Add a diagonal grey dotted-line
    fig.add_trace(go.Scatter(x=[1, 0], y=[0, 1], mode="lines", line={"dash": "dash", "color": "grey"}, name="Random"))

    fig.update_layout(
        title="Capture Rate / False Filter Rate Curve",
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

def plot_roc(roc_data: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray], float, float, str]]) -> go.Figure:
    fig = go.Figure()

    for curve_data in roc_data:
        model_name, data, selected_threshold, inference_threshold, output_dir = curve_data
        fpr, tpr, threshold = data
        tnr = 1 - fpr

        # Draw the main curve
        fig.add_trace(
            go.Scatter(
                x=tnr,
                y=tpr,
                mode="lines",
                name=model_name,
                hoverinfo="text+name",
                hovertext=[
                    f"Capture rate: {x}<br>False Filter Rate: {y}<br>Threshold: {z}"
                    for x, y, z in zip(tpr, tnr, threshold)
                ],
            )
        )

        # Add a diagonal grey dotted-line
        fig.add_trace(
            go.Scatter(x=[1, 0], y=[0, 1], mode="lines", line={"dash": "dash", "color": "grey"}, name="Random")
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
                mode="markers",
                marker={"color": "blue", "size": 10},
                name="Highest False Filter Rate at 100% Capture Rate",
                hoverinfo="text",
                hovertext=f"""Highest False Filter Rate at 100% Capture Rate<br>
    Capture rate: {tpr[highest_fr_idx]}<br>
    False Filter Rate: {tnr[highest_fr_idx]}<br>
    Threshold: {threshold[highest_fr_idx]:.6f}""",
            )
        )

        # Add annotation below the highest FR marker
        fig.add_annotation(
            x=tnr[highest_fr_idx],
            y=tpr[highest_fr_idx],
            text=f"""{model_name} Threshold = {threshold[highest_fr_idx]:.6f} <br>
    Capture Rate: {tpr[highest_fr_idx]:.4f} <br>
    False Filter Rate: {tnr[highest_fr_idx]:.4f}""",
            showarrow=False,
            yshift=-30,
        )

        ##################################################################
        # Selected threshold                                             #
        ##################################################################
        # TODO: do not call api_helper, use argument instead
        x_value, y_value = api_helper.get_roc_threshold_marker_coordinates(
            output_dir=output_dir, selected_threshold=selected_threshold
        )[0]

        # Draw marker for current selected model threshold
        fig.add_trace(
            go.Scatter(
                x=[x_value],
                y=[y_value],
                mode="markers",
                marker={"color": "red", "size": 10},
                name=f"Selected Threshold ({selected_threshold:.6f})",
                hoverinfo="text",
                hovertext=f"""Selected Threshold<br>
    Capture rate: {y_value}<br>
    False Filter Rate: {x_value}<br>
    Threshold: {selected_threshold:.6f}""",
            )
        )

        # Add annotation above current selected model threshold
        fig.add_annotation(
            x=x_value,
            y=y_value,
            text=f"""{model_name} Threshold = {selected_threshold:.6f} <br>
    Capture Rate: {y_value:.4f} <br>
    False Filter Rate: {x_value:.4f}""",
            showarrow=False,
            yshift=30,
        )

        ##################################################################
        # Default threshold                                              #
        ##################################################################
        # Draw inference threshold
        if selected_threshold != inference_threshold:
            # TODO: do not call api_helper, use argument instead
            x_value, y_value = api_helper.get_roc_threshold_marker_coordinates(
                output_dir=output_dir, selected_threshold=inference_threshold
            )[0]

            fig.add_trace(
                go.Scatter(
                    x=[x_value],
                    y=[y_value],
                    mode="markers",
                    marker={"color": "black", "size": 10},
                    name=f"Inference ({inference_threshold:.6f})",
                    hoverinfo="text",
                    hovertext=f"""Inference Threshold<br>
        Capture rate: {y_value}<br>
        False Filter Rate: {x_value}<br>
        Threshold: {inference_threshold:.6f}""",
                )
            )

            fig.add_annotation(
                x=x_value,
                y=y_value,
                text=f"""{model_name} Inference threshold = {inference_threshold:.6f} <br>
        Capture Rate: {y_value:.4f} <br>
        False Filter Rate: {x_value:.4f}""",
                showarrow=False,
                yshift=-30,
            )

    fig.update_layout(
        title="Capture Rate / False Filter Rate Curve",
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
