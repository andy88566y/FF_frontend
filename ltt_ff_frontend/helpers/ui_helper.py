from typing import Any
from ltt_ff_frontend.constant import BLANK_MODEL
from ltt_ff_frontend.helpers import api_helper
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

DEFECT_COLOR_MAPPING = {
    "D": "darkred",
    "ND": "olivedrab",
    "UNK": "blue",
}

def format_model_name(name: str | None) -> str:
    if name is None:
        return "SCRATCH"
    if name == BLANK_MODEL:
        return BLANK_MODEL
    # For base model name (e.g. base/LTT_SW#x9u#N3#M0-M2#20250124T000000Z#55032dae#55032dae.encrypted.pth)
    if "/" in name:
        model_paths = name.split("/")
        model_type = model_paths[-2]
        model_name = model_paths[-1]
        return f"[{model_type}] {model_name.replace('.encrypted', '').replace('.pth', '').replace('#', ' ')}"
    # For output model name (e.g. 13feb_minye#x9u#tl#lg20250213T151435Z#55032dae#5fb1017f)
    else:
        return f"{name.replace('.encrypted', '').replace('.pth', '').replace('#', ' ')}"

def calculate_filtered_results(
    raw_data: tuple[list[int], list[float], list[int]], selected_threshold: float
) -> dict[str, Any]:
    _, probability_list, answer_list = raw_data

    positive = answer_list.count(1)
    negative = answer_list.count(0)
    unlabeled = answer_list.count(-1)
    true_positive = sum(
        1 for prob, ans in zip(probability_list, answer_list) if prob >= selected_threshold and ans == 1
    )
    false_positive = sum(
        1 for prob, ans in zip(probability_list, answer_list) if prob >= selected_threshold and ans == 0
    )
    true_negative = sum(1 for prob, ans in zip(probability_list, answer_list) if prob < selected_threshold and ans == 0)
    false_negative = sum(
        1 for prob, ans in zip(probability_list, answer_list) if prob < selected_threshold and ans == 1
    )
    filtered_unlabeled_defect_count = sum(
        1 for prob, ans in zip(probability_list, answer_list) if prob >= selected_threshold and ans == -1
    )

    as_is_defect_count = positive + negative + unlabeled
    to_be_defect_count = true_positive + false_positive + filtered_unlabeled_defect_count

    capture_rate = true_positive / positive if positive > 0 else -1
    capture_rate = true_positive / positive if positive > 0 else -1
    false_filter_rate = true_negative / negative if negative > 0 else -1
    filter_rate = 1 - (to_be_defect_count / as_is_defect_count) if as_is_defect_count > 0 else -1

    return {
        "as_is_defect_count": as_is_defect_count,
        "to_be_defect_count": to_be_defect_count,
        "filter_rate": filter_rate,
        "as_is_true_defect_count": positive,
        "to_be_true_defect_count": true_positive,
        "capture_rate": capture_rate,
        "as_is_non_defect_count": negative,
        "to_be_non_defect_count": false_positive,
        "false_filter_rate": false_filter_rate,
        "unlabeled": unlabeled,
        "filtered_unlabeled_defect_count": filtered_unlabeled_defect_count,
    }

def get_classtype_count(defect_list: list[dict[str, Any]]) -> pd.DataFrame:
    classtype_counter: dict[str, int] = {}

    # Leave this as dict, move to backend in the future
    for defect in defect_list:
        defect_string = "Defect" if defect["Ans"] == 1 else "Non-defect" if defect["Ans"] == 0 else "Unlabeled"
        key = f"[{defect_string}] {defect['ClassType']}"
        classtype_counter[key] = classtype_counter.get(key, 0) + 1

    classtype_counter_list = []
    for key, value in classtype_counter.items():
        classification, classtype = key.split(" ")
        classtype_counter_list.append({"Classification": classification, "ClassType": classtype, "Count": value})

    # Convert to DF and sort by classtype
    classtype_counter_df = pd.DataFrame.from_records(data=classtype_counter_list).sort_values(
        by="ClassType", ascending=True, key=lambda classtype: classtype.astype(int)
    )

    # Reset index after sorting, and let it start from 1 instead of 0
    classtype_counter_df.reset_index(inplace=True, drop=True)
    classtype_counter_df.index = range(1, len(classtype_counter_df) + 1)

    return classtype_counter_df

def generate_1D_plot(
    data: tuple[list[int], list[float], list[int]], 
    threshold: float
) -> go.Figure:
    defect_ids, probs, ans = data

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

