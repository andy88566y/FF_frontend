import glob
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper
from ltt_ff_frontend.result_viewer import multi_lot_result_viewer, single_lot_result_viewer


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


def get_model_data(
    output_dir: str,
) -> tuple[dict[str, Any], tuple[list[int], list[float], list[int]]] | tuple[None, None]:
    try:
        db_metadata = helper.get_db_metadata_lists(output_dir)
        defect_id_lists = helper.get_defect_id_lists(output_dir)
        probability_list = helper.get_probability(output_dir, defect_id_lists)
        answer_list = helper.get_answer(output_dir, defect_id_lists)
        return db_metadata[0], (defect_id_lists[0], probability_list[0], answer_list[0])
    except Exception as e:
        logger.warning(f"Error getting model data from {output_dir}! {e}")
        return None, None


def get_multilot_model_data(
    output_dir: str,
) -> tuple[list[dict[str, Any]], tuple[list[list[int]], list[list[float]], list[list[int]]]] | tuple[None, None]:
    try:
        db_metadata = helper.get_db_metadata_lists(output_dir=output_dir)
        defect_id_list = helper.get_defect_id_lists(output_dir=output_dir)
        probability_list = helper.get_probability(output_dir, defect_id_list)
        answer_list = helper.get_answer(output_dir, defect_id_list)
        assert len(defect_id_list) == len(probability_list), f"IDs: {len(defect_id_list)} Prob: {len(probability_list)}"
        assert len(defect_id_list) == len(answer_list), f"IDs: {len(defect_id_list)} Ans: {len(answer_list)}"
        return db_metadata, (defect_id_list, probability_list, answer_list)
    except Exception as e:
        logger.warning(f"Error getting model data from {output_dir}! {e}")
        return None, None


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


def show_multilot_statistics(
    raw_data: tuple[list[list[int]], list[list[float]], list[list[int]]],
    model_metadata: list[dict[str, Any]],
    selected_threshold,
) -> None:
    defect_id_lists, probability_lists, answer_lists = raw_data
    for id_list, prob_list, ans_list, meta in zip(defect_id_lists, probability_lists, answer_lists, model_metadata):
        st.text(f"{meta['lot_id']}")
        count_rate_data = calculate_filtered_results((id_list, prob_list, ans_list), selected_threshold)
        r3_col1, r3_col2, r3_col3, r3_col4 = st.columns([4, 3, 3, 2])
        with r3_col1:
            st.warning(f"""**Total defect count**: As-is {count_rate_data['as_is_defect_count']}
                    → To-be: {count_rate_data['to_be_defect_count']}
                    (Filter Rate: {count_rate_data['filter_rate']:.4f})""")
        with r3_col2:
            st.error(f"""**True defect count**: {count_rate_data['as_is_true_defect_count']}
                    → {count_rate_data['to_be_true_defect_count']}
                    (Capture Rate: {count_rate_data['capture_rate']:.4f})""")
        with r3_col3:
            st.success(f"""**Non-defect count**: {count_rate_data['as_is_non_defect_count']}
                    → {count_rate_data['to_be_non_defect_count']}
                    (False Filter Rate: {count_rate_data['false_filter_rate']:.4f})""")
        with r3_col4:
            st.info(f"""**Unlabeled count**: {count_rate_data['unlabeled']}
                    → {count_rate_data['filtered_unlabeled_defect_count']}""")


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


def aggregate_lists(
    raw_data: tuple[list[int], list[float], list[int]], meta_list: list[dict[str, Any]]
) -> tuple[list[Any]]:
    defect_id_lists, prob_lists, ans_lists = raw_data
    lot_id_lists = [meta["lot_id"] for meta in meta_list]
    aggregate_id_list, aggregate_prob_list, aggregate_ans_list, aggregate_lot_id_list = [], [], [], []
    for defect_id_list, prob_list, ans_list, lot_id in zip(defect_id_lists, prob_lists, ans_lists, lot_id_lists):
        aggregate_id_list.extend(defect_id_list)
        aggregate_prob_list.extend(prob_list)
        aggregate_ans_list.extend(ans_list)
        aggregate_lot_id_list.extend([lot_id] * len(defect_id_list))

    return (aggregate_id_list, aggregate_prob_list, aggregate_ans_list, aggregate_lot_id_list)


def generate_1D_plot(m1_data: tuple[list[int], list[float], list[int]], m1_threshold: float) -> go.Figure:
    m1_defect_ids, m1_probs, m1_ans = m1_data

    df = pd.DataFrame(data={"Defect_ID": m1_defect_ids, "Probability": m1_probs, "LRF_Label": m1_ans})
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
        x0=m1_threshold,
        x1=m1_threshold,
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
    m1_data: tuple[list[int], list[float], list[int], list[str]], threshold: float
) -> go.Figure:
    id_list, prob_list, ans_list, lot_id_list = m1_data
    df = pd.DataFrame(
        data={"Defect_ID": id_list, "Probability": prob_list, "LRF_Label": ans_list, "Lot ID": lot_id_list}
    )
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


def generate_2D_plot(
    m1_data: tuple[list[int], list[float], list[int], list[str]],
    m2_data: tuple[list[int], list[float], list[int], list[str]],
    m1_threshold: float,
    m2_threshold: float,
) -> go.Figure:
    m1_defect_ids, m1_probs, m1_ans, m1_lot_ids = m1_data
    m2_defect_ids, m2_probs, m2_ans, m2_lot_ids = m2_data
    assert sorted(m1_defect_ids) == sorted(m2_defect_ids), "Defect IDs Count Mismatch!"

    defect_ids = [f"Defect ID: {defect_id}" for defect_id in m1_defect_ids]
    classifications = [
        "Defect" if a1 == 1 and a2 == 1 else "Non-defect" if a1 == 0 and a2 == 0 else "No-Label"
        for a1, a2 in zip(m1_ans, m2_ans)
    ]
    marker_text = [f"{defect_id}<br>{classification}" for defect_id, classification in zip(defect_ids, classifications)]
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
        x_value, y_value = helper.get_roc_threshold_marker_coordinates(
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
            # Search for the marker whose threshold is equal or smaller than inference threshold.
            # Note: need to reverse because threshold is from 1 to 0.
            reversed_threshold = threshold[::-1]
            infer_idx = np.searchsorted(reversed_threshold, inference_threshold, side="left")
            infer_idx = len(threshold) - infer_idx - 1

            fig.add_trace(
                go.Scatter(
                    x=[tnr[infer_idx]],
                    y=[tpr[infer_idx]],
                    mode="markers",
                    marker={"color": "black", "size": 10},
                    name=f"Inference ({inference_threshold:.6f})",
                    hoverinfo="text",
                    hovertext=f"""Inference Threshold<br>
        Capture rate: {tpr[infer_idx]}<br>
        False Filter Rate: {tnr[infer_idx]}<br>
        Threshold: {inference_threshold:.6f}""",
                )
            )

            fig.add_annotation(
                x=tnr[infer_idx],
                y=tpr[infer_idx],
                text=f"""{model_name} Inference threshold = {inference_threshold:.6f} <br>
        Capture Rate: {tpr[infer_idx]:.4f} <br>
        False Filter Rate: {tnr[infer_idx]:.4f}""",
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


def plot_multilot_roc(
    roc_data: list[tuple[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]], float, list[dict[str, Any]]]],
) -> go.Figure:
    fig = go.Figure()

    for curve_data in roc_data:
        model_name, data_list, selected_threshold, model_metadata_list, output_dir = curve_data

        selected_threshold_coord_list = helper.get_roc_threshold_marker_coordinates(
            output_dir=output_dir, selected_threshold=selected_threshold
        )

        for lot_data, model_metadata, selected_threshold_coord in zip(
            data_list, model_metadata_list, selected_threshold_coord_list
        ):
            fpr, tpr, threshold = lot_data
            tnr = 1 - fpr

            # Skip lots that do not have any true defects
            tpr_contains_nan = any(np.isnan(value) for value in tpr)
            if tpr_contains_nan:
                continue

            inference_threshold = model_metadata["model_threshold"]
            lot_id = model_metadata["lot_id"]

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
                # Search for the marker whose threshold is equal or smaller than inference threshold.
                # Note: need to reverse because threshold is from 1 to 0.
                reversed_threshold = threshold[::-1]
                infer_idx = np.searchsorted(reversed_threshold, inference_threshold, side="left")
                infer_idx = len(threshold) - infer_idx - 1

                fig.add_trace(
                    go.Scatter(
                        x=[tnr[infer_idx]],
                        y=[tpr[infer_idx]],
                        mode="markers",
                        marker={"color": "black", "size": 10},
                        name=f"Inference ({inference_threshold:.6f})",
                        hoverinfo="text",
                        hovertext=f"""Inference Threshold<br>
            Capture rate: {tpr[infer_idx]}<br>
            False Filter Rate: {tnr[infer_idx]}<br>
            Threshold: {inference_threshold:.6f}""",
                    )
                )

                fig.add_annotation(
                    x=tnr[infer_idx],
                    y=tpr[infer_idx],
                    text=f"""{model_name} Inference threshold = {inference_threshold:.6f} <br>
            Capture Rate: {tpr[infer_idx]:.4f} <br>
            False Filter Rate: {tnr[infer_idx]:.4f}""",
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


# TODO: Might be wrong! Refer to plot_roc for correct method.
def plot_prc(prc_data: list[tuple[str, Any, float]]) -> go.Figure:
    fig = go.Figure()

    for curve_data in prc_data:
        model_name, data, model_threshold = curve_data
        precision, recall, threshold = data

        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=model_name))

        # Draw current selected model threshold
        selected_idx = (np.abs(threshold - model_threshold)).argmin()
        fig.add_trace(
            go.Scatter(
                x=[recall[selected_idx]],
                y=[precision[selected_idx]],
                mode="markers",
                marker={"color": "red", "size": 10},
                name=f"Threshold = {threshold[selected_idx]:.2f}",
            )
        )
        fig.add_annotation(
            x=recall[selected_idx],
            y=precision[selected_idx],
            text=f"{model_name} Threshold = {threshold[selected_idx]:.2f} <br> Recall: {recall[selected_idx]:.4f} <br> Precision: {precision[selected_idx]:.4f}",
            showarrow=True,
            arrowhead=2,
        )

    fig.update_layout(
        title="Precision / Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend_title="Models",
        template="plotly_white",
        showlegend=True,
        xaxis={"range": [0.0, 1.05]},
        yaxis={"range": [0.0, 1.05]},
    )

    return fig


def gen_lrf(
    model_id: str, output_dir: str, gen_lrf_type: str, threshold=None, top_k=None, key_number: int = 0, lot_id: str = ""
) -> None:
    if gen_lrf_type == "top_k":
        if top_k is not None and 1 <= top_k <= 999:
            if st.button(label=f"Generate new Model {model_id} lrf", key=f"gen_lrf_top_k_{key_number}"):
                request = helper.request_top_k_lrf(output_dir=output_dir, top_k=top_k, lot_id=lot_id)

                if request.json().get("status") == "error":
                    code = request.json().get("code")
                    message = request.json().get("message")
                    st.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                    logger.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                else:
                    st.success(f"New .lrf file (top_k: {top_k}) generated at {output_dir}!")
                    logger.info(f"New .lrf file (top_k: {top_k}) generated at {output_dir}!")
        else:
            st.error(f"Top-k setting: {top_k} is invalid. Should be between 1 and 999 !")
    elif gen_lrf_type == "threshold":
        if threshold is not None and 0.0 <= threshold <= 1.0:
            if st.button(f"Generate new Model {model_id} lrf", key=f"gen_lrf_threshold_{key_number}"):
                request = helper.request_threshold_lrf(
                    output_dir=output_dir, confidence_threshold=threshold, lot_id=lot_id
                )

                if request.json().get("status") == "error":
                    code = request.json().get("code")
                    message = request.json().get("message")
                    st.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                    logger.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                else:
                    st.success(f"New .lrf file (threshold: {threshold}) generated at {output_dir}!")
                    logger.info(f"New .lrf file (threshold: {threshold}) generated at {output_dir}!")
        else:
            st.error(f"Threshold setting: {threshold} is invalid. Should be between 0.0 and 1.0 !")
    else:
        raise NotImplementedError(f"gen_lrf_type {gen_lrf_type} is not implemented.")


def app() -> None:
    logger.debug("Loading Result Viewer...")
    st.title("False Filter Result Viewer")
    st.caption("Visualize False Filter Result [Model 1 - Base] [Model 2 - Candidate (optional)]")

    r1_col1, r1_col2, r1_col3 = st.columns([3, 3, 2])

    output_dir_default = "/mnt/dbpc/xxx"
    with r1_col1:
        rv_m1_output_dir = st.text_input("Model 1 (Base) Result Directory", value=output_dir_default)
    with r1_col2:
        rv_m2_output_dir = st.text_input("Model 2 (Candidate) Result Directory", value=output_dir_default)
    with r1_col3:
        st_gen_lrf_type = st.segmented_control("lrf Generation Option", ["threshold", "top_k"], default="threshold")

    if st_gen_lrf_type is None:
        st.error("lrf Generation Option can not be None!")
        return

    st.divider()

    db_files = glob.glob(f"{rv_m1_output_dir}/*.db")
    if len(db_files) > 1:
        multi_lot_result_viewer.app(output_dir_default, rv_m1_output_dir, rv_m2_output_dir, st_gen_lrf_type)
    else:
        single_lot_result_viewer.app(output_dir_default, rv_m1_output_dir, rv_m2_output_dir, st_gen_lrf_type)
