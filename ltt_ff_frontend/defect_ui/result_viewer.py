from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


DEFECT_COLOR_MAPPING = {
    "D": "darkred",
    "ND": "olivedrab",
    "UNK": "blue",
}


def get_model_data(output_dir: str) -> tuple[dict[str, Any], tuple[list[int], list[float], list[int]]] | tuple[None, None]:
    try:
        db_metadata = helper.get_db_metadata(output_dir)
        defect_id_list = helper.get_defect_id(output_dir)
        probability_list = helper.get_probability(output_dir, defect_id_list)
        answer_list = helper.get_answer(output_dir, defect_id_list)
        return db_metadata, (defect_id_list, probability_list, answer_list)
    except Exception as e:
        logger.warning(f"Error getting model data from {output_dir}! {e}")
        return None, None


def calculate_filtered_results(raw_data: tuple[list[int], list[float], list[int]],
                               selected_threshold: float) -> dict[str, Any]:
    _, probability_list, answer_list = raw_data

    positive = answer_list.count(1)
    negative = answer_list.count(0)
    unlabeled = answer_list.count(-1)
    true_positive = sum(1 for prob, ans in zip(probability_list, answer_list)
                         if prob >= selected_threshold and ans == 1)
    false_positive = sum(1 for prob, ans in zip(probability_list, answer_list)
                         if prob >= selected_threshold and ans == 0)
    true_negative = sum(1 for prob, ans in zip(probability_list, answer_list)
                         if prob < selected_threshold and ans == 0)
    false_negative = sum(1 for prob, ans in zip(probability_list, answer_list)
                         if prob < selected_threshold and ans == 1)
    filtered_unlabeled_defect_count = sum(1 for prob, ans in zip(probability_list, answer_list)
                                     if prob >= selected_threshold and ans == -1)

    as_is_defect_count = positive + negative + unlabeled
    to_be_defect_count = true_positive + false_positive + filtered_unlabeled_defect_count

    capture_rate = 1 - (false_negative/positive) if positive > 0 else -1
    false_filter_rate = 1 - (false_positive/negative) if negative > 0 else -1
    filter_rate = 1 - (to_be_defect_count/as_is_defect_count) if as_is_defect_count > 0 else 1 - (to_be_defect_count/unlabeled)

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


def generate_1D_plot(m1_data: tuple[list[int], list[float], list[int]], m1_threshold: float) -> go.Figure:

    m1_defect_ids, m1_probs, m1_ans = m1_data

    df = pd.DataFrame(data={"Defect_ID": m1_defect_ids, "Probability": m1_probs, "LRF_Label": m1_ans})
    df["Classification"] = ["Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in df["LRF_Label"]]

    # Add histogram
    fig = px.histogram(data_frame=df,
                       x="Probability",
                       range_x=[0.0, 1.0],
                       nbins=100,
                       color="Classification",
                       color_discrete_map={"Non-defect":DEFECT_COLOR_MAPPING["ND"],
                                           "Defect":DEFECT_COLOR_MAPPING["D"],
                                           "Unlabeled":DEFECT_COLOR_MAPPING["UNK"]},
                       marginal="rug",
                       hover_name="Classification",
                       hover_data={
                                   "Probability": True,
                                   "Defect_ID": True,
                                   "LRF_Label": False,
                                   "Classification": False,
                                   },
                        labels={"LRF_Label": "Defect/non-defect",}
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


def generate_2D_plot(m1_data: tuple[list[int], list[float], list[int]],
                     m2_data: tuple[list[int], list[float], list[int]],
                     m1_threshold: float,
                     m2_threshold: float) -> go.Figure:

    m1_defect_ids, m1_probs, m1_ans = m1_data
    m2_defect_ids, m2_probs, m2_ans = m2_data
    assert m1_defect_ids == m2_defect_ids, "Defect IDs Count Mismatch!"

    defect_ids = [f"Defect ID: {defect_id}" for defect_id in m1_defect_ids]
    classifications = ['Defect' if a1 == 1 and a2 == 1 else 'Non-defect' if a1 == 0 and a2 == 0 else 'No-Label' for a1, a2 in zip(m1_ans, m2_ans)]
    marker_text = [f'{defect_id}<br>{classification}' for defect_id, classification in zip(defect_ids, classifications)]

    df = pd.DataFrame(data={"Defect_ID": m1_defect_ids,
                            "Probability_M1": m1_probs,
                            "Probability_M2": m2_probs,
                            "Classification": classifications})

    fig = px.scatter(df,
                     x="Probability_M1",
                     y="Probability_M2",
                     range_x=[0.0, 1.0],
                     range_y=[0.0, 1.0],
                     marginal_x="histogram",
                     marginal_y="histogram",
                     color="Classification",
                     color_discrete_map={"Non-defect":DEFECT_COLOR_MAPPING["ND"],
                                         "Defect":DEFECT_COLOR_MAPPING["D"],
                                         "No-Label":DEFECT_COLOR_MAPPING["UNK"]},
                     hover_data={
                         "Defect_ID": True
                     },
                     )

    # Workaround to set number of bins for the marginal histograms
    for _, trace in enumerate(fig.data):
        if trace.type == 'histogram':
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
    fig.add_shape(
        type="path",
        path="M 0 0 L 0 1 L 1 1 Z",
        line_width=0, fillcolor="lightpink", opacity=0.3
    )
    fig.add_shape(
        type="path",
        path="M 0 0 L 1 0 L 1 1 Z",
        line_width=0, fillcolor="palegreen", opacity=0.3
    )

    fig.update_layout(
        title="Model Comparision Chart",
        xaxis={"zeroline": False, "showgrid": False, "title": "Model 1 (Base)"},
        yaxis={"zeroline": False, "showgrid": False, "title": "Model 2 (Candidate)"},
        xaxis2={"zeroline": False, "showgrid": False, "title": "Model 2 Histogram"},
        yaxis2={"zeroline": False, "showgrid": False},
        xaxis3={"zeroline": False, "showgrid": False},
        yaxis3={"zeroline": False, "showgrid": False, "title": "Model 1 Histogram"},
        height=600,
        width=600,
        bargap=0,
        barmode="stack",
        hovermode="closest",
        showlegend=True,
    )

    return fig


def plot_roc(roc_data: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray], float, float]]) -> go.Figure:
    fig = go.Figure()

    for curve_data in roc_data:
        model_name, data, selected_threshold, inference_threshold = curve_data
        fpr, tpr, threshold = data
        tnr = 1 - fpr

        # Draw the main curve
        fig.add_trace(go.Scatter(x=tnr, y=tpr, mode="lines", name=model_name, hoverinfo='text+name',
                                 hovertext=[f'Capture rate: {x}<br>False Filter Rate: {y}<br>Threshold: {z}'
                                            for x, y, z in zip(tpr, tnr, threshold)]))

        # Add a diagonal grey dotted-line
        fig.add_trace(go.Scatter(x=[1, 0], y=[0, 1], mode="lines", line={"dash": "dash", "color": "grey"}, name="Random"))

        ##################################################################
        # Highest FFR when CR = 100%                                     #
        ##################################################################
        # Search for index of highest FR when CR = 1
        cr_one_indices = np.where(tpr == 1.0)[0]
        highest_fr_idx = cr_one_indices[0]

        # Draw highest FR when CR = 1
        fig.add_trace(go.Scatter(
            x=[tnr[highest_fr_idx]],
            y=[tpr[highest_fr_idx]],
            mode="markers",
            marker={"color": "blue", "size": 10},
            name="Highest Filter Rate at 100% Capture Rate",
            hoverinfo='text',
            hovertext=f"""Highest Filter Rate at 100% Capture Rate<br>
    Capture rate: {tpr[highest_fr_idx]}<br>
    Filter Rate: {tnr[highest_fr_idx]}<br>
    Threshold: {threshold[highest_fr_idx]:.6f}""",
        ))

        # Add annotation below the highest FR marker
        fig.add_annotation(
            x=tnr[highest_fr_idx],
            y=tpr[highest_fr_idx],
            text=f"""{model_name} Threshold = {threshold[highest_fr_idx]:.6f} <br>
    Capture Rate: {tpr[highest_fr_idx]:.4f} <br>
    Filter Rate: {tnr[highest_fr_idx]:.4f}""",
            showarrow=False,
            yshift=-30,
        )

        ##################################################################
        # Selected threshold                                             #
        ##################################################################
        # Search for the marker whose threshold is equal or smaller than selected threshold.
        # Note: need to reverse because threshold is from 1 to 0.
        reversed_threshold = threshold[::-1]
        selected_idx = np.searchsorted(reversed_threshold, selected_threshold, side='left')
        selected_idx = len(threshold) - selected_idx -1

        # Draw marker for current selected model threshold
        fig.add_trace(go.Scatter(
            x=[tnr[selected_idx]],
            y=[tpr[selected_idx]],
            mode="markers",
            marker={"color": "red", "size": 10},
            name=f"Selected Threshold ({selected_threshold:.6f})",
            hoverinfo='text',
            hovertext=f"""Selected Threshold<br>
    Capture rate: {tpr[selected_idx]}<br>
    Filter Rate: {tnr[selected_idx]}<br>
    Threshold: {selected_threshold:.6f}""",
        ))

        # Add annotation above current selected model threshold
        fig.add_annotation(
            x=tnr[selected_idx],
            y=tpr[selected_idx],
            text=f"""{model_name} Threshold = {selected_threshold:.6f} <br>
    Capture Rate: {tpr[selected_idx]:.4f} <br>
    Filter Rate: {tnr[selected_idx]:.4f}""",
            showarrow=False,
            yshift=30,
        )

        ##################################################################
        # Default threshold                                              #
        ##################################################################
        # Draw inference threshold
        if selected_threshold != inference_threshold:
            infer_idx = np.searchsorted(reversed_threshold, inference_threshold, side='left')
            infer_idx = len(threshold) - infer_idx - 1

            fig.add_trace(go.Scatter(
                x=[tnr[infer_idx]],
                y=[tpr[infer_idx]],
                mode="markers",
                marker={"color": "black", "size": 10},
                name=f"Inference ({inference_threshold:.6f})",
                hoverinfo='text',
                hovertext=f"""Inference Threshold<br>
        Capture rate: {tpr[infer_idx]}<br>
        Filter Rate: {tnr[infer_idx]}<br>
        Threshold: {inference_threshold:.6f}""",
            ))

            fig.add_annotation(
                x=tnr[infer_idx],
                y=tpr[infer_idx],
                text=f"""{model_name} Inference threshold = {inference_threshold:.6f} <br>
        Capture Rate: {tpr[infer_idx]:.4f} <br>
        Filter Rateee: {tnr[infer_idx]:.4f}""",
                showarrow=False,
                yshift=-30,
            )

    fig.update_layout(
        title="Capture Rate / Filter Rate Curve",
        xaxis_title="Filter Rate",
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
        fig.add_trace(go.Scatter(
            x=[recall[selected_idx]],
            y=[precision[selected_idx]],
            mode="markers",
            marker={"color": "red", "size": 10},
            name=f"Threshold = {threshold[selected_idx]:.2f}",
        ))
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


def gen_lrf(model_id: str, output_dir: str, gen_lrf_type: str, threshold=None, top_k=None) -> None:
    if gen_lrf_type == "top_k":
        if top_k is not None and 1 <= top_k <= 999:
            if st.button(f"Generate new Model {model_id} lrf"):
                request = helper.request_top_k_lrf(output_dir=output_dir, top_k=top_k)

                if request.json().get('status') == 'error':
                    code = request.json().get('code')
                    message = request.json().get('message')
                    st.error(f'.lrf file not generated!\nError code: {code}\nError message: {message}')
                    logger.error(f'.lrf file not generated!\nError code: {code}\nError message: {message}')
                else:
                    st.success(f'New .lrf file (top_k: {top_k}) generated at {output_dir}!')
                    logger.info(f'New .lrf file (top_k: {top_k}) generated at {output_dir}!')
        else:
            st.error(f"Top-k setting: {top_k} is invalid. Should be between 1 and 999 !")
    elif gen_lrf_type == "threshold":
        if threshold is not None and 0.0 <= threshold <= 1.0:
            if st.button(f"Generate new Model {model_id} lrf"):
                request = helper.request_threshold_lrf(output_dir=output_dir, confidence_threshold=threshold)

                if request.json().get('status') == 'error':
                    code = request.json().get('code')
                    message = request.json().get('message')
                    st.error(f'.lrf file not generated!\nError code: {code}\nError message: {message}')
                    logger.error(f'.lrf file not generated!\nError code: {code}\nError message: {message}')
                else:
                    st.success(f'New .lrf file (threshold: {threshold}) generated at {output_dir}!')
                    logger.info(f'New .lrf file (threshold: {threshold}) generated at {output_dir}!')
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

    r2_col1, _r2_col2 = st.columns([3, 2])

    vr1_col1, vr1_col2, vr1_col3, vr1_col4, vr1_col5 = st.columns([1, 2, 3, 2, 2])
    vr2_col1, vr2_col2, vr2_col3, vr2_col4, vr2_col5 = st.columns([1, 2, 3, 2, 2])

    st.divider()

    # Defining columns to display filter results (capture rate, filter rate, etc.)
    model_1_results_container = st.empty()
    r3_col1, r3_col2, r3_col3, r3_col4 = st.columns([4, 3, 3, 2])
    model_2_results_container = st.empty()
    r4_col1, r4_col2, r4_col3, r4_col4 = st.columns([4, 3, 3, 2])

    st.divider()

    vr3_col1, vr3_col2 = st.columns(2)

    invalid_input = [output_dir_default, '']

    if rv_m1_output_dir not in invalid_input and rv_m2_output_dir not in invalid_input:
        model_1_metadata, model_1_raw_data = get_model_data(rv_m1_output_dir)
        model_2_metadata, model_2_raw_data = get_model_data(rv_m2_output_dir)

        if model_1_metadata is None:
            with r2_col1:
                st.error(f"Error getting result data from {rv_m1_output_dir}")
                return

        if model_2_metadata is None:
            with r2_col1:
                st.error(f"Error getting result data from {rv_m2_output_dir}")
                return

        if model_1_metadata['lot_id'] != model_2_metadata['lot_id']:
            with r2_col1:
                st.error(f"Lot IDs do not match!  \nModel 1 lot ID: {model_1_metadata['lot_id']}  \nModel 2 lot ID: {model_2_metadata['lot_id']}")
                return

        # Show result database details
        with vr1_col1:
            st.text("Model 1 (Base)")
        with vr2_col1:
            st.text("Model 2 (Candidate)")
        with vr1_col2:
            st.text(f"Lot ID:\n{model_1_metadata['lot_id']}")
        with vr2_col2:
            st.text(f"Lot ID:\n{model_2_metadata['lot_id']}")
        with vr1_col3:
            st.text(f"Inference Model:\n{helper.format_model_name(model_1_metadata['model_name'])}")
        with vr2_col3:
            st.text(f"Inference Model:\n{helper.format_model_name(model_2_metadata['model_name'])}")

        if st_gen_lrf_type == "top_k":
            with vr1_col4:
                rv_m1_topk = st.number_input("Top k", 0, 999, 150, 1,
                                             help="Top-k defects ranked by Probabilities will be considered as defects.", key='m1_topk')
            with vr2_col4:
                rv_m2_topk = st.number_input("Top k", 0, 999, 150, 1,
                                             help="Top-k defects ranked by Probabilities will be considered as defects.", key='m2_topk')
            with vr1_col5:
                gen_lrf("1", rv_m1_output_dir, st_gen_lrf_type, top_k=rv_m1_topk)
            with vr2_col5:
                gen_lrf("2", rv_m2_output_dir, st_gen_lrf_type, top_k=rv_m2_topk)

            rv_m1_threshold = helper.get_topk_model_threshold(rv_m1_output_dir, rv_m1_topk)
            rv_m2_threshold = helper.get_topk_model_threshold(rv_m2_output_dir, rv_m2_topk)
        else:
            with vr1_col4:
                rv_m1_threshold = st.number_input(label="Confidence threshold:",
                                                  value=model_1_metadata['model_threshold'],
                                                  step=0.00001,
                                                  format="%.5f",
                                                  help="Probabilities above thershold will be considered as defects.",
                                                  key='m1_threshold')
            with vr2_col4:
                rv_m2_threshold = st.number_input(label="Confidence threshold:",
                                                  value=model_2_metadata['model_threshold'],
                                                  step=0.00001,
                                                  format="%.5f",
                                                  help="Probabilities above thershold will be considered as defects.",
                                                  key='m2_threshold')

            # Validate confidence thresholds
            if rv_m1_threshold < 0.0 or rv_m1_threshold > 1.0:
                logger.error(f'Confidence threshold must be between 0.0 and 1.0! Model 1 selected confidence threshold: {rv_m1_threshold}')
                st.error(f'Confidence threshold must be between 0.0 and 1.0! Model 1 selected confidence threshold: {rv_m1_threshold}')
                return
            elif rv_m2_threshold < 0.0 or rv_m2_threshold > 1.0:
                logger.error(f'Confidence threshold must be between 0.0 and 1.0! Model 2 selected confidence threshold: {rv_m2_threshold}')
                st.error(f'Confidence threshold must be between 0.0 and 1.0! Model 2 selected confidence threshold: {rv_m2_threshold}')
                return

            with vr1_col5:
                gen_lrf("1", rv_m1_output_dir, st_gen_lrf_type, threshold=rv_m1_threshold)
            with vr2_col5:
                gen_lrf("2", rv_m2_output_dir, st_gen_lrf_type, threshold=rv_m2_threshold)

        # Show Total/Defect/Non-defect/unlabeled count
        with model_1_results_container:
            st.text("Model 1 results")
            count_rate_data = calculate_filtered_results(model_1_raw_data, rv_m1_threshold)
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
        with model_2_results_container:
            st.text("Model 2 results")
            count_rate_data = calculate_filtered_results(model_2_raw_data, rv_m2_threshold)
            with r4_col1:
                st.warning(f"""**Total defect count**: As-is: {count_rate_data['as_is_defect_count']}
                        → To-be: {count_rate_data['to_be_defect_count']}
                        (Filter Rate: {count_rate_data['filter_rate']:.4f})""")
            with r4_col2:
                st.error(f"""**True defect count**: {count_rate_data['as_is_true_defect_count']}
                        → {count_rate_data['to_be_true_defect_count']}
                        (Capture Rate: {count_rate_data['capture_rate']:.4f})""")
            with r4_col3:
                st.success(f"""**Non-defect count**: {count_rate_data['as_is_non_defect_count']}
                        → {count_rate_data['to_be_non_defect_count']}
                        (False Filter Rate: {count_rate_data['false_filter_rate']:.4f})""")
            with r4_col4:
                st.info(f"""**Unlabeled count**: {count_rate_data['unlabeled']}
                        → {count_rate_data['filtered_unlabeled_defect_count']}""")

        # Draw 2D comparison chart
        with vr3_col1:
            st.plotly_chart(generate_2D_plot(model_1_raw_data, model_2_raw_data, rv_m1_threshold, rv_m2_threshold))

        with vr3_col2:
            # TODO: This should be done somewhere else
            if 1 not in set(model_1_raw_data[2]) or 1 not in set(model_2_raw_data[2]):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")

            else:
                model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, return_curve=True)
                model_2_roc_data = helper.get_roc_data(rv_m2_output_dir, return_curve=True)
                st.plotly_chart(plot_roc([
                    ("Model 1", model_1_roc_data, rv_m1_threshold, model_1_metadata['model_threshold']),
                    ("Model 2", model_2_roc_data, rv_m2_threshold, model_2_metadata['model_threshold']),
                ]))

                # model_1_prc_data = helper.get_prc_data(rv_m1_output_dir, return_curve=True)
                # model_2_prc_data = helper.get_prc_data(rv_m2_output_dir, return_curve=True)
                # st.plotly_chart(plot_prc([
                #     ("Model 1", model_1_prc_data, rv_m1_threshold),
                #     ("Model 2", model_2_prc_data, rv_m2_threshold),
                # ]))

    elif rv_m1_output_dir not in invalid_input:
        model_1_metadata, model_1_raw_data = get_model_data(rv_m1_output_dir)

        if model_1_metadata is None:
            with r2_col1:
                st.error(f"Error getting result data from {rv_m1_output_dir}")
                return

        # Show result database details
        with vr1_col1:
            st.text("Model 1 (Base)")
        with vr1_col2:
            st.text(f"Lot ID:\n{model_1_metadata['lot_id']}")
        with vr1_col3:
            st.text(f"Inference Model:\n{helper.format_model_name(model_1_metadata['model_name'])}")

        if st_gen_lrf_type == "top_k":
            with vr1_col4:
                rv_m1_topk = st.number_input("Top k", 0, 999, 150, 1,
                                            help="Top-k defects ranked by Probabilities will be considered as defects.", key='m1_topk')
            with vr1_col5:
                gen_lrf("1", rv_m1_output_dir, st_gen_lrf_type, top_k=rv_m1_topk)

            rv_m1_threshold = helper.get_topk_model_threshold(rv_m1_output_dir, rv_m1_topk)
        else:
            with vr1_col4:
                rv_m1_threshold = st.number_input(label="Confidence threshold:",
                                                  value=model_1_metadata['model_threshold'],
                                                  step=0.00001,
                                                  format="%.5f",
                                                  help="Probabilities above thershold will be considered as defects.",
                                                  key='m1_threshold')

            # Validate confidence threshold
            if rv_m1_threshold < 0.0 or rv_m1_threshold > 1.0:
                logger.error(f'Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {rv_m1_threshold}')
                st.error(f'Confidence threshold must be between 0.0 and 1.0! Selected confidence threshold: {rv_m1_threshold}')
                return

            with vr1_col5:
                gen_lrf("1", rv_m1_output_dir, st_gen_lrf_type, threshold=rv_m1_threshold)

        # Show Total/Defect/Non-defect/unlabeled count
        count_rate_data = calculate_filtered_results(model_1_raw_data, rv_m1_threshold)
        with r3_col1:
            st.warning(f"""**Total defect count**: As-is: {count_rate_data['as_is_defect_count']}
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

        # Draw 1D comparison chart
        with vr3_col1:
            st.plotly_chart(generate_1D_plot(model_1_raw_data, rv_m1_threshold))

        with vr3_col2:
            # TODO: This should be done somewhere else
            if 1 not in set(model_1_raw_data[2]):
                # All data is unlabeled or dataset consists of only non-defects
                st.markdown("##### All data is unlabeled or no defects found! Skipping chart.")

            else:
                model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, return_curve=True)
                st.plotly_chart(plot_roc([("Model 1", model_1_roc_data, rv_m1_threshold, model_1_metadata['model_threshold'])]))

    else:
        pass
