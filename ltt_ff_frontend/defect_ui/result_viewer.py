from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


DEFECT_COLOR_MAPPING = {
    "D": "darkred",
    "ND": "olivedrab",
    "UNK": "blue",
}


def get_model_data(output_dir: str):
    try:
        db_metadata = helper.get_db_metadata(output_dir)
        defect_id_list = helper.get_defect_id(output_dir)
        probability_list = helper.get_probability(output_dir, defect_id_list)
        answer_list = helper.get_answer(output_dir, defect_id_list)
        return db_metadata, (defect_id_list, probability_list, answer_list)
    except Exception as e:
        logger.warning(f"Error getting model data from {output_dir}! {e}")
        return None, None


def generate_1D_plot(m1_data, m1_threshold: float):
    m1_defect_ids, m1_probs, m1_ans = m1_data

    df = pd.DataFrame(data={"Defect_ID": m1_defect_ids, "Probability": m1_probs, "LRF_Label": m1_ans})

    # TODO: Change to use plotly.express and add marginal="rug"

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df[df["LRF_Label"] == 1]["Probability"],
            marker={"color": DEFECT_COLOR_MAPPING["D"]}, xbins={"start": 0.00, "end": 1.00, "size": 0.01},
            name="Defect"
        )
    )
    fig.add_trace(
        go.Histogram(
            x=df[df["LRF_Label"] == 0]["Probability"],
            marker={"color": DEFECT_COLOR_MAPPING["ND"]}, xbins={"start": 0.00, "end": 1.00, "size": 0.01},
            name="Non-Defect"
        )
    )
    fig.add_trace(
        go.Histogram(
            x=df[df["LRF_Label"] == -1]["Probability"],
            marker={"color": DEFECT_COLOR_MAPPING["UNK"]}, xbins={"start": 0.00, "end": 1.00, "size": 0.01},
            name="No-Label"
        )
    )

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


def generate_2D_plot(m1_data, m2_data, m1_threshold: float, m2_threshold: float):
    m1_defect_ids, m1_probs, m1_ans = m1_data
    m2_defect_ids, m2_probs, m2_ans = m2_data
    assert m1_defect_ids == m2_defect_ids, "Defect IDs Count Mismatch!"

    defect_ids = [f"Defect ID: {defect_id}" for defect_id in m1_defect_ids]
    classifications = ['Defect' if a1 == 1 and a2 == 1 else 'Non-defect' if a1 == 0 and a2 == 0 else 'No-Label' for a1, a2 in zip(m1_ans, m2_ans)]
    marker_text = [f'{defect_id}<br>{classification}' for defect_id, classification in zip(defect_ids, classifications)]

    # 2D scatter plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=m1_probs,
            y=m2_probs,
            xaxis="x",
            yaxis="y",
            mode="markers",
            marker={
                "color": [
                    DEFECT_COLOR_MAPPING["D"] if a1 == 1 and a2 == 1 else DEFECT_COLOR_MAPPING["ND"] if a1 == 0 and a2 == 0 else DEFECT_COLOR_MAPPING["UNK"]
                    for a1, a2 in zip(m1_ans, m2_ans)
                ],
                "size": 5,
            },
            text=marker_text,
            hoverinfo="text",
            hovertemplate="%{text}<br>Model 1 Prob: %{x}<br>Model 2 Prob: %{y}",
            name=""
        )
    )

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

    # Add side histograms
    fig.add_trace(
        go.Histogram(
            y=[p for p, a in zip(m2_probs, m2_ans) if a == 1], xaxis="x2",
            marker={"color": DEFECT_COLOR_MAPPING["D"]}, ybins={"start": 0.00, "end": 1.00, "size": 0.01},
            name='Defects',
        )
    )
    fig.add_trace(
        go.Histogram(
            y=[p for p, a in zip(m2_probs, m2_ans) if a == 0], xaxis="x2",
            marker={"color": DEFECT_COLOR_MAPPING["ND"]}, ybins={"start": 0.00, "end": 1.00, "size": 0.01},
            name='Non-defects',
        )
    )
    fig.add_trace(
        go.Histogram(
            y=[p for p, a in zip(m2_probs, m2_ans) if a == -1], xaxis="x2",
            marker={"color": DEFECT_COLOR_MAPPING["UNK"]}, ybins={"start": 0.00, "end": 1.00, "size": 0.01},
            name='No-Label',
        )
    )

    fig.add_trace(
        go.Histogram(
            x=[p for p, a in zip(m1_probs, m1_ans) if a == 1], yaxis="y2",
            marker={"color": DEFECT_COLOR_MAPPING["D"]}, xbins={"start": 0.00, "end": 1.00, "size": 0.01},
            name='Defects',
        )
    )
    fig.add_trace(
        go.Histogram(
            x=[p for p, a in zip(m1_probs, m1_ans) if a == 0], yaxis="y2",
            marker={"color": DEFECT_COLOR_MAPPING["ND"]}, xbins={"start": 0.00, "end": 1.00, "size": 0.01},
            name='Non-defects',
        )
    )
    fig.add_trace(
        go.Histogram(
            x=[p for p, a in zip(m2_probs, m2_ans) if a == -1], yaxis="y2",
            marker={"color": DEFECT_COLOR_MAPPING["UNK"]}, ybins={"start": 0.00, "end": 1.00, "size": 0.01},
            name='No-Label',
        )
    )

    fig.update_layout(
        title="Model Comparision Chart",
        autosize=False,
        xaxis={"zeroline": False, "domain": [0, 0.85], "showgrid": False, "title": "Model 1 (Base)"},
        yaxis={"zeroline": False, "domain": [0, 0.85], "showgrid": False, "title": "Model 2 (Candidate)"},
        xaxis2={"zeroline": False, "domain": [0.85, 1], "showgrid": False, "title": "Model 2"},
        yaxis2={"zeroline": False, "domain": [0.85, 1], "showgrid": False, "title": "Model 1"},
        height=600,
        width=600,
        bargap=0,
        barmode="stack",
        hovermode="closest",
        showlegend=False,
    )

    return fig


def plot_roc(roc_data: list[tuple[str, Any, float, float]]):
    fig = go.Figure()

    for curve_data in roc_data:
        model_name, data, selected_threshold, inference_threshold = curve_data
        fpr, tpr, threshold = data
        tnr = 1 - fpr

        # Draw the main curve
        fig.add_trace(go.Scatter(x=tnr, y=tpr, mode="lines", name=model_name, hoverinfo='text+name',
                                 hovertext=[f'Capture rate: {x}<br>Filter Rate: {y}<br>Threshold: {z}' for x, y, z in zip(tpr, tnr, threshold)]))

        # Add a diagonal grey dotted-line
        fig.add_trace(go.Scatter(x=[1, 0], y=[0, 1], mode="lines", line={"dash": "dash", "color": "grey"}, name="Random"))

        # Search for index of highest FR when CR = 1
        cr_one_indices = np.where(tpr == 1.0)[0]
        highest_fr_idx = cr_one_indices[0]
        for idx in cr_one_indices:
            if tnr[idx] > tnr[highest_fr_idx]:
                highest_fr_idx = idx
        logger.debug(f'Index of highest filter rate when capture rate is 100%: {highest_fr_idx}')

        # Draw highest FR when CR = 1
        fig.add_trace(go.Scatter(
            x=[tnr[highest_fr_idx]],
            y=[tpr[highest_fr_idx]],
            mode="markers",
            marker={"color": "blue", "size": 10},
            name="Highest Filter Rate at 100% Capture Rate",
            hoverinfo='text',
            hovertext=f"Highest Filter Rate at 100% Capture Rate<br> Capture rate: {tpr[highest_fr_idx]}<br>Filter Rate: {tnr[highest_fr_idx]}<br>Threshold: {threshold[highest_fr_idx]:.5f}",
        ))

        # Add annotation above the highest FR marker
        fig.add_annotation(
            x=tnr[highest_fr_idx],
            y=tpr[highest_fr_idx],
            text=f"{model_name} Threshold = {threshold[highest_fr_idx]:.5f} <br> Capture Rate: {tpr[highest_fr_idx]:.4f} <br> Filter Rate: {tnr[highest_fr_idx]:.4f}",
            showarrow=False,
            yshift=-30,
        )

        # Search for the marker whose threshold is equal or smaller than selected threshold.
        # Note: need to reverse because threshold is from 1 to 0.
        reversed_threshold = threshold[::-1]
        selected_idx = np.searchsorted(reversed_threshold, selected_threshold, side='left')
        selected_idx = len(threshold) - selected_idx
        if selected_idx == len(threshold):
            selected_idx -= 1

        # Draw marker for current selected model threshold
        fig.add_trace(go.Scatter(
            x=[tnr[selected_idx]],
            y=[tpr[selected_idx]],
            mode="markers",
            marker={"color": "red", "size": 10},
            name=f"Selected Threshold ({selected_threshold:.5f})",
            hoverinfo='text',
            hovertext=f"Selected Threshold<br>Capture rate: {tpr[selected_idx]}<br>Filter Rate: {tnr[selected_idx]}<br>Threshold: {selected_threshold:.5f}",
        ))

        # Add annotation above current selected model threshold
        fig.add_annotation(
            x=tnr[selected_idx],
            y=tpr[selected_idx],
            text=f"{model_name} Threshold = {selected_threshold:.5f} <br> Capture Rate: {tpr[selected_idx]:.4f} <br> Filter Rate: {tnr[selected_idx]:.4f}",
            showarrow=False,
            yshift=30,
        )

        # Draw inference threshold
        if selected_threshold != inference_threshold:
            infer_idx = np.searchsorted(reversed_threshold, inference_threshold, side='left')
            infer_idx = len(threshold) - infer_idx
            if infer_idx == len(threshold):
                infer_idx -= 1

            fig.add_trace(go.Scatter(
                x=[tnr[infer_idx]],
                y=[tpr[infer_idx]],
                mode="markers",
                marker={"color": "black", "size": 10},
                name=f"Inference ({inference_threshold:.5f})",
                hoverinfo='text',
                hovertext=f"Inference Threshold<br>Capture rate: {tpr[infer_idx]}<br>Filter Rate: {tnr[infer_idx]}<br>Threshold: {inference_threshold:.5f}",
            ))

            fig.add_annotation(
                x=tnr[infer_idx],
                y=tpr[infer_idx],
                text=f"{model_name} Inference threshold = {inference_threshold:.5f} <br> Capture Rate: {tpr[infer_idx]:.4f} <br> Filter Rate: {tnr[infer_idx]:.4f}",
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


def plot_prc(prc_data: list[tuple[str, Any, float]]):
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


def gen_lrf(model_id, output_dir, gen_lrf_type, threshold=None, top_k=None):
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
                rv_m1_threshold = st.number_input("Confidence threshold:", 0.0, 1.0, model_1_metadata['model_threshold'], 0.00001, format="%.5f",
                                                help="Probabilities above thershold will be considered as defects.", key='m1_threshold')
            with vr2_col4:
                rv_m2_threshold = st.number_input("Confidence threshold:", 0.0, 1.0, model_2_metadata['model_threshold'], 0.00001, format="%.5f",
                                                help="Probabilities above thershold will be considered as defects.", key='m2_threshold')
            with vr1_col5:
                gen_lrf("1", rv_m1_output_dir, st_gen_lrf_type, threshold=rv_m1_threshold)
            with vr2_col5:
                gen_lrf("2", rv_m2_output_dir, st_gen_lrf_type, threshold=rv_m2_threshold)

        # Draw 2D comparison chart
        with vr3_col1:
            st.plotly_chart(generate_2D_plot(model_1_raw_data, model_2_raw_data, rv_m1_threshold, rv_m2_threshold))

        with vr3_col2:
            # TODO: This should be done somewhere else
            if set(model_1_raw_data[2]) == {-1} or set(model_2_raw_data[2]) == {-1}:
                # All data is unlabeled
                st.markdown("##### All data is unlabeled! Skipping chart.")

                # If all data is unlabeled, just calculate the filter rates
                defect_list_1, prob_list_1, _ = model_1_raw_data
                total_defects_1 = len(defect_list_1)
                filtered_count_1 = len([p for p in prob_list_1 if p < rv_m1_threshold])
                st.success(f'Model 1: False Filter Rate is {filtered_count_1/total_defects_1:.4f} at selected threshold ({rv_m1_threshold:.5f})')

                defect_list_2, prob_list_2, _ = model_2_raw_data
                total_defects_2 = len(defect_list_2)
                filtered_count_2 = len([p for p in prob_list_2 if p < rv_m2_threshold])
                st.success(f'Model 2: False Filter Rate is {filtered_count_2/total_defects_2:.4f} at selected threshold ({rv_m2_threshold:.5f})')

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
                rv_m1_threshold = st.number_input("Confidence threshold:", 0.0, 1.0, model_1_metadata['model_threshold'], 0.00001, format="%.5f",
                                                help="Probabilities above thershold will be considered as defects.", key='m1_threshold')
            with vr1_col5:
                gen_lrf("1", rv_m1_output_dir, st_gen_lrf_type, threshold=rv_m1_threshold)

        # Draw 1D comparison chart
        with vr3_col1:
            st.plotly_chart(generate_1D_plot(model_1_raw_data, rv_m1_threshold))

        with vr3_col2:
            # TODO: This should be done somewhere else
            if set(model_1_raw_data[2]) == {-1}:
                # All data is unlabeled
                st.markdown("##### All data is unlabeled! Skipping chart.")

                # If no ROC, just calculate filter rate
                defect_list, prob_list, _ = model_1_raw_data
                total_defects = len(defect_list)
                filtered_count = len([p for p in prob_list if p < rv_m1_threshold])
                st.success(f'False Filter Rate is {filtered_count/total_defects:.4f} at selected threshold ({rv_m1_threshold:.5f})')

            else:
                model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, return_curve=True)
                st.plotly_chart(plot_roc([("Model 1", model_1_roc_data, rv_m1_threshold, model_1_metadata['model_threshold'])]))

                # model_1_prc_data = helper.get_prc_data(rv_m1_output_dir, return_curve=True)
                # st.plotly_chart(plot_prc([("Model 1", model_1_prc_data, rv_m1_threshold)]))
    else:
        pass
