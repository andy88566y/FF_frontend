from typing import Any
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper
from ltt_ff_frontend.constant import DEFAULT_THRESHOLD


DEFECT_COLOR_MAPPING = {
    "D": "darkred",
    "ND": "olivedrab",
    "UNK": "blue",
}

def get_model_data(output_dir: str, lot_id: str, model_name: str):
    defect_id_list = helper.get_defect_id(output_dir, lot_id, model_name)
    probability_list = helper.get_probability(output_dir, lot_id, model_name, defect_id_list)
    answer_list = helper.get_answer(output_dir, lot_id, model_name, defect_id_list)
    return (defect_id_list, probability_list, answer_list)


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
        xaxis={"zeroline": False, "domain": [0, 0.85], "showgrid": False, "title": "Model 1"},
        yaxis={"zeroline": False, "domain": [0, 0.85], "showgrid": False, "title": "Model 2"},
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


def plot_roc(roc_data: list[tuple[str, Any, float]]):
    fig = go.Figure()

    for curve_data in roc_data:
        model_name, data, model_threshold = curve_data
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
            hovertext=f"Capture rate: {tpr[highest_fr_idx]}<br>Filter Rate: {tnr[highest_fr_idx]}",
        ))

        # Add annotation above the highest FR marker
        fig.add_annotation(
            x=tnr[highest_fr_idx],
            y=tpr[highest_fr_idx],
            text=f"{model_name} Threshold = {threshold[highest_fr_idx]:.5f} <br> Capture Rate: {tpr[highest_fr_idx]:.4f} <br> Filter Rate: {tnr[highest_fr_idx]:.4f}",
            showarrow=False,
            yshift=-30,
        )

        # Draw marker for current selected model threshold
        selected_idx = np.argmin(np.abs(threshold - model_threshold))
        fig.add_trace(go.Scatter(
            x=[tnr[selected_idx]],
            y=[tpr[selected_idx]],
            mode="markers",
            marker={"color": "red", "size": 10},
            name=f"Selected Threshold ({threshold[selected_idx]:.5f})",
            hovertext=f"Capture rate: {tpr[selected_idx]}<br>Filter Rate: {tnr[selected_idx]}",
        ))

        # Add annotation above current selected model threshold
        fig.add_annotation(
            x=tnr[selected_idx],
            y=tpr[selected_idx],
            text=f"{model_name} Threshold = {threshold[selected_idx]:.5f} <br> Capture Rate: {tpr[selected_idx]:.4f} <br> Filter Rate: {tnr[selected_idx]:.4f}",
            showarrow=False,
            yshift=30,
        )

        if model_threshold != DEFAULT_THRESHOLD:
            # Draw default 0.174 threshold
            default_idx = np.argmin(np.abs(threshold - 0.174))
            fig.add_trace(go.Scatter(
                x=[tnr[default_idx]],
                y=[tpr[default_idx]],
                mode="markers",
                marker={"color": "black", "size": 10},
                name=f"Default ({threshold[default_idx]:.2f})",
                hovertext=f"Capture rate: {tpr[default_idx]}<br>Filter Rate: {tnr[default_idx]}",
            ))

            fig.add_annotation(
                x=tnr[default_idx],
                y=tpr[default_idx],
                text=f"{model_name} Default threshold = {threshold[default_idx]:.5f} <br> Capture Rate: {tpr[default_idx]:.4f} <br> Filter Rate: {tnr[default_idx]:.4f}",
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


def app() -> None:
    logger.debug("Loading Result Viewer...")
    st.title("False Filter Result Viewer")
    st.caption("Visualize False Filter Result (Model 2 fields are optional)")

    r1_col1, r1_col2, r1_col3, r1_col4, r1_col5 = st.columns([3, 2, 2, 2, 2])
    r2_col1, r2_col2, r2_col3, r2_col4, r2_col5 = st.columns([3, 2, 2, 2, 2])

    output_dir_default = "/mnt/dbpc/xxx"

    with r1_col1:
        rv_m1_output_dir = st.text_input("Model 1 Result Directory", value=output_dir_default)
    with r2_col1:
        rv_m2_output_dir = st.text_input("Model 2 Result Directory", value=output_dir_default)

    with r1_col2:
        rv_m1_lot_id = st.text_input("Model 1 Lot ID", value="")
    with r2_col2:
        rv_m2_lot_id = st.text_input("Model 2 Lot ID", value="")

    with r1_col3:
        rv_m1_model_name = st.selectbox("Model 1 Model Name", options=helper.get_base_models(), index=0,
                                        format_func=helper.format_model_name)
    with r2_col3:
        rv_m2_model_name = st.selectbox("Model 2 Model Name", options=helper.get_base_models(), index=0,
                                        format_func=helper.format_model_name)

    with r1_col4:
        rv_m1_threshold = st.number_input("Confidence threshold:", 0.0, 1.0, DEFAULT_THRESHOLD, 0.00001, format="%.5f", help="Probabilities above thershold will be considered as defects.", key='m1_threshold')
    with r2_col4:
        rv_m2_threshold = st.number_input("Confidence threshold:", 0.0, 1.0, DEFAULT_THRESHOLD, 0.00001, format="%.5f", help="Probabilities above thershold will be considered as defects.", key='m2_threshold')

    with r1_col5:
        if st.button("Generate new Model 1 .lrf"):

            if rv_m1_output_dir == '' or rv_m1_lot_id == '':
                logger.error('Missing Model 1 Result Directory or Lot ID.')
                st.error('Missing Model 1 Result Directory or Lot ID.')

            request = helper.request_lrf(output_dir=rv_m1_output_dir, lot_id=rv_m1_lot_id, model_name=rv_m1_model_name,
                                         confidence_threshold=rv_m1_threshold)

            if request.json().get('status') == 'error':
                code = request.json().get('code')
                message = request.json().get('message')
                st.error(f'.lrf file not generated!\nError code: {code}\nError message: {message}')
                logger.error(f'.lrf file not generated!\nError code: {code}\nError message: {message}')
            else:
                # TODO: Check file generated
                st.success(f'New .lrf file (threshold: {rv_m1_threshold}) generated at {rv_m1_output_dir}!')
                logger.info(f'New .lrf file (threshold: {rv_m1_threshold}) generated at {rv_m1_output_dir}!')
    with r2_col5:
        if st.button("Generate new Model 2 .lrf"):

            if rv_m2_output_dir == '' or rv_m2_lot_id == '':
                logger.error('Missing Model 2 Result Directory or Lot ID.')
                st.error('Missing Model 2 Result Directory or Lot ID.')

            request = helper.request_lrf(output_dir=rv_m2_output_dir, lot_id=rv_m2_lot_id, model_name=rv_m2_model_name,
                                         confidence_threshold=rv_m2_threshold)

            if request.json().get('status') == 'error':
                code = request.json().get('code')
                message = request.json().get('message')
                st.error(f'.lrf file not generated!\nError code: {code}\nError message: {message}')
                logger.error(f'.lrf file not generated!\nError code: {code}\nError message: {message}')
            else:
                # TODO: Check file generated
                st.success(f'New .lrf file (threshold: {rv_m2_threshold}) generated at {rv_m2_output_dir}!')
                logger.info(f'New .lrf file (threshold: {rv_m2_threshold}) generated at {rv_m2_output_dir}!')

    st.divider()

    if st.button("Visualize Result", type="primary"):

        vr1_col1, vr1_col2 = st.columns(2)

        invalid_input = [output_dir_default, '']

        if rv_m1_output_dir not in invalid_input and rv_m2_output_dir not in invalid_input and rv_m1_lot_id not in invalid_input and rv_m2_lot_id not in invalid_input:

            if rv_m1_lot_id != rv_m2_lot_id:
                st.error(f'Lot IDs do not match!  \nModel 1 lot ID: {rv_m1_lot_id}  \nModel 2 lot ID: {rv_m2_lot_id}')
                return

            # Draw 2D comparison chart
            model_1_raw_data = get_model_data(rv_m1_output_dir, rv_m1_lot_id, rv_m1_model_name)
            model_2_raw_data = get_model_data(rv_m2_output_dir, rv_m2_lot_id, rv_m2_model_name)

            with vr1_col1:
                st.plotly_chart(generate_2D_plot(model_1_raw_data, model_2_raw_data, rv_m1_threshold, rv_m2_threshold))

            with vr1_col2:
                # TODO: This should be done somewhere else
                if set(model_1_raw_data[2]) == {-1} or set(model_2_raw_data[2]) == {-1}:
                    # All data is unlabeled
                    st.markdown("##### All data is unlabeled! Skipping chart.")
                else:

                    model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, rv_m1_lot_id, rv_m1_model_name, return_curve=True)
                    model_2_roc_data = helper.get_roc_data(rv_m2_output_dir, rv_m2_lot_id, rv_m2_model_name, return_curve=True)
                    st.plotly_chart(plot_roc([("Model 1", model_1_roc_data, rv_m1_threshold), ("Model 2", model_2_roc_data, rv_m2_threshold)]))

                    # model_1_prc_data = helper.get_prc_data(rv_m1_output_dir, rv_m1_lot_id, rv_m1_model_name, return_curve=True)
                    # model_2_prc_data = helper.get_prc_data(rv_m2_output_dir, rv_m2_lot_id, rv_m2_model_name, return_curve=True)
                    # st.plotly_chart(plot_prc([("Model 1", model_1_prc_data, rv_m1_threshold), ("Model 2", model_2_prc_data, rv_m2_threshold)]))

        else:
            if rv_m1_output_dir == '':
                st.error(f'Model 1 Result Directory input field is empty!')
                return
            elif rv_m1_lot_id == '':
                st.error(f'Model 1 Lot ID input field is empty!')
                return


            # Draw 1D comparison chart
            model_1_raw_data = get_model_data(rv_m1_output_dir, rv_m1_lot_id, rv_m1_model_name)

            with vr1_col1:
                st.plotly_chart(generate_1D_plot(model_1_raw_data, rv_m1_threshold))

            with vr1_col2:
                # TODO: This should be done somewhere else
                if set(model_1_raw_data[2]) == {-1}:
                    # All data is unlabeled
                    st.markdown("##### All data is unlabeled! Skipping chart.")
                else:
                    model_1_roc_data = helper.get_roc_data(rv_m1_output_dir, rv_m1_lot_id, rv_m1_model_name, return_curve=True)
                    st.plotly_chart(plot_roc([("Model 1", model_1_roc_data, rv_m1_threshold)]))

                    # model_1_prc_data = helper.get_prc_data(rv_m1_output_dir, rv_m1_lot_id, rv_m1_model_name, return_curve=True)
                    # st.plotly_chart(plot_prc([("Model 1", model_1_prc_data, rv_m1_threshold)]))
