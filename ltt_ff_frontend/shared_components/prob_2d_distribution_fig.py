import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import helper
from ltt_ff_frontend.shared_components.prob_distribution_fig import get_color_map


def gen(result_dir_1: str, result_dir_2: str) -> None:
    if helper.is_valid_output_dir(result_dir_1) and helper.is_valid_output_dir(result_dir_2):
        multi_lot_model_data_1 = api_helper.get_multilot_model_data(result_dir_1)
        multi_lot_model_data_2 = api_helper.get_multilot_model_data(result_dir_2)
        recipe_model_count_1 = multi_lot_model_data_1.model_metadata_list[0]["model_count"]
        recipe_model_count_2 = multi_lot_model_data_2.model_metadata_list[0]["model_count"]

        if recipe_model_count_1 == 1 and recipe_model_count_2 == 1:
            m1_threshold = multi_lot_model_data_1.model_metadata_list[0]['model_threshold_0']
            m2_threshold = multi_lot_model_data_2.model_metadata_list[0]['model_threshold_0']
            aggregated_model_data_1 = helper.aggregate_lists(
                (
                    multi_lot_model_data_1.defect_id_lists,
                    multi_lot_model_data_1.probability_lists,
                    multi_lot_model_data_1.answer_lists,
                ),
                multi_lot_model_data_1.model_metadata_list,
            )
            aggregated_model_data_2 = helper.aggregate_lists(
                (
                    multi_lot_model_data_2.defect_id_lists,
                    multi_lot_model_data_2.probability_lists,
                    multi_lot_model_data_2.answer_lists,
                ),
                multi_lot_model_data_2.model_metadata_list
            )
            col1, col2 = st.columns(2)
            with col1:
                input_m1_threshold = st.number_input(
                    label="Model 1 threshold:",
                    value=m1_threshold,
                    step=1e-5,
                    format="%.5f",
                    help="Probabilities below threshold will be considered as non-defects.",
                )
            with col2:
                input_m2_threshold = st.number_input(
                    label="Model 2 threshold:",
                    value=m2_threshold,
                    step=1e-5,
                    format="%.5f",
                    help="Probabilities below threshold will be considered as non-defects.",
                )

            _, plot_container, _ = st.columns([1,8,1])
            with plot_container:
                st.plotly_chart(
                    generate_fig(
                        aggregated_model_data_1,
                        aggregated_model_data_2,
                        input_m1_threshold,
                        input_m2_threshold,
                    )
                )

def generate_fig(
    aggregated_model_data_1: tuple[list[int], list[float], list[int], list[str]],
    aggregated_model_data_2: tuple[list[int], list[float], list[int], list[str]],
    m1_threshold: float,
    m2_threshold: float,
) -> go.Figure:
    m1_defect_ids, m1_probs, m1_ans, m1_lot_ids = aggregated_model_data_1
    m2_defect_ids, m2_probs, m2_ans, m2_lot_ids = aggregated_model_data_2
    assert sorted(m1_lot_ids) == sorted(m2_lot_ids), "Lot IDs Mismatch!"
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
