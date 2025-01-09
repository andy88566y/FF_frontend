import numpy as np
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def model_data(db_path: str) -> dict:
    defect_id_list = helper.get_defect_id(db_path)
    probability_list = helper.get_probability(db_path, defect_id_list)
    answer_list = helper.get_answer(db_path, defect_id_list)

    data = {
        defect_id_list[i] : {
            "db_path" : db_path,
            "defect_id" : defect_id_list[i],
            "probability" : probability_list[i],
            "answer" : answer_list[i],
        }
        for i in range(len(defect_id_list))
    }
    return data


def generate_2D_plot(model_1: dict, model_2: dict, slith1: float, slith2: float):

    probabilities_mod1 = [entry['probability'] for entry in model_1.values()]
    yes_prob_mod1 = [entry['probability'] for entry in model_1.values() if entry['answer']]
    no_prob_mod1 = [entry['probability'] for entry in model_1.values() if not entry['answer']]

    probabilities_mod2 = [entry['probability'] for entry in model_2.values()]
    yes_prob_mod2 = [entry['probability'] for entry in model_2.values() if entry['answer']]
    no_prob_mod2 = [entry['probability'] for entry in model_2.values() if not entry['answer']]

    # true / false to be determined

    threshold_1, threshold_2 = slith1, slith2

    answer_1 = [entry['answer'] for entry in model_1.values()]
    answer_2 = [entry['answer'] for entry in model_2.values()]

    indices = [entry["defect_id"] for entry in model_1.values()]

    colors = ['rgba(0, 255, 0, 0.3)' if a1 and a2 else
              'rgba(255, 0, 0, 0.3)' if not a1 and not a2 else
              'blue' for a1, a2 in zip(answer_1, answer_2)]

    # 2D scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x = probabilities_mod1,
            y = probabilities_mod2,
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = {
                "color": colors,
                "size": 5,
            },
            text=[f'Index: {index}' for index in indices],
            hoverinfo='text',
            hovertemplate='%{text}<br>Model 1 Pred: %{x}<br>Model 2 Pred: %{y}'
        ))


    # threshold plot
    fig.add_shape(type='line', x0= threshold_1, x1=threshold_1, y0=0, y1=1, xref='x', yref='paper',
                  line={"color": 'Red', "width": 2, "dash": "dash"})

    fig.add_shape(type='line', x0= 0, x1= 1, y0=threshold_2, y1=threshold_2, xref='paper', yref='y',
                  line={"color": 'Red', "width": 2, "dash": "dash"})

    # histograms
    fig.add_trace(go.Histogram(
        y = yes_prob_mod2,
        xaxis = 'x2',
        marker = {
            "color": 'olivedrab'
        },
        ybins={"start": 0.00, "end": 1.00, "size": 0.01}
    ))
    fig.add_trace(go.Histogram(
        y = no_prob_mod2,
        xaxis = 'x2',
        marker = {
            "color": 'darkred'
        },
        ybins={"start": 0.00, "end": 1.00, "size": 0.01}
    ))

    fig.add_trace(go.Histogram(
        x = yes_prob_mod1,
        yaxis = 'y2',
        marker = {
            "color": 'olivedrab'
        },
        xbins={"start": 0.00, "end": 1.00, "size": 0.01}
    ))

    fig.add_trace(go.Histogram(
        x = no_prob_mod1,
        yaxis = 'y2',
        marker = {
            "color": 'darkred'
        },
        xbins={"start": 0.00, "end": 1.00, "size": 0.01}
    ))

    fig.update_layout(
        autosize = False,
        xaxis = {
            "zeroline": False,
            "domain": [0,0.85],
            "showgrid": False,
            "title": 'Model:1'
        },
        yaxis = {
            "zeroline": False,
            "domain": [0,0.85],
            "showgrid": False,
            "title": 'Model:2'
        },
        xaxis2 = {
            "zeroline": False,
            "domain": [0,0.85],
            "showgrid": False,
            "title": 'Model:1'
        },
        yaxis2 = {
            "zeroline": False,
            "domain": [0,0.85],
            "showgrid": False,
            "title": 'Model:2'
        },
        height = 600,
        width = 600,
        bargap = 0,
        barmode = 'stack',
        hovermode = 'closest',
        showlegend = False
    )

    return fig


def generate_1D_plot(model: dict, slith: float):
    true_probs = [entry['probability'] for entry in model.values() if entry['answer']]
    false_probs = [entry['probability'] for entry in model.values() if not entry['answer']]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x = true_probs,
        yaxis = 'y2',
        marker = {
            "color": 'olivedrab'
        },
        xbins={"start": 0.00, "end": 1.00, "size": 0.01}
    ))

    fig.add_trace(go.Histogram(
        x = false_probs,
        yaxis = 'y2',
        marker = {
            "color": 'darkred'
        },
        xbins={"start": 0.00, "end": 1.00, "size": 0.01}
    ))

    fig.add_shape(type='line', x0= slith, x1=slith, y0=0, y1=1, xref='x', yref='paper',
                  line={"color": 'Red', "width": 2, "dash": "dash"})

    fig.update_layout(
        barmode='stack',
        xaxis_title='Probabilities',
        yaxis_title='Frequency',
        title='Stacked Histogram by Probabilities'
    )

    return fig


def plot_init_prc(prc_data_ndarray, slith: float):

    precision, recall, threshold = prc_data_ndarray
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = recall, y = precision, mode = 'lines', name = 'Model 1'))

    if slith is not None:
        idx = (np.abs(threshold - slith)).argmin()
        threshold_trace = go.Scatter(
            x=[recall[idx]],
            y=[precision[idx]],
            mode='markers',
            marker={"color": 'red', "size": 10},
            name=f'Threshold = {threshold[idx]:.2f}'
        )
        fig.add_trace(threshold_trace)
        fig.add_annotation(
            x = recall[idx],
            y = precision[idx],
            text = f"Model 1 Threshold = {threshold[idx]:.2f} <br> Recall: {recall[idx]:.4f} <br> Precision: {precision[idx]:.4f}",
            showarrow = True,
            arrowhead = 2
        )

    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend_title='Models',
        template='plotly_white',
        showlegend = True,
        xaxis={"range": [0.8, 1.05]},
        yaxis={"range": [0.8, 1.05]},
    )

    return fig

def update_prc(org_fig, prc_data_ndarray, slith: float):
    fig = org_fig
    precision, recall, threshold = prc_data_ndarray

    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Model 2'))

    if slith is not None:
        idx = (np.abs(threshold - slith)).argmin()
        threshold_trace = go.Scatter(
            x=[recall[idx]],
            y=[precision[idx]],
            mode='markers',
            marker={"color": 'red', "size": 10},
            name=f'Threshold = {threshold[idx]:.2f}'
        )
        fig.add_trace(threshold_trace)

        fig.add_annotation(
            x = recall[idx],
            y = precision[idx],
            text = f"Model 2 Threshold = {threshold[idx]:.2f} <br> Recall: {recall[idx]:.4f} <br> Precision: {precision[idx]:.4f}",
            showarrow = True,
            arrowhead = 2
        )

    return fig


def plot_init_roc(roc_data_ndarray, slith: float):
    fpr, tpr, threshold = roc_data_ndarray
    fig = go.Figure()
    fig.add_trace(go.Scatter(x= fpr, y= tpr, mode='lines', name='Model 1'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line={"dash": 'dash'}))

    if slith is not None:
        idx = (np.abs(threshold - slith)).argmin()
        threshold_trace = go.Scatter(
            x=[fpr[idx]],
            y=[tpr[idx]],
            mode='markers',
            marker={"color": 'red', "size": 10},
            name=f'Threshold = {threshold[idx]:.2f}'
        )
        fig.add_trace(threshold_trace)

        fig.add_annotation(
            x = fpr[idx],
            y = tpr[idx],
            text = f"Model 1 Threshold = {threshold[idx]:.2f} <br> Sensitivity: {fpr[idx]:.4f} <br> 1-Specificity: {tpr[idx]:.4f}",
            showarrow = True,
            arrowhead = 2
        )

    # Update layout
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='1-Specificity',
        yaxis_title='Sensitivity',
        legend_title='Models',
        template='plotly_white',
        xaxis={"range": [-0.05, 0.25]},
        yaxis={"range": [0.8, 1.05]},
    )

    return fig

def update_roc(org_fig, roc_data_ndarray, slith: float):
    fig = org_fig
    fpr, tpr, threshold = roc_data_ndarray
    fig.add_trace(go.Scatter(x= fpr, y=tpr, mode='lines', name='Model 2'))
    if slith is not None:
        idx = (np.abs(threshold - slith)).argmin()
        threshold_trace = go.Scatter(
            x=[fpr[idx]],
            y=[tpr[idx]],
            mode='markers',
            marker={"color": 'red', "size": 10},
            name=f'Threshold = {threshold[idx]:.2f}'
        )
        fig.add_trace(threshold_trace)
        fig.add_annotation(
            x = fpr[idx],
            y = tpr[idx],
            text = f"Model 2 Threshold = {threshold[idx]:.2f} <br> Sensitivity: {fpr[idx]:.4f} <br> 1-Specificity: {tpr[idx]:.4f}",
            showarrow = True,
            arrowhead = 2
        )
    return fig

def app() -> None:
    logger.debug("Model Comparison Dashboard")
    st.title("Model Comparison Dashboard")

    st.header("Import Models with Database Path")
    dir_model_1 = st.text_input('Model 1', value='', help='First Model Database Directory')
    dir_model_2 = st.text_input('Model 2 (optional)', value='', help='Second Model Database Directory')

    slider_threshold_1 = st.slider("Select confidence threshold for model 1:", 0.0, 1.0, 0.5)
    st.caption(f"Probabilities above :blue[{slider_threshold_1}] in Model 1 will be considered defects.")
    slider_threshold_2 = st.slider("Select confidence threshold for model 2:", 0.0, 1.0, 0.5)
    st.caption(f"Probabilities above :blue[{slider_threshold_2}] in Model 2 will be considered defects.")

    if st.button("Visualize model", type = 'primary'):
        if dir_model_1 and dir_model_2:
            model_1 = model_data(dir_model_1)
            model_2 = model_data(dir_model_2)

            fig_2d = generate_2D_plot(model_1, model_2, slider_threshold_1, slider_threshold_2)
            st.plotly_chart(fig_2d)
            helper.gap(2)

            prc_data_ndarray_1 = helper.get_prc_data(db_path = dir_model_1, return_curve = True)
            prc_data_ndarray_2 = helper.get_prc_data(db_path = dir_model_2, return_curve = True)

            fig_prc = plot_init_prc(prc_data_ndarray_1, slider_threshold_1)
            update_prc(fig_prc, prc_data_ndarray_2, slider_threshold_2)
            st.plotly_chart(fig_prc)

            roc_data_ndarray_1 = helper.get_roc_data(db_path = dir_model_1, return_curve = True)
            roc_data_ndarray_2 = helper.get_roc_data(db_path = dir_model_2, return_curve = True)

            fig_roc = plot_init_roc(roc_data_ndarray_1, slider_threshold_1)
            update_roc(fig_roc, roc_data_ndarray_2, slider_threshold_2)
            st.plotly_chart(fig_roc)

        else:
            if dir_model_1 and not dir_model_2:
                dir_model = dir_model_1
                model = model_data(dir_model_1)
                slider_threshold = slider_threshold_1
            else:
                dir_model = dir_model_2
                model = model_data(dir_model_2)
                slider_threshold = slider_threshold_2

            fig_1d = generate_1D_plot(model, slider_threshold)
            st.plotly_chart(fig_1d)

            prc_data_ndarray = helper.get_prc_data(db_path = dir_model, return_curve = True)
            fig_prc = plot_init_prc(prc_data_ndarray, slider_threshold)
            st.plotly_chart(fig_prc)

            roc_data_ndarray = helper.get_roc_data(db_path = dir_model, return_curve = True)
            fig_roc = plot_init_roc(roc_data_ndarray, slider_threshold)
            st.plotly_chart(fig_roc)
