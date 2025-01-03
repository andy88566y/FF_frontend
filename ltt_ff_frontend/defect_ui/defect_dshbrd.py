import glob
import os

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random

import requests
import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import API_ROOT
from ltt_ff_frontend.defect_ui import defect_ui_helper as helper

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def dummy_data_gen(seed: int) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    size = 7500
    def m_shape(size):
        dist1 = np.random.normal(loc=random.uniform(random.uniform(0.10,0.35),random.uniform(0.25,0.55)), scale=0.15, size=size//2)
        dist2 = np.random.normal(loc=random.uniform(random.uniform(0.35,0.65),random.uniform(0.55,0.85)), scale=0.15, size=size//2)
        combined_dist = np.concatenate([dist1, dist2])
        combined_dist = np.clip(combined_dist, 0, 1)
        return combined_dist
    combined_dist = m_shape(size)
    data ={ 
        i: {
            "seed": seed,
            "index": i,
            "probability": combined_dist[i],
            "answer": random.choice([True, False]),
            "threshold" : float(0.5)
        }
        for i in range(size)
    }
    return data


def generate_2D_plot(model_1: dict, model_2: dict, slith1: float, slith2: float) -> None :
    # t = np.linspace(-1, 1.2, 2000)

    probabilities_mod1 = [entry['probability'] for entry in model_1.values()]
    probabilities_mod2 = [entry['probability'] for entry in model_2.values()]
    seed_1 = [entry['seed'] for entry in model_1.values()][0]
    seed_2 = [entry['seed'] for entry in model_2.values()][0]
    thresmod1 = [entry['threshold'] for entry in model_1.values()]
    # threshold_1 = thresmod1[0]
    threshold_1 = slith1
    # assert(threshold_1,0.5)
    thresmod2 = [entry['threshold'] for entry in model_2.values()]
    # threshold_2 = thresmod2[0]
    threshold_2 = slith2
    # assert(threshold_2,0.5)
    #use answer_1 as reference, default assert answer1 equals answer2
    # base & candidate    
    answer_1 = [entry['answer'] for entry in model_1.values()]
    answer_2 = [entry['answer'] for entry in model_1.values()]

    indices = [entry["index"] for entry in model_1.values()]

    colors = ['rgba(0, 255, 0, 0.3)' if answer_1 and answer_2 else 'rgba(255, 0, 0, 0.3)' 
              for answer_1, answer_2 in zip(answer_1, answer_2)]
    
    # 2D scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x = probabilities_mod1,
            y = probabilities_mod2,
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = dict(
                color = colors,
                size = 5
            ),
            text=[f'Index: {index}' for index in indices],
            hoverinfo='text',
            hovertemplate='%{text}<br>Model 1 Pred: %{x}<br>Model 2 Pred: %{y}'
        ))

    # threshold plot
    fig.add_shape(type='line',
              x0= threshold_1, x1=threshold_1, y0=0, y1=1,
              xref='x', yref='paper',
              line=dict(color='Red', width=2, dash='dash'))

    fig.add_shape(type='line',
              x0= 0, x1= 1, y0=threshold_2, y1=threshold_2,
              xref='paper', yref='y',
              line=dict(color='Red', width=2, dash='dash'))

    # histograms
    fig.add_trace(go.Histogram(
            y = probabilities_mod2,
            xaxis = 'x2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            ),
            ybins=dict(start=0.00, end=1.00, size=0.01)
        ))
    fig.add_trace(go.Histogram(
            x = probabilities_mod1,
            yaxis = 'y2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            ),
            xbins=dict(start=0.00, end=1.00, size=0.01)
        ))
    
    seed_1 = [entry['seed'] for entry in model_1.values()][0]
    seed_2 = [entry['seed'] for entry in model_2.values()][0]

    fig.update_layout(
        autosize = False,
        xaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False,
            title = f'Model:{seed_1}'
        ),
        yaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False,
            title = f'Model:{seed_2}'

        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False,
            title = f'Model:{seed_1}'
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False,
            title = f'Model:{seed_2}'
        ),
        height = 600,
        width = 600,
        bargap = 0,
        hovermode = 'closest',
        showlegend = False
    )

    return fig

# def get_pr_curve

def prep_plot(model_1: dict, model_2: dict) -> None:

    # convert to dataframe
    df_model_1 = pd.DataFrame.from_dict(model_1, orient='index')
    df_model_2 = pd.DataFrame.from_dict(model_2, orient='index')

    #  true and pred
    y_true_mod1 = df_model_1['answer'].values
    y_pred_mod1 = df_model_2['probability'].values

    y_true_mod2 = df_model_2['answer'].values
    y_pred_mod2 = df_model_2['probability'].values

    return y_true_mod1, y_true_mod2, y_pred_mod1, y_pred_mod2

def plot_prc(model_1: dict, model_2: dict):
    y_true_mod1, y_true_mod2, y_pred_mod1, y_pred_mod2 = prep_plot(model_1, model_2)
    precision_mod1, recall_mod1, threshold_1 = precision_recall_curve(y_true_mod1, y_pred_mod1)
    precision_mod2, recall_mod2, threshold_2 = precision_recall_curve(y_true_mod2, y_pred_mod2)

    # Create the plotly figure
    fig = go.Figure()
    
    # Add traces for both models
    fig.add_trace(go.Scatter(x=recall_mod1, y=precision_mod1, mode='lines', name='Model 1'))
    fig.add_trace(go.Scatter(x=recall_mod2, y=precision_mod2, mode='lines', name='Model 2'))
    
    # Update layoutff
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend_title='Models',
        template='plotly_white'
    )
    return fig

def plot_roc(model_1: dict, model_2: dict):
    y_true_mod1, y_true_mod2, y_pred_mod1, y_pred_mod2 = prep_plot(model_1, model_2)
    
    fpr_mod1, tpr_mod1, threshold_1 = metrics.roc_curve(y_true_mod1,y_pred_mod1, pos_label=1)
    fpr_mod2, tpr_mod2, threshold_2 = metrics.roc_curve(y_true_mod2,y_pred_mod2, pos_label=1)

    # Create the plotly figure
    fig = go.Figure()
    
    # Add traces for both models
    fig.add_trace(go.Scatter(x= fpr_mod1, y=1-tpr_mod1, mode='lines', name='Model 1'))
    fig.add_trace(go.Scatter(x=fpr_mod2, y=1-tpr_mod2, mode='lines', name='Model 2'))

    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], mode='lines', name='Random Classifier', line=dict(dash='dash')))
    
    # Update layoutff
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend_title='Models',
        template='plotly_white'
    )
    return fig

def app() -> None:
    logger.debug("Opening Defect Dashboard Page")
    st.title("Defect Probability Visualization Dashboard")
    st.caption("Pages that contain three dashboards, 2D model comparison chart")

    st.header("Select Models")
    modelsize = 10

    modelarr = [dummy_data_gen(i) for i in range(modelsize)]
    model_name = [list(model.values())[0]['seed'] for model in modelarr]    

    # initialize session state for the selected model
    if 'selected_model_1' not in st.session_state:
        st.session_state.selected_model_1 = model_name[0]
    if 'selected_model_2' not in st.session_state:
        st.session_state.selected_model_2 = model_name[1]


    # Select box
    selected_model_1_name = st.selectbox(
        'Select your first model input', model_name, index = model_name.index(st.session_state.selected_model_1))
    selected_model_2_name = st.selectbox(
        'Select your second model input', model_name, index = model_name.index(st.session_state.selected_model_2))

    # update the session state
    st.session_state.selected_model_1 = selected_model_1_name
    st.session_state.selected_model_2 = selected_model_2_name

    # Get the selected models
    selected_model_1 = modelarr[model_name.index(selected_model_1_name)]
    selected_model_2 = modelarr[model_name.index(selected_model_2_name)]

    # 2D comparison
    slider_threshold_1 = st.slider("Select confidence threshold for model 1:", 0.0, 1.0, 0.5)
    st.caption(f"Probabilities above :blue[{slider_threshold_1}] in Model 1 will be considered defects.")
    helper.gap(2)

    slider_threshold_2 = st.slider("Select confidence threshold for model 2:", 0.0, 1.0, 0.5)
    st.caption(f"Probabilities above :blue[{slider_threshold_2}] in Model 2 will be considered defects.")

    fig_2d = generate_2D_plot(selected_model_1,selected_model_2, slider_threshold_1, slider_threshold_2)
    st.plotly_chart(fig_2d)
    helper.gap(2)
    prep_plot(selected_model_1, selected_model_2)

    fig_prc = plot_prc(selected_model_1, selected_model_2)
    st.plotly_chart(fig_prc)

    fig_roc = plot_roc(selected_model_1, selected_model_2)
    st.plotly_chart(fig_roc)

