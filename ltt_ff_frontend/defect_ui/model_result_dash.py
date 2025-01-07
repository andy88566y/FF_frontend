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

# Modified as of 01/06

def dummy_data_gen(seed: int) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    size = 1500
    def m_shape(size):
        dist1 = np.random.normal(loc=random.uniform(random.uniform(0.20,0.35),random.uniform(0.25,0.55)), scale=0.10, size=size//2)
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
            "answer": random.choice([True,False]),
            "threshold" : float(0.5)
        }
        for i in range(size)
    }
    return data


def generate_2D_plot(model_1: dict, model_2: dict, slith1: float, slith2: float) -> None :
    # t = np.linspace(-1, 1.2, 2000)

    probabilities_mod1 = [entry['probability'] for entry in model_1.values()]
    yes_prob_mod1 = [entry['probability'] for entry in model_1.values() if entry['answer'] == True]
    no_prob_mod1 = [entry['probability'] for entry in model_1.values() if entry['answer'] == False]


    probabilities_mod2 = [entry['probability'] for entry in model_2.values()]
    yes_prob_mod2 = [entry['probability'] for entry in model_2.values() if entry['answer'] == True]
    no_prob_mod2 = [entry['probability'] for entry in model_2.values() if entry['answer'] == False]


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
    answer_2 = [entry['answer'] for entry in model_2.values()]

    indices = [entry["index"] for entry in model_1.values()]

    colors = ['rgba(0, 255, 0, 0.3)' if a1 == True and a2 == True else
            'rgba(255, 0, 0, 0.3)' if a1 == False and a2 == False else
            'rgba(241, 90, 34, 1)'
            for a1, a2 in zip(answer_1, answer_2)]

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
            y = yes_prob_mod2,
            xaxis = 'x2',
            marker = dict(
                color = 'olivedrab'
            ),
            ybins=dict(start=0.00, end=1.00, size=0.01)
        ))
    fig.add_trace(go.Histogram(
        y = no_prob_mod2,
        xaxis = 'x2',
        marker = dict(
            color = 'darkred'
        ),
        ybins=dict(start=0.00, end=1.00, size=0.01)
    ))




    fig.add_trace(go.Histogram(
            x = yes_prob_mod1,
            yaxis = 'y2',
            marker = dict(
                color = 'olivedrab'
            ),
            xbins=dict(start=0.00, end=1.00, size=0.01)
        ))

    fig.add_trace(go.Histogram(
            x = no_prob_mod1,
            yaxis = 'y2',
            marker = dict(
                color = 'darkred'
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
        barmode = 'stack',
        hovermode = 'closest',
        showlegend = False
    )

    return fig


def generate_1D_plot(model: dict, slith: float):

    probabilities = [entry['probability'] for entry in model.values()]
    indices = [entry["index"] for entry in model.values()]
    # answer_true = [entry[['answer'] = True] for entry in model.values()]
    # answer_false = [entry[['answer'] = False] for entry in model.values()]

    true_probs = [entry['probability'] for entry in model.values() if entry['answer'] == True]
    false_probs = [entry['probability'] for entry in model.values() if entry['answer'] == False]


    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x = true_probs,
        yaxis = 'y2',
        marker = dict(
            color = 'olivedrab'
        ),
        xbins=dict(start=0.00, end=1.00, size=0.01)
    ))

    fig.add_trace(go.Histogram(
    x = false_probs,
    yaxis = 'y2',
    marker = dict(
        color = 'darkred'
    ),
    xbins=dict(start=0.00, end=1.00, size=0.01)
    ))

    fig.add_shape(type='line',
              x0= slith, x1=slith, y0=0, y1=1,
              xref='x', yref='paper',
              line=dict(color='Red', width=2, dash='dash'))

    fig.update_layout(
    barmode='stack',
    xaxis_title='Probabilities',
    yaxis_title='Frequency',
    title='Stacked Histogram by Probabilities')

    return fig

def prep_plot(model: dict) -> None:
    df_model = pd.DataFrame.from_dict(model, orient='index')
    y_true = df_model['answer'].values
    y_pred = df_model['probability'].values
    return y_true,y_pred

def plot_init_prc(model: dict, slith: float):

    y_true, y_pred = prep_plot(model)
    precision, recall, threshold = precision_recall_curve(y_true,y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = recall, y = precision, mode = 'lines', name = 'Model 1'))

    if slith is not None:
        idx = (np.abs(threshold - slith)).argmin()
        threshold_trace = go.Scatter(
        x=[recall[idx]],
        y=[precision[idx]],
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'Threshold = {threshold[idx]:.2f}'
        )
    fig.add_trace(threshold_trace)

    fig.update_layout(
    title='Precision-Recall Curve',
    xaxis_title='Recall',
    yaxis_title='Precision',
    legend_title='Models',
    template='plotly_white',
    showlegend = True
    )
    return fig

def update_prc(org_fig, model: dict, slith: float):
    fig = org_fig

    y_true, y_pred = prep_plot(model)
    precision, recall, threshold = precision_recall_curve(y_true,y_pred)

    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Model 2'))

    if slith is not None:
        idx = (np.abs(threshold - slith)).argmin()
        threshold_trace = go.Scatter(
        x=[recall[idx]],
        y=[precision[idx]],
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'Threshold = {threshold[idx]:.2f}'
        )
    fig.add_trace(threshold_trace)

    return fig


def plot_init_roc(model: dict, slith: float):
    y_true, y_pred = prep_plot(model)
    fpr, tpr, threshold = metrics.roc_curve(y_true,y_pred, pos_label=1)
    fnr = 1-tpr

    fig = go.Figure()
    fig.add_trace(go.Scatter(x= fpr, y=fnr, mode='lines', name='Model 1'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], mode='lines', name='Random Classifier', line=dict(dash='dash')))

    if slith is not None:
        idx = (np.abs(threshold - slith)).argmin()
        threshold_trace = go.Scatter(
        x=[fpr[idx]],
        y=[fnr[idx]],
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'Threshold = {threshold[idx]:.2f}'
        )
    fig.add_trace(threshold_trace)


    # Update layoutff
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend_title='Models',
        template='plotly_white'
    )
    return fig

def update_roc(org_fig, model: dict, slith: float):
    fig = org_fig
    y_true, y_pred = prep_plot(model)
    fpr, tpr, threshold = metrics.roc_curve(y_true,y_pred, pos_label=1)
    fnr = 1-tpr

    fig.add_trace(go.Scatter(x= fpr, y=fnr, mode='lines', name='Model 2'))

    if slith is not None:
        idx = (np.abs(threshold - slith)).argmin()
        threshold_trace = go.Scatter(
        x=[fpr[idx]],
        y=[fnr[idx]],
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'Threshold = {threshold[idx]:.2f}'
        )
    fig.add_trace(threshold_trace)


    return fig



def app() -> None:
    logger.debug("Opening Model Result Dashboard")
    st.title("Model Result Dashboard")
    st.caption("Pages that contain three dashboards, 2D model comparison chart")

    st.header("Import Models with Database Path")
    dir_model_1 = st.text_input('Model 1', value='', help='Your First Model Database Directory.')
    dir_model_2 = st.text_input('Model 2 (optional)', value='', help='Your Second Model Database Directory.')



    slider_threshold_1 = st.slider("Select confidence threshold for model 1:", 0.0, 1.0, 0.5)
    st.caption(f"Probabilities above :blue[{slider_threshold_1}] in Model 1 will be considered defects.")
    slider_threshold_2 = st.slider("Select confidence threshold for model 2:", 0.0, 1.0, 0.5)
    st.caption(f"Probabilities above :blue[{slider_threshold_2}] in Model 2 will be considered defects.")

    # #dummy data
    # modelsize = 10
    # modelarr = [dummy_data_gen(i) for i in range(modelsize)]
    # model_name = [list(model.values())[0]['seed'] for model in modelarr]

    if st.button("Visualize model", type = 'primary'):

        if dir_model_1 and dir_model_2:
            dir_model_1 = int(dir_model_1)
            dir_model_2 = int(dir_model_2)
            model_1 = dummy_data_gen(dir_model_1)
            model_2 = dummy_data_gen(dir_model_2)


            fig_2d = generate_2D_plot(model_1, model_2, slider_threshold_1, slider_threshold_2)
            st.plotly_chart(fig_2d)
            helper.gap(2)

            fig_prc = plot_init_prc(model_1, slider_threshold_1)
            update_prc(fig_prc,model_2, slider_threshold_2)
            st.plotly_chart(fig_prc)

            fig_roc = plot_init_roc(model_1, slider_threshold_1)
            update_roc(fig_roc, model_2, slider_threshold_2)
            st.plotly_chart(fig_roc)

        else:
            if dir_model_1 and not dir_model_2:
                dir_model_1 = int(dir_model_1)
                model = dummy_data_gen(dir_model_1)
                slider_threshold = slider_threshold_1
            else:
                dir_model_2 = int(dir_model_2)
                model = dummy_data_gen(dir_model_2)
                slider_threshold = slider_threshold_2

            # generate 1-D stacked histogram
            fig_1d = generate_1D_plot(model, slider_threshold)
            st.plotly_chart(fig_1d)

            fig_prc = plot_init_prc(model, slider_threshold)
            st.plotly_chart(fig_prc)

            fig_roc = plot_init_roc(model,slider_threshold)
            st.plotly_chart(fig_roc)