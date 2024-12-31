import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def prep_dist_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Prepare the data needed to draw the defect distribution chart by grouping
    defect probabilities into 100 intervals, getting the count for each interval,
    and adding labels.

    Args:
        df: The dataframe containing inference results.

    Returns a dataframe that plotly can use to draw the defect distribution chart.
    '''
    df_count = df[['probabilities']].groupby(
            pd.cut(df['probabilities'], np.arange(0.00, 1.01, 0.01)), observed=False).count()

    labels = [f'{x*0.01:.2f}' for x in range(100)]
    df_count['labels'] = labels

    return df_count


def draw_defect_dist_chart(df_count: pd.DataFrame, slider_threshold: float) -> plotly.graph_objects.Figure:
    fig = px.histogram(df_count,
                        x="labels",
                        y='probabilities',
                        labels={
                            "labels": "Probability of defect",
                            "probabilities": "Count"
                        },
                        nbins=100)
    fig.update_layout(bargap=0.1)
    fig.add_vline(x=slider_threshold, line_dash = 'dash', line_color = 'firebrick')

    return fig


def app() -> None:
    logger.debug("Opening Chart page")

    st.header("Visualize inference results")

    lot_id = st.text_input("Lot ID", value='', help='Name of the lot of images to run inference on.')
    output_dir = st.text_input("Output directory", value='', help='The directory to store the generated .lrf and .db files.')

    if lot_id != '' and output_dir != '':

        st.subheader(lot_id)

        slider_threshold = st.slider("Select confidence threshold:", 0.0, 1.0, 0.5, key=lot_id+'threshold')
        st.text(f"Probabilities above {slider_threshold} will be considered defects.")

        df = helper.read_database(output_dir, lot_id)

        if st.button("Show defect list", type="primary", key=lot_id+'filter'):
            defect_list = helper.generate_defect_list(df, slider_threshold)
            defect_count = len(defect_list)
            st.markdown(f"Defect count: {defect_count}")
            st.markdown("IDs of defect images:")
            st.markdown(defect_list)

        if st.button("Generate .lrf", type="primary", key=lot_id+'generate_lrf'):

            request = helper.request_lrf(lot_id=lot_id,
                                    output_dir=output_dir,
                                    confidence_threshold=slider_threshold)

            status = request.json()['status']

            # TODO: check file generated instead of just checking status == started
            if status == 'started':
                st.success(f'.lrf file generated at {output_dir}!')
                logger.info(f'.lrf file generated at {output_dir}!')
            else:
                st.error(f".lrf file not generated! Error message: {request.json()['message']}")
                logger.error(f".lrf file not generated! Error message: {request.json()['message']}")


        df_count = prep_dist_chart_data(df)

        st.plotly_chart(draw_defect_dist_chart(df_count, slider_threshold),
                        use_container_width=True, key=lot_id+'chart')

        helper.gap(5)
