# import numpy as np
# import pandas as pd
# import plotly.express as px
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


# TODO: Change function name
# def draw_defect_dist_chart(df_count: pd.DataFrame, slider_threshold: float):
#     fig = px.histogram(df_count,
#                         x="labels",
#                         y='probabilities',
#                         labels={
#                             "labels": "Probability of defect",
#                             "probabilities": "Count"
#                         },
#                         nbins=100)
#     fig.update_layout(bargap=0.1)
#     fig.add_vline(x=slider_threshold, line_dash = 'dash', line_color = 'firebrick')


def app() -> None:
    logger.debug("Opening Chart page")

    st.header("Visualize inference results")

    # image dir and lrf_path not needed for viz but still need a legitimate path to pass the checks.
    # TODO: Add a skip-check flag for visualization in V2.
    image_dir = st.text_input('Image directory', value='', help='The directory that contains the Images folder.')
    lot_id = st.text_input("Lot ID", value='')
    lrf_path = st.text_input('.lrf path', value='', help='Absolute path to the selected .lrf file.')
    output_dir = st.text_input("Output directory", value='')

    st.subheader(lot_id)

    slider_threshold = st.slider("Select confidence threshold:", 0.0, 1.0, 0.5, key=lot_id+'threshold')
    st.text(f"Probabilities above {slider_threshold} will be considered defects.")

    if st.button("Show defect list", type="primary", key=lot_id+'filter'):
        defect_list = helper.generate_defect_list(output_dir, lot_id, slider_threshold)
        logger.debug(defect_list)
        defect_count = len(defect_list)
        st.markdown(f"Defect count: {defect_count}")
        st.markdown("IDs of defect images:")
        st.markdown(defect_list)

    # generate .lrf file from FalseFilter API using use_cache=True
    if st.button("Generate .lrf", type="primary", key=lot_id+'generate_lrf'):

        request = helper.request_lrf(image_dir=image_dir,
                                 lot_id=lot_id,
                                 lrf_path=lrf_path,
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

    # hot-fix
    if lot_id != '':
        # TODO: Use list comprehension
        labels = []
        for x in range(100):
            val = x * 0.01
            labels.append(f"{val:.2f}")

        # TODO: Migrate this to helper or pull this to a separate function
        # df_count = df[['probabilities']].groupby(
        #     pd.cut(df['probabilities'], np.arange(0.00, 1.01, 0.01)), observed=False).count()
        # df_count['labels'] = labels

        # st.plotly_chart(draw_defect_dist_chart(df_count, slider_threshold),
        #                 use_container_width=True, key=lot_id+'chart')

        helper.gap(5)
