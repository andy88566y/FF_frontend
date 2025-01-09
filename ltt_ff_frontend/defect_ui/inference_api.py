import streamlit as st
from loguru import logger

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def dummy_get_progress(status):
    if status == "starting":
        return 10
    elif status == "running":
        return 70
    elif status == "completed":
        return 100
    elif status == "error":
        return 0
    else:
        return 0

def get_color(status):
    if status in ["starting", "running", "completed"]:
        return "green"
    elif status == "error":
        return "red"
    else:
        return "grey"



def app() -> None:
    logger.debug("Opening Inference API page")

    st.title("BDDetector False Filtering API")
    st.caption("API for running inference and fine-tuning False Filtering models.")

    st.header("Inference")


    with st.expander("Run inference on images"):

        st.markdown(":violet[Runs inference on images in the specified directory.]")

        st.header("Parameters")

        image_dir = st.text_input('Image directory', value='/mnt/fs0/x9u_detection_result/N3_M2-6_20240923_000532/N3_M2-6_20240923_000532', help='The directory that contains the Images folder.')
        lrf_path = st.text_input('.lrf path', value='/mnt/fs0/x9u_detection_result/N3_M2-6_20240923_000532/N3_M2-6_20240923_000532_classified.lrf', help='Absolute path to the selected .lrf file.')
        lot_id = st.text_input('Lot ID', value='N3_M2-6_20240923_000532', help='Name of the lot of images to run inference on.')
        output_dir = st.text_input('Output directory', value='/mnt/fs0/shawn', help='The directory to store generated .lrf and .db')
        inference_batch_size = st.select_slider("Inference batch size", [4, 8, 16, 32])

        # selected_lot = st.selectbox(
        #     "Which lot do you want to run inference on?",
        #     helper.get_all_lots()
        # )
        # helper.gap(2)

        slider_threshold = st.slider("Select confidence threshold:", 0.0, 1.0, 0.5)
        st.caption(f"Probabilities above :blue[{slider_threshold}] will be considered defects.")
        helper.gap(2)

        # # add model selection here in the future
        # abs_model_paths = ...
        # model_basename: list[str] = []
        # selected_model = st.selectbox(
        #     "[Placeholder] Which model do you want to run?",
        #     model_basename
        # )
        # helper.gap(2)

        overwrite = st.toggle(label="Overwrite files in output directory", value=True)
        helper.gap(2)

        if st.button("Run inference", type='primary'):
            request = helper.request_inference(image_dir,
                                           lrf_path,
                                           lot_id,
                                           output_dir,
                                           inference_batch_size,
                                           slider_threshold,
                                           overwrite)
            if request.json().get('status') == 'error':
                code = request.json().get('code')
                message = request.json().get('message')
                st.text(f'Error code: {code}\nError message: {message}')
            else:
                inference_id = request.json().get('inference_id')
                st.text(f'Inference ID: {inference_id}')

            # st.divider()
            # # TODO: check with Carl on how to get inference progress
            # st.warning("TODO: Update progress bar")
            # progress_text = 'Inference in progress...'
            # progress_bar = st.progress(0.0, text=progress_text)

    with st.expander("Current jobs"):
        infer_id = st.text_input("Inference ID", help="The unique number generated after click Generate .lrf", value = '')
        if st.button("Check status"):
            status = helper.request_inference_status(infer_id)
            st.json(status.json())

    if st.button('Check all inference jobs'):
        all_statuses = helper.request_all_inference_statuses()
        detail_status_df = pd.DataFrame()

        # rearrange and define which info should be displayed
        brief = ['inference_id', 'status','progress_bar', 'processed_images', 'total_images']
        reorder = ['start_time','inference_id', 'status', 'processed_images', 'total_images', 'progress',
        'estimated_time_remaining', 'lot_id', 'output_dir', 'error_message' ]

        for status in all_statuses:
            new_row = pd.DataFrame([status])
            detail_status_df = pd.concat([detail_status_df, new_row], ignore_index=True)

        # convert start_time float to date time
        for column in detail_status_df.columns:
            if detail_status_df[column].dtype == 'object':
                detail_status_df[column] = detail_status_df[column].astype(str)

        detail_status_df['start_time'] = pd.to_datetime(detail_status_df['start_time'], unit='s')
        detail_status_df['start_time'] = detail_status_df['start_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
        detail_status_df['progress_bar'] = detail_status_df['status'].apply(dummy_get_progress)
        detail_status_df['color'] = detail_status_df['status'].apply(get_color)

        detail_status_df = detail_status_df.iloc[::-1].reset_index(drop=True)
        status_df = detail_status_df[brief]

        st.session_state.status_df = status_df
        st.session_state.detail = detail_status_df[reorder]

    progress_column = st.column_config.ProgressColumn(
        label="progress_bar",
        min_value=0,
        max_value=100
    )

    if "status_df" not in st.session_state:
        st.session_state.status_df = pd.DataFrame(columns=  ['inference_id', 'status','progress_bar', 'processed_images', 'total_images'])
    if "detail" not in st.session_state:
        st.session_state.detail = pd.DataFrame(columns =  ['start_time','inference_id', 'status', 'processed_images', 'total_images', 'progress',
        'estimated_time_remaining', 'lot_id', 'output_dir', 'error_message' ])

    # Pagination settings
    if not st.session_state.status_df.empty:
        page_size = 10
        page_number = st.number_input('Page number', min_value=1, value=1, step=1)
        start_index = (page_number - 1) * page_size
        end_index = page_number * page_size

    # Selection to find more detail
    event = st.dataframe(
        st.session_state.status_df.iloc[start_index:end_index],
        key = "statuses",
        on_select = "rerun",
        selection_mode = "multi-row",
        use_container_width=True,
        column_config={"progress_bar": progress_column}
    ) if not st.session_state.status_df.empty else st.write("")


    if event and event.selection:
        # Check if the "row" value's list is not empty
        if event.selection["rows"]:
            # Extract the selected rows based on the indices
            selected_indices = [start_index + st.session_state.status_df.index[i] for i in event.selection["rows"]]
            selected_rows = st.session_state.detail.loc[selected_indices]
            transposed_detail = selected_rows.T
            st.dataframe(transposed_detail, use_container_width= True )

'''
  page_size = 10
    page_number = st.number_input('Page number', min_value=1, value=1, step=1)  if not st.session_state.status_df.empty else st.write("")
    start_index = (page_number-1) * page_size
    end_index = page_number * page_size
'''