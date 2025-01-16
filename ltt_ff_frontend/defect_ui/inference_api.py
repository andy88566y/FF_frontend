import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def app() -> None:
    logger.debug("Loading Inference Dashboard...")
    st.title("False Filter Inference")
    st.caption("Inference lot data with selected model")

    r1_col1, r1_col2, r1_col3 = st.columns([3, 2, 2])
    with r1_col1:
        inf_base_model = st.selectbox("Base model", options=helper.get_base_models(), index=0,
                                        format_func=lambda x: x.replace("#", " "))
    with r1_col2:
        inf_filter_threshold = st.slider("Confidence threshold:", 0.0, 1.0, 0.5, 0.01, help="Probabilities above thershold will be considered as defects.")
    with r1_col3:
        inf_overwrite = st.toggle(label="Overwrite files in output directory", value=False)

    r2_col1, r2_col2 = st.columns([1, 1])
    with r2_col1:
        inf_lot_id = st.text_input(label='Lot ID', value='')
    with r2_col2:
        inf_output_dir = st.text_input(label='Result directory', value='/mnt/dbpc/xxx', help='The directory to store generated .lrf and .db')

    r3_col1, r3_col2 = st.columns([1, 1])
    with r3_col1:
        inf_image_dir = st.text_input(label='Image directory', value='/mnt/dbpc/xxx', help='The directory that contains the Images folder.')
    with r3_col2:
        inf_lrf_path = st.text_input(label='.lrf path', value='/mnt/dbpc/xxx', help='Absolute path to the selected .lrf file.')

    if st.button("Start Inference Job", type="primary"):
        request = helper.request_inference(base_model=inf_base_model,
                                           lot_id=inf_lot_id,
                                           output_dir=inf_output_dir,
                                           image_dir=inf_image_dir,
                                           lrf_path=inf_lrf_path,
                                           confidence_threshold=inf_filter_threshold,
                                           overwrite=inf_overwrite)

        if request.json().get('status') == 'error':
            code = request.json().get('code')
            message = request.json().get('message')
            st.text(f'Error code: {code}\nError message: {message}')
        else:
            inference_id = request.json().get('inference_id')
            st.text(f'Inference Job ID: {inference_id}')

    st.divider()

    # TODO: Simplify below

    if st.button('Check all Inference Jobs'):
        all_statuses = helper.request_all_inference_statuses()
        columns = ["estimated_time_remaining", "lot_id", "output_dir", "processed_images", "progress", "start_time", "status", "total_images"]
        detail_status_df = pd.DataFrame(columns=columns)

        # rearrange and define which info should be displayed
        brief = ['inference_id', 'status', 'progress_bar', 'processed_images', 'total_images']
        reorder = ["inference_id", "start_time", "estimated_time_remaining", "lot_id", "output_dir", "processed_images", "progress", "status", "total_images"]

        for inference_id, status in all_statuses.items():
            status['inference_id'] = inference_id
            new_row = pd.DataFrame([status])
            detail_status_df = pd.concat([detail_status_df, new_row], ignore_index=True)

        # convert start_time float to date time
        for column in detail_status_df.columns:
            if detail_status_df[column].dtype == 'object':
                detail_status_df[column] = detail_status_df[column].astype(str)

        detail_status_df['start_time'] = pd.to_datetime(detail_status_df['start_time'], unit='s')
        detail_status_df['start_time'] = detail_status_df['start_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
        detail_status_df['progress_bar'] = detail_status_df['status'].apply(lambda x: helper.return_status_style(x)[1])
        detail_status_df['color'] = detail_status_df['status'].apply(lambda x: helper.return_status_style(x)[0])

        detail_status_df = detail_status_df.iloc[::-1].reset_index(drop=True)
        status_df = detail_status_df[brief]

        st.session_state.status_df_inf = status_df
        st.session_state.detail_inf = detail_status_df[reorder]

    progress_column = st.column_config.ProgressColumn(
        label="progress_bar",
        min_value=0,
        max_value=100
    )

    if "status_df_inf" not in st.session_state:
        st.session_state.status_df_inf = pd.DataFrame(columns=  ['inference_id', 'status','progress_bar', 'processed_images', 'total_images'])
    if "detail_inf" not in st.session_state:
        st.session_state.detail_inf = pd.DataFrame(columns =  ["inference_id", "estimated_time_remaining", "lot_id", "output_dir", "processed_images", "progress", "start_time", "status", "total_images"])

    start_index = 0
    end_index = 10
    page_size = 10

    # Pagination settings
    if not st.session_state.status_df_inf.empty:
        page_size = 10
        page_number = st.number_input('Page number', min_value=1, value=1, step=1)
        start_index = (page_number - 1) * page_size
        end_index = page_number * page_size

    # Selection to find more detail
    event_inf = st.dataframe(
        st.session_state.status_df_inf.iloc[start_index:end_index],
        key = "statuses_inference",
        on_select = "rerun",
        selection_mode = "multi-row",
        use_container_width=True,
        column_config={"progress_bar": progress_column}
    ) if not st.session_state.status_df_inf.empty else st.write("")

    if event_inf and event_inf.selection:
        # Check if the "row" value's list is not empty hi
        if event_inf.selection["rows"]:
            # Extract the selected rows based on the indices
            selected_indices = [start_index + st.session_state.status_df_inf.index[i] for i in event_inf.selection["rows"]]
            selected_rows = st.session_state.detail_inf.loc[selected_indices]
            transposed_detail = selected_rows.T
            st.dataframe(transposed_detail, use_container_width= True )
