import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def app() -> None:
    logger.debug("Loading Fine-Tuning Dashboard...")
    st.title("False Filter Fine-Tuning")
    st.caption("Train new model with selected lot data")

    r1_col1, r1_col2 = st.columns([3, 2])

    with r1_col1:
        ft_base_model = st.selectbox("Base model", options=helper.get_base_models(), index=0,
                                     format_func=helper.format_model_name)

    yaml_help_text = '''
    **Example of a valid .yaml config file:**\n
    data_paths:\n
    \- lot_id: N0_M0-0_20240101_000000\n
    &nbsp;&nbsp;lrf_path: /mnt/dbpc/xxx/N0_M0-0_20240101_000000_classified.lrf\n
    &nbsp;&nbsp;image_dir: /mnt/dbpc/xxx/N0_M0-0_20240101_000000/N0_M0-0_20240101_000000\n
    \- lot_id: N0_M0-0_20240101_000000\n
    &nbsp;&nbsp;lrf_path: /mnt/dbpc/xxx/N0_M0-0_20240101_000000_classified.lrf\n
    &nbsp;&nbsp;image_dir: /mnt/dbpc/xxx/N0_M0-0_20240101_000000/N0_M0-0_20240101_000000
'''

    with r1_col2:
        ft_configfile = st.file_uploader("Upload Multi-lot Fine-Tuning Config (.yaml)", type=".yaml", help=yaml_help_text)

    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns([1, 1, 2, 2])
    with r2_col1:
        ft_site = st.text_input("Site", max_chars=20)
    with r2_col2:
        ft_tool = st.text_input("Tool", value ="x9u", max_chars=10)
    with r2_col3:
        ft_techlayer = st.text_input("Tech Layer", max_chars=50)
    with r2_col4:
        ft_layergroup = st.text_input("Layer Group", max_chars=50)

    with st.expander("Fine-Tuning Parameters"):
        ft_epochs = st.number_input('Epochs', value=10)
        ft_lr = st.number_input('Learning Rate', value=0.0001, step=0.0001, format="%0.4f")

    if ft_configfile is not None:
        ft_config = yaml.load(ft_configfile, Loader=yaml.Loader)
        # TODO: Validate yaml file format from backend and pass error message
        st.json(ft_config)

    if st.button("Start Fine-Tuning Job", type="primary"):

        # Validate user input first
        required_input = [ft_site, ft_tool, ft_techlayer, ft_layergroup, ft_configfile]
        for item in required_input:
            if not item:
                logger.error('Missing user input detected. Please enter Site/Tool/Tech Layer/Layer Group, and upload a .yaml config file.')
                st.error('Missing user input detected. Please enter Site/Tool/Tech Layer/Layer Group, and upload a .yaml config file.')
                return

        request = helper.request_finetune(base_model=ft_base_model,
                                          model_naming=(ft_site, ft_tool, ft_techlayer, ft_layergroup),
                                          multilot_config=ft_config,
                                          epochs=ft_epochs, lr=ft_lr)

        if request.json().get('status') == 'error':
            code = request.json().get('code')
            message = request.json().get('message')
            st.text(f'Error code: {code}\nError message: {message}')
        else:
            training_id = request.json().get('training_id')
            st.text(f'Training Job ID: {training_id}')

    st.divider()

    if 'status_df_fin' not in st.session_state:
        st.session_state.status_df_fin = pd.DataFrame()
    if 'detailed_df_fin' not in st.session_state:
        st.session_state.detailed_df_fin = pd.DataFrame()

    col1, col2 = st.columns(2, vertical_alignment='bottom')

    with col1:
        if st.button('Check all finetuning jobs'):
            page_size = 10
            current_page = 1
            st.session_state.status_df_fin = helper.request_paginated_finetuning_status(page_size, current_page)

    progress_column = st.column_config.ProgressColumn(
        label='progress_bar',
        min_value=0,
        max_value=100
    )

    # Pagination settings
    with col2:
        page_size = 10
        current_page = st.number_input('Page number', min_value=1, value=1, step=1)
        st.session_state.status_df_fin = helper.request_paginated_finetuning_status(page_size, current_page)

    st.header('All fine-tuning jobs') if not st.session_state.status_df_fin.empty else st.write('')

    # Selection to find more detail
    event_fin = st.dataframe(
        st.session_state.status_df_fin,
        key = 'statuses_finetuning',
        on_select = 'rerun',
        selection_mode = 'multi-row',
        use_container_width=True,
        column_config={'progress': progress_column}
    ) if not st.session_state.status_df_fin.empty else st.write('')

    if event_fin and event_fin.selection:
        if event_fin.selection['rows']:
            # Get list of training_id for all selected fientuning jobs
            selected_finetuning_id = [st.session_state.status_df_fin.iloc[i]['training_id'] for i in event_fin.selection['rows']]

            # Get detailed statuses for each finetuning job and combine into one df
            st.session_state.detailed_df_fin = helper.request_finetuning_statuses(selected_finetuning_id)
            st.dataframe(st.session_state.detailed_df_fin, use_container_width=True)
