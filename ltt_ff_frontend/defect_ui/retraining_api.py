import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.constant import (
    LOSS_PARAMS,
    LOSS_TYPE,
    LR_SCHEDULER_PARAMS,
    LR_SCHEDULER_TYPE,
    OPTIMIZER_PARAMS,
    OPTIMIZER_TYPE,
)
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import helper, stop_job_button


def app() -> None:
    logger.debug("Loading Fine-Tuning Dashboard...")
    st.title("False Filter Fine-Tuning")
    st.caption("Train new model with selected lot data")

    training_option = st.segmented_control(
        label="Training option", options=["Fine-tune", "Base-train"], default="Fine-tune"
    )

    if training_option is None:
        st.error("Please select a training method!")
        return

    st.divider()

    r1_col1, r1_col2 = st.columns([3, 2])

    yaml_help_text = """
    **Example of a valid .yaml config file:**\n
    data_paths:\n
    \\- lot_id: N0_M0-0_20240101_000000\n
    &nbsp;&nbsp;lrf_path: /mnt/dbpc/xxx/N0_M0-0_20240101_000000_classified.lrf\n
    &nbsp;&nbsp;image_dir: /mnt/dbpc/xxx/N0_M0-0_20240101_000000/N0_M0-0_20240101_000000\n
    \\- lot_id: N0_M0-0_20240101_000000\n
    &nbsp;&nbsp;lrf_path: /mnt/dbpc/xxx/N0_M0-0_20240101_000000_classified.lrf\n
    &nbsp;&nbsp;image_dir: /mnt/dbpc/xxx/N0_M0-0_20240101_000000/N0_M0-0_20240101_000000
"""
    if training_option == "Fine-tune":
        with r1_col1:
            ft_base_model = st.selectbox(
                "Base model", options=api_helper.get_base_models(), index=0, format_func=helper.format_model_name
            )
        with r1_col2:
            ft_configfile = st.file_uploader(
                "Upload Multi-lot Fine-Tuning Config (.yaml)", type=".yaml", help=yaml_help_text
            )
    else:
        with r1_col1:
            ft_configfile = st.file_uploader(
                "Upload Multi-lot Base-training Config (.yaml)", type=".yaml", help=yaml_help_text
            )

    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns([1, 1, 2, 2])
    with r2_col1:
        ft_site = st.text_input("Site", max_chars=20)
    with r2_col2:
        ft_tool = st.text_input("Tool", value="x9u", max_chars=10)
    with r2_col3:
        ft_techlayer = st.text_input("Tech Layer", max_chars=50)
    with r2_col4:
        ft_layergroup = st.text_input("Layer Group", max_chars=50)

    with st.expander("Fine-Tuning Parameters"):
        if training_option == "Fine-tune":
            ft_epochs = st.number_input("Epochs", value=10)
            ft_lr = st.number_input("Learning Rate", value=0.0001, step=0.0001, format="%0.4f")
        else:
            fc_r1_col1, fc_r1_col2 = st.columns([2, 2])

            with fc_r1_col1:
                ft_epochs = st.number_input("Epochs", value=10)
            with fc_r1_col2:
                ft_lr = st.number_input("Learning Rate", value=0.0001, step=0.0001, format="%0.4f")

            fc_r2_col1, fc_r2_col2, fc_r2_col3 = st.columns([1, 1, 1])

            with fc_r2_col1:
                ft_channel_size_1 = st.number_input("Channel Size 1", value=128)
            with fc_r2_col2:
                ft_channel_size_2 = st.number_input("Channel Size 2", value=256)
            with fc_r2_col3:
                ft_channel_size_3 = st.number_input("Channel Size 3", value=512)

            fc_r3_col1, fc_r3_col2, fc_r3_col3 = st.columns([1, 1, 1])

            with fc_r3_col1:
                ft_kernel_size_1 = st.number_input("Kernel Size 1", value=7)
            with fc_r3_col2:
                ft_kernel_size_2 = st.number_input("Kernel Size 2", value=5)
            with fc_r3_col3:
                ft_kernel_size_3 = st.number_input("Kernel Size 3", value=3)

        optimizer_input_params, loss_input_params, lr_scheduler_input_params = {}, {}, {}
        ft_optimizer_type = st.selectbox(label="Optimizer", options=OPTIMIZER_TYPE, index=0)
        for optimizer_param_name, optimizer_param_data in OPTIMIZER_PARAMS[ft_optimizer_type].items():
            param_default, param_accuracy, param_min, param_max = optimizer_param_data
            st.number_input(
                label=optimizer_param_name,
                value=param_default,
                min_value=param_min,
                max_value=param_max,
                format=param_accuracy,
                key=f"optimizer_{optimizer_param_name}",
            )
            optimizer_input_params[optimizer_param_name] = st.session_state[f"optimizer_{optimizer_param_name}"]

        ft_loss_type = st.selectbox(label="Loss Type", options=LOSS_TYPE, index=0)
        for loss_param_name, loss_param_data in LOSS_PARAMS[ft_loss_type].items():
            param_default, param_accuracy, param_min, param_max = loss_param_data
            st.number_input(
                label=loss_param_name,
                value=param_default,
                min_value=param_min,
                max_value=param_max,
                format=param_accuracy,
                key=f"loss_{loss_param_name}",
            )
            loss_input_params[loss_param_name] = st.session_state[f"loss_{loss_param_name}"]

        ft_lr_scheduler_type = st.selectbox(label="LR Scheduler Type", options=LR_SCHEDULER_TYPE, index=0)
        for lr_scheduler_param_name, lr_scheduler_param_data in LR_SCHEDULER_PARAMS[ft_lr_scheduler_type].items():
            param_default, param_accuracy, param_min, param_max = lr_scheduler_param_data
            st.number_input(
                label=lr_scheduler_param_name,
                value=param_default,
                min_value=param_min,
                max_value=param_max,
                format=param_accuracy,
                key=f"lr_scheduler_{lr_scheduler_param_name}",
            )
            lr_scheduler_input_params[lr_scheduler_param_name] = st.session_state[
                f"lr_scheduler_{lr_scheduler_param_name}"
            ]

    if ft_configfile is not None:
        ft_config = yaml.load(ft_configfile, Loader=yaml.Loader)
        # TODO: Validate yaml file format from backend and pass error message
        st.json(ft_config)

    if st.button(f"Start {training_option} Job", type="primary"):
        # Validate user input first
        required_input = [ft_site, ft_tool, ft_techlayer, ft_layergroup, ft_configfile]
        for item in required_input:
            if not item:
                logger.error(
                    "Missing user input detected. Please enter Site/Tool/Tech Layer/Layer Group, and upload a .yaml config file."
                )
                st.error(
                    "Missing user input detected. Please enter Site/Tool/Tech Layer/Layer Group, and upload a .yaml config file."
                )
                return

        if training_option == "Fine-tune":
            request = api_helper.request_finetune(
                base_model=ft_base_model,
                model_naming=(ft_site, ft_tool, ft_techlayer, ft_layergroup),
                multilot_config=ft_config,
                epochs=ft_epochs,
                lr=ft_lr,
                optimizer_type=ft_optimizer_type,
                optimizer_params=optimizer_input_params,
                loss_type=ft_loss_type,
                loss_params=loss_input_params,
                lr_scheduler_type=ft_lr_scheduler_type,
                lr_scheduler_params=lr_scheduler_input_params,
            )
        else:
            request = api_helper.request_basetrain(
                model_naming=(ft_site, ft_tool, ft_techlayer, ft_layergroup),
                multilot_config=ft_config,
                channel_size=(ft_channel_size_1, ft_channel_size_2, ft_channel_size_3),
                kernel_size=(ft_kernel_size_1, ft_kernel_size_2, ft_kernel_size_3),
                epochs=ft_epochs,
                lr=ft_lr,
                optimizer_type=ft_optimizer_type,
                optimizer_params=optimizer_input_params,
                loss_type=ft_loss_type,
                loss_params=loss_input_params,
                lr_scheduler_type=ft_lr_scheduler_type,
                lr_scheduler_params=lr_scheduler_input_params,
            )

        if request.json().get("status") == "error":
            code = request.json().get("code")
            message = request.json().get("message")
            st.text(f"Error code: {code}\nError message: {message}")
        else:
            training_id = request.json().get("training_id")
            st.text(f"Training Job ID: {training_id}")

    st.divider()

    if "status_df_fin" not in st.session_state:
        st.session_state.status_df_fin = pd.DataFrame()
    if "detailed_df_fin" not in st.session_state:
        st.session_state.detailed_df_fin = pd.DataFrame()

    col1, col2 = st.columns(2, vertical_alignment="bottom")

    with col1:
        if st.button("Check all finetuning jobs"):
            page_size = 10
            current_page = 1
            st.session_state.status_df_fin = api_helper.request_paginated_finetuning_status(page_size, current_page)

    progress_column = st.column_config.ProgressColumn(label="progress_bar", min_value=0, max_value=100)

    # Pagination settings
    with col2:
        page_size = 10
        current_page = st.number_input("Page number", min_value=1, value=1, step=1)
        st.session_state.status_df_fin = api_helper.request_paginated_finetuning_status(page_size, current_page)

    st.header("All fine-tuning jobs") if not st.session_state.status_df_fin.empty else st.write("")

    # Selection to find more detail
    event_fin = (
        st.dataframe(
            st.session_state.status_df_fin,
            key="statuses_finetuning",
            on_select="rerun",
            selection_mode="multi-row",
            use_container_width=True,
            column_config={"progress": progress_column},
        )
        if not st.session_state.status_df_fin.empty
        else st.write("")
    )

    if event_fin and event_fin.selection:
        if event_fin.selection["rows"]:
            # Get list of training_id for all selected fientuning jobs
            selected_finetuning_id = [
                st.session_state.status_df_fin.iloc[i]["training_id"] for i in event_fin.selection["rows"]
            ]

            # Get detailed statuses for each inference job and combine into one df
            raw_df_fin = api_helper.request_finetuning_statuses(selected_finetuning_id)
            st.session_state.detailed_df_fin = raw_df_fin
            st.dataframe(st.session_state.detailed_df_fin, use_container_width=True)
            # Stop job button
            stop_job_button.gen(raw_df_fin)
