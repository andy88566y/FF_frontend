import typing
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile

from ltt_ff_frontend.constant import LOSS_CONFIGS, LR_SCHEDULER_CONFIGS, OPTIMIZER_CONFIGS, TrainingOption
from ltt_ff_frontend.datamodel.ff_core.request import FFCoreTrainingRequest, ModelParams, HyperParams
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.helpers import training_helper
from ltt_ff_frontend.shared_components import helper, stop_job_button


if typing.TYPE_CHECKING:
    from ltt_ff_frontend.constant import NameNumInputParamsConfig


def app() -> None:
    logger.debug("Loading Training Dashboard...")
    st.title("False Filter Training")
    st.caption("Train new model with selected lot data")

    training_option_val = st.segmented_control(
        label="Training option", options=[option.value for option in TrainingOption], default="Fine-tune"
    )

    if training_option_val is None:
        st.error("Please select a training method!")
        return

    st.divider()

    training_option = TrainingOption(training_option_val)
    base_model_name, multilot_config_file = _create_training_option_columns(training_option)
    multilot_config = _load_multilot_config(multilot_config_file)
    site, tool, tech_layer, layer_group = _create_output_model_name_inputs()

    with st.expander(f"{training_option} Parameters"):
        epochs, learning_rate = _create_basic_hyper_params_inputs(training_option)

        channel_size: list[int] = []
        kernel_size: list[int] = []
        if training_option == TrainingOption.BASE_TRAIN:
            channel_size, kernel_size = _create_model_params_inputs()

        optimizer_type, optimizer_params = _create_optimizer_options()
        loss_type, loss_params = _create_loss_options()
        lr_scheduler_type, lr_scheduler_params = _create_lr_scheduler_options()

    # These two are not exposed to the users yet

    batch_size = 32
    model_type = "dual_stream_cnn"  # Currently only one option available

    _start_training_job_button(
        training_option,
        base_model_name,
        model_type,
        channel_size,
        kernel_size,
        site,
        tool,
        tech_layer,
        layer_group,
        multilot_config,
        optimizer_type,
        optimizer_params,
        loss_type,
        loss_params,
        lr_scheduler_type,
        lr_scheduler_params,
        batch_size,
        epochs,
        learning_rate
    )

    st.divider()

    # _init_session_state_df()

    if "status_df_fin" not in st.session_state:
        st.session_state.status_df_fin = pd.DataFrame()

    if "detailed_df_fin" not in st.session_state:
        st.session_state.detailed_df_fin = pd.DataFrame()

    col1, col2 = st.columns(2, vertical_alignment="bottom")
    with col1:
        if st.button("Check all training jobs"):
            page_size = 10
            current_page = 1
            st.session_state.status_df_fin = api_helper.request_paginated_finetuning_status(page_size, current_page)

    progress_column = st.column_config.ProgressColumn(label="progress_bar", min_value=0, max_value=100)

    # Pagination settings
    with col2:
        page_size = 10
        current_page = st.number_input("Page number", min_value=1, value=1, step=1)
        st.session_state.status_df_fin = api_helper.request_paginated_finetuning_status(page_size, current_page)

    if st.session_state.status_df_fin.empty:
        st.write("")
    else:
        st.header("All training jobs")

    # Selection to find more detail
    if st.session_state.status_df_fin.empty:
        event_fin = None
        st.write("")
    else:
        event_fin = st.dataframe(
            st.session_state.status_df_fin,
            key="statuses_finetuning",
            on_select="rerun",
            selection_mode="multi-row",
            use_container_width=True,
            column_config={"progress": progress_column},
        )

    if event_fin and event_fin.selection:  # type: ignore[attr-defined] # "DataframeState" has no attribute "selection"
        if event_fin.selection["rows"]:  # type: ignore[attr-defined] # "DataframeState" has no attribute "selection"
            # Get list of training_id for all selected fientuning jobs
            selected_finetuning_id = [
                st.session_state.status_df_fin.iloc[i]["training_id"] for i in event_fin.selection["rows"]
            ]

            # Get detailed statuses for each inference job and combine into one df
            raw_df_fin = api_helper.request_training_job_records(selected_finetuning_id)
            st.session_state.detailed_df_fin = raw_df_fin
            st.dataframe(st.session_state.detailed_df_fin, use_container_width=True)
            # Stop job button
            stop_job_button.gen(raw_df_fin)


def _create_training_option_columns(training_option: TrainingOption) -> tuple[str | None, UploadedFile | None]:
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
    if training_option == TrainingOption.FINE_TUNE:
        with r1_col1:
            base_model_name = st.selectbox(
                "Base model", options=api_helper.get_base_models(), index=0, format_func=helper.format_model_name
            )
        with r1_col2:
            multilot_config_file = st.file_uploader(
                "Upload Multi-lot Fine-Tuning Config (.yaml)", type=".yaml", help=yaml_help_text
            )
    else:
        base_model_name = None
        with r1_col1:
            multilot_config_file = st.file_uploader(
                "Upload Multi-lot Base-training Config (.yaml)", type=".yaml", help=yaml_help_text
            )

    return base_model_name, multilot_config_file


def _load_multilot_config(multilot_config_file: UploadedFile | None) -> Any:
    multilot_config = None
    if multilot_config_file is not None:
        multilot_config = yaml.load(multilot_config_file, Loader=yaml.Loader)
        # TODO: Validate yaml file format from backend and pass error message
        st.json(multilot_config)

    return multilot_config


def _create_output_model_name_inputs() -> tuple[str, str, str, str]:
    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns([1, 1, 2, 2])
    with r2_col1:
        site = st.text_input("Site", max_chars=20)
    with r2_col2:
        tool = st.text_input("Tool", value="x9u", max_chars=10)
    with r2_col3:
        tech_layer = st.text_input("Tech Layer", max_chars=50)
    with r2_col4:
        layer_group = st.text_input("Layer Group", max_chars=50)

    return site, tool, tech_layer, layer_group


def _create_basic_hyper_params_inputs(training_option: TrainingOption) -> tuple[int, float]:
    if training_option == TrainingOption.FINE_TUNE:
        epochs = st.number_input("Epochs", value=10)
        learning_rate = st.number_input("Learning Rate", value=0.0001, step=0.0001, format="%0.4f")
    else:
        r1_col1, r1_col2 = st.columns([2, 2])
        with r1_col1:
            epochs = st.number_input("Epochs", value=10)

        with r1_col2:
            learning_rate = st.number_input("Learning Rate", value=0.0001, step=0.0001, format="%0.4f")

    return epochs, learning_rate


def _create_model_params_inputs() -> tuple[list[int], list[int]]:

    def _helper(label_template: str, default_values: list[int]) -> list[int]:
        cols = st.columns([1, 1, 1])
        results = []
        for i, (col, val) in enumerate(zip(cols, default_values)):
            with col:
                results.append(st.number_input(label_template.format(i + 1), value=val))

        return results

    channel_size = _helper("Channel Size {}", default_values=[128, 256, 512])
    kernel_size = _helper("Kernel Size {}", default_values=[7, 5, 3])

    return channel_size, kernel_size


# TODO Need a better function name
def _create_type_and_params(
    label: str, configs: dict[str, "NameNumInputParamsConfig"], key_prefix: str
) -> tuple[str, dict[str, Any]]:
    _type = st.selectbox(label=label, options=configs.keys(), index=0)
    select_config = configs[_type]
    params = {}
    for param in select_config.num_input_params:
        key = f"{key_prefix}_{param.name}"
        st.number_input(
            label=param.name,
            value=param.default_value,
            min_value=param.min_value,
            max_value=param.max_value,
            format=param.accuracy,
            step=param.step,
            key=key
        )
        params[param.name] = st.session_state[key]

    return _type, params


def _create_optimizer_options() -> tuple[str, dict[str, Any]]:
    return _create_type_and_params("Optimizer Type", OPTIMIZER_CONFIGS, "optimizer")  # type: ignore[arg-type]


def _create_loss_options() -> tuple[str, dict[str, Any]]:
    return _create_type_and_params("Loss Type", LOSS_CONFIGS, "loss")  # type: ignore[arg-type]


def _create_lr_scheduler_options() -> tuple[str, dict[str, Any]]:
    return _create_type_and_params("LR Scheduler Type", LR_SCHEDULER_CONFIGS, "lr_scheduler")  # type: ignore[arg-type]


def _missing_required_inputs(required_inputs: list[Any]) -> bool:
    for item in required_inputs:
        if not item:
            logger.error(
                "Missing user input detected. "
                "Please enter Site/Tool/Tech Layer/Layer Group, and upload a .yaml config file."
            )
            st.error(
                "Missing user input detected. "
                "Please enter Site/Tool/Tech Layer/Layer Group, and upload a .yaml config file."
            )
            return True

    return False


def _start_training_job_button(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    training_option: TrainingOption,
    base_model_name: str | None,
    model_type: str,
    channel_size: list[int],
    kernel_size: list[int],
    site: str,
    tool: str,
    tech_layer: str,
    layer_group: str,
    multilot_config: Any,
    optimizer_type: str,
    optimizer_params: dict[str, Any],
    loss_type: str,
    loss_params: dict[str, Any],
    lr_scheduler_type: str,
    lr_scheduler_params: dict[str, Any],
    batch_size: int,
    epochs: int,
    learning_rate: float
) -> None:
    if st.button(f"Start {training_option} Job", type="primary"):
        # Validate user input first TODO: Should this be handled by backend?
        required_inputs = [site, tool, tech_layer, layer_group, multilot_config]
        if training_helper == TrainingOption.FINE_TUNE:
            required_inputs.append(base_model_name)

        if _missing_required_inputs(required_inputs):
            return

        output_model_name = training_helper.get_output_model_name(site, tool, tech_layer, layer_group)
        hyper_params = HyperParams(
            optimizer_type=optimizer_type,
            optimizer_params=optimizer_params,
            loss_type=loss_type,
            loss_params=loss_params,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_params=lr_scheduler_params,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            tool=tool,
        )

        if training_option == TrainingOption.FINE_TUNE:
            assert base_model_name is not None
            ff_core_training_request = FFCoreTrainingRequest(
                base_model_name=base_model_name,
                model_params=None,
                output_model_name=output_model_name,
                multilot_config=multilot_config,
                hyper_params=hyper_params
            )
            response = api_helper.request_finetune(ff_core_training_request)
        else:
            model_params = ModelParams(  # type: ignore[call-arg]
                model_type=model_type,
                model_init_params={
                    "channel_size": channel_size,
                    "kernel_size": kernel_size,
                }
            )
            ff_core_training_request = FFCoreTrainingRequest(
                base_model_name=None,
                model_params=model_params,
                output_model_name=output_model_name,
                multilot_config=multilot_config,
                hyper_params=hyper_params
            )
            response = api_helper.request_basetrain(ff_core_training_request)

        if response.json().get("status") == "error":
            code = response.json().get("code")
            message = response.json().get("message")
            st.text(f"Error code: {code}\nError message: {message}")
        else:
            training_id = response.json().get("training_id")
            st.text(f"Training Job ID: {training_id}")
