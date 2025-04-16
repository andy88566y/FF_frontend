import base64
from pprint import pformat
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import API_ROOT, BLANK_MODEL, TIMEOUT


#####################################################################################################
# Formatting                                                                                        #
#####################################################################################################
def gap(size: int) -> None:
    """
    Simple function to space out Streamlit UI elements.

    Args:
        size: The number of newlines.
    """
    for _ in range(size):
        st.write("")


def format_model_name(name: str | None) -> str:
    if name is None:
        return "SCRATCH"
    if name == BLANK_MODEL:
        return BLANK_MODEL
    # For base model name (e.g. base/LTT_SW#x9u#N3#M0-M2#20250124T000000Z#55032dae#55032dae.encrypted.pth)
    if "/" in name:
        model_paths = name.split("/")
        model_type = model_paths[-2]
        model_name = model_paths[-1]
        return f"[{model_type}] {model_name.replace('.encrypted', '').replace('.pth', '').replace('#', ' ')}"
    # For output model name (e.g. 13feb_minye#x9u#tl#lg20250213T151435Z#55032dae#5fb1017f)
    else:
        return f"{name.replace('.encrypted', '').replace('.pth', '').replace('#', ' ')}"


#####################################################################################################
# Get model information                                                                             #
#####################################################################################################
@st.cache_data(ttl="10s")
def get_base_models(include_blank: bool = False) -> list[str]:
    """
    Returns a list of all available models to be used for inference or fine-tuning.
    """
    r = requests.get(f"{API_ROOT}get_model_list", timeout=TIMEOUT)

    if r.json()["status"] == "error":
        logger.error(r.json()["message"])
        return []
    else:
        base_model_list = r.json()["model_list"]
        logger.info(f"List of base models: {base_model_list}")
        return base_model_list if not include_blank else [BLANK_MODEL] + base_model_list


@st.cache_data(ttl="300s")
def get_model_threshold(model_name: str) -> float:
    """
    Return model threshold for selected model
    """
    params = {"model_name": model_name}
    r = requests.get(f"{API_ROOT}get_model_threshold", params=params, timeout=TIMEOUT)

    if r.json()["status"] == "error":
        logger.error(r.json()["message"])
        return 0.0
    else:
        model_threshold = r.json()["model_threshold"]
        logger.info(f"Model threshold for {model_name}: {model_threshold}")
        return model_threshold


#####################################################################################################
# Generate LRF                                                                                      #
#####################################################################################################
def request_recipe_lrf(output_dir: str, recipe: dict[str, Any], lot_id: str) -> requests.Response:
    """
    Args:
        output_dir: Output root directory. The generated lrf will be stored in output_dir/LRF/
        recipe: Recipe for lrf generation
        lot_id: Name of the lot of defect images.

    Returns the reponse of the API request.
    """
    r = requests.post(
        API_ROOT + "generate_lrf",
        json={
            "output_dir": output_dir,
            "recipe": recipe,
            "lot_id": lot_id,
        },
        timeout=TIMEOUT,
    )

    status = r.json()["status"]

    if status == "started":
        logger.info(".lrf generation requested successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return r


def request_threshold_lrf(output_dir: str, confidence_threshold: float, lot_id: str) -> requests.Response:
    """
    Args:
        output_dir: Output root directory. The generated lrf will be stored in output_dir/LRF/
        confidence_threshold: Images with defect probability lower than confidence threshold
                                is considered defective.
        lot_id: Name of the lot of defect images.

    Returns the reponse of the API request.
    """
    meta = get_db_metadata_lists(output_dir)[0]
    model_name = meta.get("model_name", meta.get("model_name_0", ""))
    recipe = {"recipes": [{"model_name": model_name, "threshold": confidence_threshold}]}

    r = requests.post(
        API_ROOT + "generate_lrf",
        json={
            "output_dir": output_dir,
            "recipe": recipe,
            "lot_id": lot_id,
        },
        timeout=TIMEOUT,
    )

    status = r.json()["status"]

    if status == "started":
        logger.info(".lrf generation requested successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return r


def request_top_k_lrf(output_dir: str, top_k: int, lot_id: str) -> requests.Response:
    """
    Call FalseFilter API to generate an .lrf with top K defects

    Args:
        output_dir: Output root directory. The generated lrf will be stored in output_dir/LRF/
        top_k: The top k number of defects will be labeled as defects.

    Returns the reponse of the API request.
    """
    r = requests.post(
        API_ROOT + "generate_top_k_lrf",
        json={
            "output_dir": output_dir,
            "top_k": top_k,
            "lot_id": lot_id,
        },
        timeout=TIMEOUT,
    )

    status = r.json()["status"]

    if status == "started":
        logger.info("Top k .lrf generation requested successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return r


#####################################################################################################
# Inference / Multilot Inference                                                                    #
#####################################################################################################
def request_inference(
    image_dir: str,
    lrf_path: str,
    lot_id: str,
    output_dir: str,
    recipe: Optional[dict[str, Any]] = None,
    base_model: Optional[str] = "",
    confidence_threshold: Optional[float] = 0.0,
    inference_batch_size: int = 32,
    overwrite: bool = False,
) -> requests.Response:
    """
    Calls FalseFilter API to run inference.

    Args:
        recipe: Inference recipe containing models names and thresholds.
        image_dir: Directory containing defect images.
        lrf_path: Absolute path to the .lrf file for the defect images.
        lot_id: Name of the lot of defect images.
        output_dir: Directory to store the generated database file and filtered .lrf file.
        inference_batch_size: Inference batch size. Higher batch size: faster but requires more memory.
        overwrite: If overwrite=False and the result directory contains anything, the inference job will be stopped.
                   If overwrite=True, the entire result directory will be cleared.

    Returns the reponse of the API request.
    """
    if recipe is None:
        recipe = {"recipes": [{"model_name": base_model, "threshold": confidence_threshold}]}

    r = requests.post(
        API_ROOT + "inference",
        json={
            "recipe": recipe,
            "image_dir": image_dir,
            "lrf_path": lrf_path,
            "lot_id": lot_id,
            "output_dir": output_dir,
            "batch_size": inference_batch_size,
            "overwrite": overwrite,
        },
        timeout=TIMEOUT,
    )

    status = r.json()["status"]

    if status == "started":
        logger.info("Inference started running successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return r


def request_multilot_inference(
    output_dir: str,
    multilot_config: dict,
    recipe: Optional[dict[str, Any]] = None,
    base_model: Optional[str] = "",
    confidence_threshold: Optional[float] = 0.0,
    inference_batch_size: int = 32,
    overwrite: bool = False,
    gen_optimized_recipe: bool = False,
) -> requests.Response:
    """
    Calls FalseFilter API to run multilot inference.

    Args:
        output_dir: Directory to store the generated database file and filtered .lrf file.
        multilot_config: Dict containing lot info (lot id, lrf path, image dir)
        recipe: Inference recipe containing models names and thresholds.
        base_model: Name of inference model.
        confidence_threshold: Images with defect probability higher than confidence threshold is considered defective.
        inference_batch_size: Inference batch size. Higher batch size: faster but requires more memory.
        overwrite: If overwrite=False and the result directory contains anything, the inference job will be stopped.
                   If overwrite=True, the entire result directory will be cleared.
        gen_optimized_recipe: If set to True, generate a new recipe with optimized threshold by threshold picker.

    Returns the reponse of the API request.
    """
    if recipe is None:
        recipe = {"recipes": [{"model_name": base_model, "threshold": confidence_threshold}]}

    r = requests.post(
        API_ROOT + "multilot_inference",
        json={
            "output_dir": output_dir,
            "lot_info": multilot_config,
            "recipe": recipe,
            "batch_size": inference_batch_size,
            "overwrite": overwrite,
            "gen_optimized_recipe": gen_optimized_recipe,
        },
        timeout=TIMEOUT,
    )

    status = r.json()["status"]

    if status == "started":
        logger.info("Multilot inference started running successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return r


@st.cache_data(ttl="1s")
def request_paginated_inference_status(page_size: int, current_page: int) -> str:
    """
    Gets pagainated inference status by calling FalseFilter API

    Args:
        page_size : the number of entries to be shown on the dataframe
        current_page : the page that is current requested

    Returns the response of the API request
    """
    r = requests.get(
        f"{API_ROOT}inference/get_paginated_status?page_size={page_size}&current_page={current_page}", timeout=TIMEOUT
    )
    paged_statuses = r.json()

    if paged_statuses["status"] == "error":
        logger.error(f"Error occurred when retrieving inference status from RedisDB: {r.json()['message']}")
        raise ValueError(f"Error occurred when retrieving inference status from RedisDB: {r.json()['message']}")

    logger.info(f"Status of inference request [{current_page}, {page_size}]: {paged_statuses}")

    paged_statuses_df = pd.DataFrame.from_dict(paged_statuses["value"]).T

    if not paged_statuses_df.empty:
        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        paged_statuses_df["start_time"] = pd.to_datetime(paged_statuses_df["start_time"], unit="s").dt.floor("s")
        paged_statuses_df["start_time"] = (
            paged_statuses_df["start_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
        )

        # Sort rows by start time and rename current index column to "inference_id"
        paged_statuses_df = paged_statuses_df.sort_values(by="start_time", ascending=False).reset_index(
            drop=False, names="inference_id"
        )

        # Calculate index based on current page and page size
        start_index = page_size * (current_page - 1) + 1
        paged_statuses_df.index = range(start_index, start_index + len(paged_statuses_df))

        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        if "end_time" in paged_statuses_df.columns:
            paged_statuses_df["end_time"] = pd.to_datetime(paged_statuses_df["end_time"], unit="s").dt.floor("s")
            paged_statuses_df["end_time"] = (
                paged_statuses_df["end_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
            )

            # Calculate runtime only for rows that have end_time
            paged_statuses_df["runtime"] = paged_statuses_df.apply(
                lambda row: row["end_time"] - row["start_time"] if pd.notnull(row["end_time"]) else None, axis=1
            )
            paged_statuses_df["runtime"] = paged_statuses_df["runtime"].apply(
                lambda x: f"{x.components.hours:02}:{x.components.minutes:02}:{x.components.seconds:02}"
                if pd.notnull(x)
                else None
            )

    # Change ordering
    sorted_paged_statuses_df = paged_statuses_df.reindex(
        columns=["inference_id", "status", "progress", "lot_id", "total_images", "start_time", "end_time", "runtime"]
    )

    # For columns not included above, just add them to the back.
    for column in paged_statuses_df.columns:
        if column not in sorted_paged_statuses_df.columns:
            sorted_paged_statuses_df[column] = paged_statuses_df[column]

    return sorted_paged_statuses_df


@st.cache_data(ttl="1s")
def request_paginated_multilot_inference_status(page_size: int, current_page: int) -> str:
    """
    Gets pagainated multilot inference status by calling FalseFilter API

    Args:
        page_size : the number of entries to be shown on the dataframe
        current_page : the page that is current requested

    Returns the response of the API request
    """
    r = requests.get(
        f"{API_ROOT}multilot_inference/get_paginated_status?page_size={page_size}&current_page={current_page}",
        timeout=TIMEOUT,
    )
    paged_statuses = r.json()
    if paged_statuses["status"] == "error":
        logger.error(f"Error occurred when retrieving multilot inference status from RedisDB: {r.json()['message']}")
        raise ValueError(
            f"Error occurred when retrieving multilot inference status from RedisDB: {r.json()['message']}"
        )

    logger.info(f"Status of multilot inference request [{current_page}, {page_size}]: {paged_statuses}")

    paged_statuses_df = pd.DataFrame.from_dict(paged_statuses["value"]).T

    if not paged_statuses_df.empty:
        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        paged_statuses_df["start_time"] = pd.to_datetime(paged_statuses_df["start_time"], unit="s").dt.floor("s")
        paged_statuses_df["start_time"] = (
            paged_statuses_df["start_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
        )

        # Just show lot_id, don't show image_dir and lrf_path
        # This line has to happen before renaming the index column, otherwise we won't be able to access index 0
        paged_statuses_df["lot_info"] = pformat(
            [data_path.get("lot_id", None) for data_path in paged_statuses_df["lot_info"].iloc[0].get("data_paths", [])]
        )

        # Sort rows by start time and rename current index column to "multilot_inference_id"
        paged_statuses_df = paged_statuses_df.sort_values(by="start_time", ascending=False).reset_index(
            drop=False, names="multilot_inference_id"
        )

        # Calculate index based on current page and page size
        start_index = page_size * (current_page - 1) + 1
        paged_statuses_df.index = range(start_index, start_index + len(paged_statuses_df))

        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        if "end_time" in paged_statuses_df.columns:
            paged_statuses_df["end_time"] = pd.to_datetime(paged_statuses_df["end_time"], unit="s").dt.floor("s")
            paged_statuses_df["end_time"] = (
                paged_statuses_df["end_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
            )

            # Calculate runtime only for rows that have end_time
            paged_statuses_df["runtime"] = paged_statuses_df.apply(
                lambda row: row["end_time"] - row["start_time"] if pd.notnull(row["end_time"]) else None, axis=1
            )
            paged_statuses_df["runtime"] = paged_statuses_df["runtime"].apply(
                lambda x: f"{x.components.hours:02}:{x.components.minutes:02}:{x.components.seconds:02}"
                if pd.notnull(x)
                else None
            )

    # Change ordering
    sorted_paged_statuses_df = paged_statuses_df.reindex(
        columns=[
            "multilot_inference_id",
            "status",
            "progress",
            "start_time",
            "end_time",
            "runtime",
            "lot_info",
        ]
    )

    # For columns not included above, just add them to the back.
    for column in paged_statuses_df.columns:
        if column not in sorted_paged_statuses_df.columns:
            sorted_paged_statuses_df[column] = paged_statuses_df[column]

    return sorted_paged_statuses_df


@st.cache_data(ttl="1s")
def request_inference_status(inference_id: str) -> requests.Response:
    """
    Gets inference status by calling FalseFilter API

    Args:
      inference_id: Name of the inference job

    Returns the response of the API request
    """
    r = requests.get(f"{API_ROOT}inference/status/{inference_id}", timeout=TIMEOUT)
    return r.json()


@st.cache_data(ttl="1s")
def request_inference_statuses(inference_id_list: list[str]) -> pd.DataFrame:
    """
    Gets inference status by calling FalseFilter API

    Args:
      inference_id_list: List of inference id to get statuses for.

    Returns the response of the API request
    """
    detailed_inference_statuses = {}
    for inference_id in inference_id_list:
        detailed_inference_statuses[inference_id] = request_inference_status(inference_id)

    return format_inference_status(pd.DataFrame.from_dict(detailed_inference_statuses).T).T


@st.cache_data(ttl="1s")
def request_multilot_inference_status(multilot_inference_id: str) -> requests.Response:
    """
    Gets multilot inference status by calling FalseFilter API

    Args:
      multilot_inference_id: Name of the multilot inference job

    Returns the response of the API request
    """
    r = requests.get(f"{API_ROOT}multilot_inference/status/{multilot_inference_id}", timeout=TIMEOUT)
    return r.json()


@st.cache_data(ttl="1s")
def request_multilot_inference_statuses(multilot_inference_id_list: list[str]) -> pd.DataFrame:
    """
    Gets multilot inference status by calling FalseFilter API

    Args:
      multilot_inference_id_list: List of multilot inference id to get statuses for.

    Returns the response of the API request
    """
    detailed_multilot_inference_statuses = {}
    for multilot_inference_id in multilot_inference_id_list:
        detailed_multilot_inference_statuses[multilot_inference_id] = request_multilot_inference_status(
            multilot_inference_id
        )

    return format_multilot_inference_status(pd.DataFrame.from_dict(detailed_multilot_inference_statuses).T).T


def format_inference_status(inference_status: pd.DataFrame) -> pd.DataFrame:
    """
    Format and sort the detailed inference status dataframe.

    Args:
        inference_status: Dataframe containing raw inference job status details.

    Returns a processed dataframe with adjusted timezones and formatted details.
    """
    if not inference_status.empty:
        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        inference_status["start_time"] = pd.to_datetime(inference_status["start_time"], unit="s").dt.floor("s")
        inference_status["start_time"] = (
            inference_status["start_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
        )

        # Convert model name to user-readable format
        if "model_name" in inference_status.columns:
            inference_status["model_name"] = inference_status["model_name"].apply(format_model_name)

        # Rename index column so that detailed status table will show 'inference_id' instead of 'index'
        inference_status = inference_status.rename(columns={"index": "inference_id"})

        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        if "end_time" in inference_status.columns:
            inference_status["end_time"] = pd.to_datetime(inference_status["end_time"], unit="s").dt.floor("s")
            inference_status["end_time"] = (
                inference_status["end_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
            )

            # Calculate runtime only for rows that have end_time
            inference_status["runtime"] = inference_status.apply(
                lambda row: row["end_time"] - row["start_time"] if pd.notnull(row["end_time"]) else None, axis=1
            )
            inference_status["runtime"] = inference_status["runtime"].apply(
                lambda x: f"{x.components.hours:02}:{x.components.minutes:02}:{x.components.seconds:02}"
                if pd.notnull(x)
                else None
            )

        # If error_message is not in the DF then it shows up as 'nan' on the DF table
        if "error_message" not in inference_status.columns:
            inference_status["error_message"] = "None"

        # TODO: Make hyper-link work
        # inference_status['Review Link'] = inference_status[["output_dir", "image_dir"]].apply(
        #     lambda x: format_url({'result_dir': x['output_dir'], 'image_dir': x['image_dir']}), axis=1
        # )

    # Change ordering
    sorted_inference_statuses_df = inference_status.reindex(
        columns=[
            "status",
            "progress",
            "start_time",
            "end_time",
            "runtime",
            "lot_id",
            # 'Review Link',
            "total_images",
            "defect_count",
            "non_defect_count",
            "unlabeled_count",
            "image_dir",
            "lrf_path",
            "lrf_type",
            "output_dir",
            "message",
            "error_message",
        ]
    )

    # For columns not included above, just add them to the back.
    for column in inference_status.columns:
        if column not in sorted_inference_statuses_df.columns:
            sorted_inference_statuses_df[column] = inference_status[column]

    # st.dataframe will complain when converting non-string type objects
    return sorted_inference_statuses_df.astype(str)


def format_multilot_inference_status(multilot_inference_status: pd.DataFrame) -> pd.DataFrame:
    """
    Format and sort the detailed multilot inference status dataframe.

    Args:
        multilot_inference_status: Dataframe containing raw multilot inference job status details.

    Returns a processed dataframe with adjusted timezones and formatted details.
    """
    if not multilot_inference_status.empty:
        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        multilot_inference_status["start_time"] = pd.to_datetime(
            multilot_inference_status["start_time"], unit="s"
        ).dt.floor("s")
        multilot_inference_status["start_time"] = (
            multilot_inference_status["start_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
        )

        # Convert model name to user-readable format
        if "model_name" in multilot_inference_status.columns:
            multilot_inference_status["model_name"] = multilot_inference_status["model_name"].apply(format_model_name)

        # Rename index column so that detailed status table will show 'inference_id' instead of 'index'
        multilot_inference_status = multilot_inference_status.rename(columns={"index": "inference_id"})

        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        if "end_time" in multilot_inference_status.columns:
            multilot_inference_status["end_time"] = pd.to_datetime(
                multilot_inference_status["end_time"], unit="s"
            ).dt.floor("s")
            multilot_inference_status["end_time"] = (
                multilot_inference_status["end_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
            )

            # Calculate runtime only for rows that have end_time
            multilot_inference_status["runtime"] = multilot_inference_status.apply(
                lambda row: row["end_time"] - row["start_time"] if pd.notnull(row["end_time"]) else None, axis=1
            )
            multilot_inference_status["runtime"] = multilot_inference_status["runtime"].apply(
                lambda x: f"{x.components.hours:02}:{x.components.minutes:02}:{x.components.seconds:02}"
                if pd.notnull(x)
                else None
            )

        # TODO: Make hyper-link work
        # inference_status['Review Link'] = inference_status[["output_dir", "image_dir"]].apply(
        #     lambda x: format_url({'result_dir': x['output_dir'], 'image_dir': x['image_dir']}), axis=1
        # )

    # Change ordering
    sorted_multilot_inference_statuses_df = multilot_inference_status.reindex(
        columns=[
            "status",
            "progress",
            "start_time",
            "end_time",
            "runtime",
            "lot_info",
            "children_job_id",
            # 'Review Link',
            "model_name",
            "threshold",
            "output_dir",
            "message",
            "error_message",
        ]
    )

    # For columns not included above, just add them to the back.
    for column in multilot_inference_status.columns:
        if column not in sorted_multilot_inference_statuses_df.columns:
            sorted_multilot_inference_statuses_df[column] = multilot_inference_status[column]

    return sorted_multilot_inference_statuses_df.astype(str)


#####################################################################################################
# Finetune / Basetrain                                                                              #
#####################################################################################################
def request_finetune(
    base_model: str,
    model_naming: tuple[str, str, str, str],
    multilot_config: dict,
    epochs: int,
    lr: float,
    optimizer_type: str,
    optimizer_params: dict[str, Any],
    loss_type: str,
    loss_params: dict[str, Any],
    lr_scheduler_type: str,
    lr_scheduler_params: dict[str, Any],
) -> requests.Response:
    """
    Calls FFA model fine-tuning.

    Args:
        base_model: Name of model to finetune.
        model_naming: Details to be used for re-trained model (site, tool, tech layer, layer group)
        multilot_config: Dict containing training data info (lot id, lrf path, image dir)
        epochs: Number of training epochs.
        lr: Learning rate.
        optimizer_type: Adam, AdamW, etc
        optimizer_params: Parameters required for the selected optimizer type, if any.
        loss_type: bce, focal, etc.
        loss_params: Parameter required for the selected loss type, if any.
        lr_scheduler_type: disable, plateau, etc
        lr_scheduler_params: Parameters required for the selected lr scheduler, if any.

    Returns the reponse of the API request.
    """
    # TODO: Check multilot_config is valid structure

    r = requests.post(
        API_ROOT + "finetune",
        json={
            "base_model_name": base_model,
            "batch_size": 32,
            "epochs": epochs,
            "learning_rate": lr,
            "model_naming": model_naming,
            "training_info": multilot_config,
            "optimizer_type": optimizer_type,
            "optimizer_params": optimizer_params,
            "loss_type": loss_type,
            "loss_params": loss_params,
            "lr_scheduler_type": lr_scheduler_type,
            "lr_scheduler_params": lr_scheduler_params,
        },
        timeout=TIMEOUT,
    )

    status = r.json()["status"]

    if status == "started":
        logger.info("Model fine-tuning started running successfully!")
    else:
        logger.error(f"Error occurred when calling fine-tuning API: {r.json()['message']}")

    return r


def request_basetrain(
    model_naming: tuple[str, str, str, str],
    multilot_config: dict,
    channel_size: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    epochs: int,
    lr: float,
    optimizer_type: str,
    optimizer_params: dict[str, Any],
    loss_type: str,
    loss_params: dict[str, Any],
    lr_scheduler_type: str,
    lr_scheduler_params: dict[str, Any],
) -> requests.Response:
    """
    Calls FFA model base-training.

    Args:
        model_naming: Details to be used for re-trained model (site, tool, tech layer, layer group)
        multilot_config: Dict containing training data info (lot id, lrf path, image dir)
        channel_size, kernel_size: tuple of model structure config
        epochs: Number of training epochs.
        lr: Learning rate.
        optimizer_type: Adam, AdamW, etc
        optimizer_params: Parameters required for the selected optimizer type, if any.
        loss_type: bce, focal, etc.
        loss_params: Parameter required for the selected loss type, if any.
        lr_scheduler_type: disable, plateau, etc
        lr_scheduler_params: Parameters required for the selected lr scheduler, if any.

    Returns the reponse of the API request.
    """
    # TODO: Check multilot_config is valid structure

    r = requests.post(
        API_ROOT + "basetrain",
        json={
            "batch_size": 32,
            "epochs": epochs,
            "learning_rate": lr,
            "model_naming": model_naming,
            "training_info": multilot_config,
            "model_params": {
                "channel_size": list(channel_size),
                "kernel_size": list(kernel_size),
            },
            "optimizer_type": optimizer_type,
            "optimizer_params": optimizer_params,
            "loss_type": loss_type,
            "loss_params": loss_params,
            "lr_scheduler_type": lr_scheduler_type,
            "lr_scheduler_params": lr_scheduler_params,
        },
        timeout=TIMEOUT,
    )

    status = r.json()["status"]

    if status == "started":
        logger.info("Model base-training started running successfully!")
    else:
        logger.error(f"Error occurred when calling base-training API: {r.json()['message']}")

    return r


@st.cache_data(ttl="1s")
def request_paginated_finetuning_status(page_size: int, current_page: int) -> pd.DataFrame:
    """
    Gets pagainated inference status by calling FalseFilter API

    Args:
        page_size : the number of entries to be shown on the dataframe
        current_page : the page that is current requested

    Returns the response of the API request
    """
    r = requests.get(
        f"{API_ROOT}finetune/get_paginated_status?page_size={page_size}&current_page={current_page}", timeout=TIMEOUT
    )
    paged_statuses = r.json()
    if paged_statuses["status"] == "error":
        logger.error(f"Error occurred when retrieving finetuning status from RedisDB: {r.json()['message']}")
        raise ValueError(f"Error occurred when retrieving finetuning status from RedisDB: {r.json()['message']}")
    logger.info(f"Status of finetuning request [{current_page}, {page_size}]: {paged_statuses}")

    paged_statuses_df = pd.DataFrame.from_dict(paged_statuses["value"]).T

    if not paged_statuses_df.empty:
        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        paged_statuses_df["start_time"] = pd.to_datetime(paged_statuses_df["start_time"], unit="s").dt.floor("s")
        paged_statuses_df["start_time"] = (
            paged_statuses_df["start_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
        )

        # Convert model name to user-readable format
        paged_statuses_df["base_model_name"] = paged_statuses_df["base_model_name"].apply(format_model_name)

        # Sort rows by start time and rename current index column to "training_id"
        paged_statuses_df = paged_statuses_df.sort_values(by="start_time", ascending=False).reset_index(
            drop=False, names="training_id"
        )

        # Calculate index based on current page and page size
        start_index = page_size * (current_page - 1) + 1
        paged_statuses_df.index = range(start_index, start_index + len(paged_statuses_df))

        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        if "end_time" in paged_statuses_df.columns:
            paged_statuses_df["end_time"] = pd.to_datetime(paged_statuses_df["end_time"], unit="s").dt.floor("s")
            paged_statuses_df["end_time"] = (
                paged_statuses_df["end_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
            )

            # Calculate runtime only for rows that have end_time
            paged_statuses_df["runtime"] = paged_statuses_df.apply(
                lambda row: row["end_time"] - row["start_time"] if pd.notnull(row["end_time"]) else None, axis=1
            )
            paged_statuses_df["runtime"] = paged_statuses_df["runtime"].apply(
                lambda x: f"{x.components.hours:02}:{x.components.minutes:02}:{x.components.seconds:02}"
                if pd.notnull(x)
                else None
            )

    # Change ordering
    sorted_paged_statuses_df = paged_statuses_df.reindex(
        columns=[
            "training_id",
            "status",
            "progress",
            "start_time",
            "end_time",
            "runtime",
            "base_model_name",
            "site",
            "tool",
            "tech_layer",
            "layer_group",
        ]
    )

    # For columns not included above, just add them to the back.
    for column in paged_statuses_df.columns:
        if column not in sorted_paged_statuses_df.columns:
            sorted_paged_statuses_df[column] = paged_statuses_df[column]

    return sorted_paged_statuses_df


@st.cache_data(ttl="1s")
def request_finetuning_status(finetuning_id: str) -> requests.Response:
    """
    Gets finetuning status by calling FalseFilter API

    Args:
      finetuning_id: Name of the finetuning job

    Returns the response of the API request
    """
    r = requests.get(f"{API_ROOT}finetune/status/{finetuning_id}", timeout=TIMEOUT)
    return r.json()


@st.cache_data(ttl="1s")
def request_finetuning_statuses(finetuning_id_list: list[str]) -> pd.DataFrame:
    """
    Gets finetuning status by calling FalseFilter API

    Args:
      finetuning_id_list: List of training id to get statuses for.

    Returns the response of the API request
    """
    detailed_finetuning_statuses = {}
    for training_id in finetuning_id_list:
        detailed_finetuning_statuses[training_id] = request_finetuning_status(training_id)

    return format_finetuning_status(pd.DataFrame.from_dict(detailed_finetuning_statuses).T).T


def format_finetuning_status(finetuning_status: pd.DataFrame) -> pd.DataFrame:
    """
    Format and sort the detailed finetuning status dataframe.

    Args:
        finetuning_status: Dataframe containing raw finetuning job status details.

    Returns a processed dataframe with adjusted timezones and formatted details.
    """
    if not finetuning_status.empty:
        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        finetuning_status["start_time"] = pd.to_datetime(finetuning_status["start_time"], unit="s").dt.floor("s")
        finetuning_status["start_time"] = (
            finetuning_status["start_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
        )

        # Convert model name to user-readable format
        finetuning_status["base_model_name"] = finetuning_status["base_model_name"].apply(format_model_name)
        finetuning_status["output_model_name"] = finetuning_status["output_model_name"].apply(format_model_name)

        # Format training info (from yaml config) to be easily readable
        finetuning_status["training_info"] = finetuning_status["training_info"].map(lambda x: pformat(x))

        # Format epoch loss and validation loss to be more readable
        if "debug" in finetuning_status.columns:
            finetuning_status["debug"] = finetuning_status["debug"].map(lambda x: pformat(x))

        # Rename index column so that detailed status table will show 'inference_id' instead of 'index'
        finetuning_status = finetuning_status.rename(columns={"index": "inference_id"})

        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        if "end_time" in finetuning_status.columns:
            finetuning_status["end_time"] = pd.to_datetime(finetuning_status["end_time"], unit="s").dt.floor("s")
            finetuning_status["end_time"] = (
                finetuning_status["end_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
            )

            # Calculate runtime only for rows that have end_time
            finetuning_status["runtime"] = finetuning_status.apply(
                lambda row: row["end_time"] - row["start_time"] if pd.notnull(row["end_time"]) else None, axis=1
            )
            finetuning_status["runtime"] = finetuning_status["runtime"].apply(
                lambda x: f"{x.components.hours:02}:{x.components.minutes:02}:{x.components.seconds:02}"
                if pd.notnull(x)
                else None
            )

    # Change ordering
    sorted_finetuning_statuses_df = finetuning_status.reindex(
        columns=[
            "status",
            "progress",
            "start_time",
            "end_time",
            "runtime",
            "base_model_name",
            "model_params",
            "site",
            "tool",
            "tech_layer",
            "layer_group",
            "output_model_name",
            "current_epoch",
            "total_epochs",
            "batch_size",
            "learning_rate",
            "optimizer_type",
            "optimizer_params",
            "loss_type",
            "loss_params",
            "lr_scheduler_type",
            "lr_scheduler_params",
            "training_info",
            "debug",
            "message",
            "error_message",
        ]
    )

    # For columns not included above, just add them to the back.
    for column in finetuning_status.columns:
        if column not in sorted_finetuning_statuses_df.columns:
            sorted_finetuning_statuses_df[column] = finetuning_status[column]

    return sorted_finetuning_statuses_df.astype(str)


#####################################################################################################
# Database                                                                                          #
#####################################################################################################
@st.cache_data(ttl="10s")
def get_db_metadata_lists(output_dir: str) -> list[dict[str, Any]]:
    """
    Get Result DB metadata.

    Args:
        output_dir: Root output directory where inference results were stored.

        Returns:
            A dictionary of result database metadata
    """
    r = requests.get(API_ROOT + "result/get_db_metadata_lists", params={"output_dir": output_dir}, timeout=TIMEOUT)

    if r.json()["status"] == "completed":
        return r.json()["db_metadata"]
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")


@st.cache_data(ttl="10s")
def get_defect_id_lists(output_dir: str) -> list[list[str]]:
    """
    Get list of defect IDs from a database.

    Args:
        output_dir: Root output directory where inference results were stored.

        Returns:
            A list of the defect IDs of a lot of images.
    """
    r = requests.get(API_ROOT + "result/get_defect_id_lists", params={"output_dir": output_dir}, timeout=TIMEOUT)

    if r.json()["status"] == "completed":
        return r.json()["defect_id_list"]
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")


@st.cache_data(ttl="1s")
def get_topk_model_threshold(output_dir: str, top_k: int = 150, lot_id: str = "") -> float:
    """
    Return model threshold for selected model
    """
    params = {"output_dir": output_dir, "top_k": top_k, "lot_id": lot_id}
    r = requests.get(f"{API_ROOT}result/get_topk_threshold", params=params, timeout=TIMEOUT)

    if r.json()["status"] == "error":
        logger.error(r.json()["message"])
        return 0.0
    else:
        model_threshold = r.json()["threshold"]
        logger.info(f"Model threshold for `{output_dir}` top_k={top_k}: {model_threshold}")
        return model_threshold


# TODO: Split this into smaller functions
@st.cache_data(ttl="30s")
def get_lrf_data_lists(output_dir: str, cols: list[str], include_prob: bool = False) -> list[list[dict[str, Any]]]:
    """
    Return lrf data with selected columns
    """
    params = {"output_dir": output_dir, "cols": ",".join(cols)}
    r = requests.get(f"{API_ROOT}result/get_lrf_data_lists", params=params, timeout=TIMEOUT)
    if r.json()["status"] == "error":
        logger.error(f"Error occurred when calling get LRF API (lrf): {r.json()['message']}")
        raise ValueError(f"Error occurred when calling get LRF API (lrf): {r.json()['message']}")
    else:
        lrf_data_list = r.json()["lrf_data"]
        for lrf_data in lrf_data_list:
            logger.info(f"LRF data of {len(lrf_data)} defects loaded from `{output_dir}`")

    r = requests.get(API_ROOT + "result/get_answer", json={"output_dir": output_dir}, timeout=TIMEOUT)
    if r.json()["status"] == "error":
        logger.error(f"Error occurred when calling LRF API (ans): {r.json()['message']}")
        raise ValueError(f"Error occurred when calling LRF API (ans): {r.json()['message']}")
    else:
        answer_lists = r.json()["answer_list"]
        lrf_data_with_ans_list = []
        for lrf_data, answer_list in zip(lrf_data_list, answer_lists):
            if len(lrf_data) != len(answer_list):
                logger.error(f"Difference in length between lrf data and answer {len(lrf_data)} {len(answer_list)}")
                raise ValueError(f"Difference in length between lrf data and answer {len(lrf_data)} {len(answer_list)}")
            lrf_data_with_ans = []
            for data, ans in zip(lrf_data, answer_list):
                lrf_data_with_ans.append({**data, "Ans": ans})
            lrf_data_with_ans_list.append(lrf_data_with_ans)

    if include_prob:
        r = requests.get(API_ROOT + "result/get_probability", json={"output_dir": output_dir}, timeout=TIMEOUT)

        if r.json()["status"] == "error":
            logger.error(f"Error occurred when calling get LRF API (prob): {r.json()['message']}")
            raise ValueError(f"Error occurred when calling get LRF API (prob): {r.json()['message']}")
        else:
            probability_lists = r.json()["probability_list"]
            lrf_data_with_prob_list = []
            for lrf_data_with_ans, probability_list in zip(lrf_data_with_ans_list, probability_lists):
                if len(lrf_data_with_ans) != len(probability_list):
                    logger.error(
                        f"Difference in length between lrf data and probability {len(lrf_data_with_ans)} {len(probability_list)}"
                    )
                    raise ValueError(
                        f"Difference in length between lrf data and probability {len(lrf_data_with_ans)} {len(probability_list)}"
                    )
                lrf_data_with_prob = []
                for data, prob in zip(lrf_data_with_ans, probability_list):
                    lrf_data_with_prob.append({**data, "Probability": prob})
                lrf_data_with_prob_list.append(lrf_data_with_prob)
            return lrf_data_with_prob_list
    else:
        return lrf_data_with_ans_list


@st.cache_data(ttl="10s")
def get_prc_data(output_dir: str, return_curve: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the data needed to draw a PRC curve.

    Args:
        output_dir: Root output directory of inference resuits.
        lot_id: Name of the lof of defect images.
        model_name: Name of inference results.
        return_curve: If false, just return the area under the curve (AUPRC)
    """
    r = requests.get(
        API_ROOT + "result/get_prc_data",
        params={"output_dir": output_dir, "return_curve": return_curve},
        timeout=TIMEOUT,
    )

    prc_data_list = r.json()["prc_data"]
    prc_data_ndarray = tuple(np.array(data_list) for data_list in prc_data_list)

    return prc_data_ndarray


@st.cache_data(ttl="10s")
def get_roc_data(output_dir: str, return_curve: bool = True) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get the data needed to draw an ROC curve (fpr, tpr, threshold)

    Args:
        output_dir: Root output directory of inference resuits.
        lot_id: Name of the lof of defect images.
        model_name: Name of inference results.
        return_curve: If false, just return the area under the curve (AUROC)
    """
    r = requests.get(
        API_ROOT + "result/get_roc_data",
        params={"output_dir": output_dir, "return_curve": return_curve},
        timeout=TIMEOUT,
    )

    roc_data_lists = r.json()["roc_data"]
    roc_data_ndarray_list = [
        tuple(np.array(data_list) for data_list in roc_data_list) for roc_data_list in roc_data_lists
    ]

    return roc_data_ndarray_list


@st.cache_data(ttl="10s")
def get_roc_threshold_marker_coordinates(
    output_dir: str, selected_threshold: float | None = None
) -> list[tuple[float, float]]:
    """
    Get the coordinates to draw the threshold marker on the CR/FFR curve.

    Args:
        output_dir: Root output directory of inference resuits.
    """
    r = requests.get(
        API_ROOT + "result/get_roc_selected_threshold",
        params={"output_dir": output_dir, "selected_threshold": selected_threshold},
        timeout=TIMEOUT,
    )
    threshold_coordinates_list = r.json()["threshold_coordinates_list"]

    return threshold_coordinates_list


@st.cache_data(ttl="10s")
def get_probability(output_dir: str, defect_id: list[list[int]]) -> list[list[float]]:
    """
    Read a list of the defect probabilities from a database.

    Args:
        output_dir: Root output directory where inference results were stored.
        defect_id: ID of the defect images

    Returns:
        A list of the defect probabilities of a lot of images.
    """
    r = requests.get(
        API_ROOT + "result/get_probability",
        json={"output_dir": output_dir, "defect_id_list": defect_id},
        timeout=TIMEOUT,
    )

    if r.json()["status"] == "completed":
        return r.json()["probability_list"]
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")


@st.cache_data(ttl="10s")
def get_answer(output_dir: str, defect_id: list[list[str]]) -> list[list[int]]:
    """
    Read a list of the ground truths from a database.

    Args:
        output_dir: Root output directory where inference results were stored.
        defect_id: ID of the defect images

    Returns:
        A list of the ground truths of a lot of images.
    """
    r = requests.get(
        API_ROOT + "result/get_answer", json={"output_dir": output_dir, "defect_id_list": defect_id}, timeout=TIMEOUT
    )

    if r.json()["status"] == "completed":
        return r.json()["answer_list"]
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")


@st.cache_data(ttl="10s")
def get_predictions(
    output_dir: str,
    recipe: dict[str, Any],
    defect_list: Optional[list[list[str]]] = None,
) -> list[list[int]]:
    """
    Read a list of the ground truths from a database.

    Args:
        output_dir: Root output directory where inference results were stored.
        defect_id: ID of the defect images

    Returns:
        A list of the ground truths of a lot of images.
    """
    r = requests.post(
        API_ROOT + "result/get_predictions",
        json={
            "output_dir": output_dir,
            "defect_id_list": defect_list,
            "recipe": recipe,
        },
        timeout=TIMEOUT,
    )

    if r.json()["status"] == "completed":
        return r.json()["predictions_list"]
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")


def format_url(params: dict[str, str]) -> str:
    # TODO: Get correct base url
    base_url = "http://xxx:6501"
    param_strs = []
    for k, v in params.items():
        param_strs.append(f"{k}={base64.urlsafe_b64encode(str.encode(v)).decode()}")
    return f"{base_url}/?{'&'.join(param_strs)}"


#####################################################################################################
# Common API                                                                                        #
#####################################################################################################
@st.cache_data(ttl="1s")
def request_stop_job(job_id: str) -> str:
    r = requests.post(f"{API_ROOT}stop_job?job_id={job_id}", timeout=TIMEOUT)

    return f"{r.status_code}: {r.json().get('message', 'message not found...')}"


def add_stop_job_button(df: pd.DataFrame) -> None:
    render_cols = st.columns(len(df.columns) + 1, vertical_alignment="top")
    for job_id, render_col in zip(df.columns, render_cols[1:]):
        with render_col:
            cannot_stop = df.loc["status", job_id] in ["completed", "error", "stopped", "stopping"]
            if st.button(f"stop {job_id}", key=f"stop-{job_id}", use_container_width=True, disabled=cannot_stop):
                message = request_stop_job(job_id)
                st.write(message)
