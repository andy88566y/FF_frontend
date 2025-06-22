import typing
from pprint import pformat
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import API_ROOT, BLANK_MODEL, TIMEOUT, APIGroup
from ltt_ff_frontend.shared_components.helper import format_model_name


if typing.TYPE_CHECKING:
    from ltt_ff_frontend.datamodel.ff_core.request import FFCoreTrainingRequest


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
        return base_model_list if not include_blank else [BLANK_MODEL] + base_model_list


@st.cache_data(ttl="10s")
def get_ds_model_dirs() -> list[str]:
    """
    Returns a list of all directories found in DS_models. This will be used to filter the DS models list.
    """
    r = requests.get(f"{API_ROOT}get_ds_model_dirs", timeout=TIMEOUT)

    if r.json()["status"] == "error":
        logger.error(r.json()["message"])
        return []
    else:
        ds_model_dirs = r.json()["ds_model_dirs"]
        return [""] + ds_model_dirs


@st.cache_data(ttl="10s")
def get_ds_models(filter_week: str = "") -> list[str]:
    """
    Returns a list of DS models, filtered by week of release.
    """
    params = {"filter_week": filter_week}
    r = requests.get(f"{API_ROOT}get_ds_model_list", params=params, timeout=TIMEOUT)

    if r.json()["status"] == "error":
        logger.error(r.json()["message"])
        return []
    else:
        ds_model_list = r.json()["ds_model_list"]
        return ds_model_list


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


@st.cache_data(ttl="300s")
def get_model_details(model_name: str) -> dict[str, Any]:
    """
    Return model details for selected model (threshold, metrics, params)
    """
    params = {"model_name": model_name}
    r = requests.get(f"{API_ROOT}get_model_details", params=params, timeout=TIMEOUT)

    if r.json()["status"] == "error":
        logger.error(r.json()["message"])
        return {}
    else:
        model_details = r.json()["model_details"]
        logger.info(f"Model details for {model_name}: {model_details}")
        return model_details


def get_recipe_filtered_results_from_api(output_dir: str, recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """
    get calculated results from ff core

    return example:
    [
        {
            "as_is_defect_count": 100,
            "to_be_defect_count": 10,
            "filter_rate": 0.9,
            "as_is_true_defect_count": 10,
            "to_be_true_defect_count": 10,
            "capture_rate": 1.0,
            "as_is_non_defect_count": 90,
            "to_be_non_defect_count": 90,
            "false_filter_rate": 1.0,
            "unlabeled": 2,
            "filtered_unlabeled_defect_count": 1,
        },
        {...}
    ]
    """
    params = {"inference_result_dir": output_dir, "recipe": recipe}
    r = requests.get(API_ROOT + "result/get_filtered_stats", json=params, timeout=TIMEOUT)

    if r.status_code == requests.codes.ok:
        return r.json()["filtered_stats"]
    else:
        r.raise_for_status()


def get_missed_defects(
    recipe: dict[str, Any],
    inference_result_dir: str,
) -> list[dict[str, Any]]:
    """
    Get a list of True Defects that are undetected by the recipe.
    """
    params = {"inference_result_dir": inference_result_dir, "recipe": recipe}
    r = requests.get(API_ROOT + "result/get_missed_defects", json=params, timeout=TIMEOUT)

    if r.json()["status"] == "completed":
        return r.json()["missed_defects"]
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")


def get_particle_mode_only_defects(
    inference_result_dir: str,
) -> list[dict[str, Any]]:
    """
    Get a list of defects that are detected by particle mode only.
    """
    params = {"inference_result_dir": inference_result_dir}
    r = requests.get(API_ROOT + "result/get_particle_mode_only_defects", json=params, timeout=TIMEOUT)

    if r.json()["status"] == "completed":
        return r.json()["particle_mode_only_defects"]
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")


@st.cache_data(ttl="10s")
def get_db_metadata_lists(output_dir: str, lot_id: str = "") -> list[dict[str, Any]]:
    """
    Get Result DB metadata.

    Args:
        output_dir: Root output directory where inference results were stored.
        lot_id: A specific lot ID to be filtered.

        Returns:
            A dictionary of result database metadata
    """
    r = requests.get(
        API_ROOT + "result/get_db_metadata_lists", params={"output_dir": output_dir, "lot_id": lot_id}, timeout=TIMEOUT
    )

    if r.json()["status"] == "completed":
        return r.json()["db_metadata"]
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")


@st.cache_data(ttl="10s")
def get_answer(output_dir: str, defect_id_lists: list[list[str]]) -> list[list[int]]:
    """
    Read a list of the ground truths from a database.

    Args:
        output_dir: Root output directory where inference results were stored.
        defect_id: ID of the defect images

    Returns:
        A list of the ground truths of a lot of images.
    """
    r = requests.get(
        API_ROOT + "result/get_answer",
        json={"output_dir": output_dir, "defect_id_list": defect_id_lists},
        timeout=TIMEOUT,
    )

    if r.json()["status"] == "completed":
        return r.json()["answer_list"]
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")


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


@st.cache_data(ttl="10s")
def get_predictions(
    output_dir: str,
    recipe: dict[str, Any],
    defect_lists: Optional[list[list[str]]] = None,
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
            "defect_id_list": defect_lists,
            "recipe": recipe,
        },
        timeout=TIMEOUT,
    )

    if r.json()["status"] == "completed":
        return r.json()["predictions_list"]
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")


# TODO: Split this into smaller functions
@st.cache_data(ttl="30s")
def get_lrf_data_lists(
    output_dir: str, cols: list[str], include_prob: bool = False, lot_id: str = ""
) -> list[list[dict[str, Any]]]:
    """
    Return lrf data with selected columns
    """
    params = {"output_dir": output_dir, "cols": ",".join(cols), "lot_id": lot_id}
    r = requests.get(f"{API_ROOT}result/get_lrf_data_lists", params=params, timeout=TIMEOUT)
    if r.json()["status"] == "error":
        logger.error(f"Error occurred when calling get LRF API (lrf): {r.json()['message']}")
        raise ValueError(f"Error occurred when calling get LRF API (lrf): {r.json()['message']}")
    else:
        lrf_data_list = r.json()["lrf_data"]
        for lrf_data in lrf_data_list:
            logger.info(f"LRF data of {len(lrf_data)} defects loaded from `{output_dir}`")

    r = requests.get(API_ROOT + "result/get_answer", json={"output_dir": output_dir, "lot_id": lot_id}, timeout=TIMEOUT)
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
        r = requests.get(
            API_ROOT + "result/get_probability", json={"output_dir": output_dir, "lot_id": lot_id}, timeout=TIMEOUT
        )

        if r.json()["status"] == "error":
            logger.error(f"Error occurred when calling get LRF API (prob): {r.json()['message']}")
            raise ValueError(f"Error occurred when calling get LRF API (prob): {r.json()['message']}")
        else:
            probability_lists = r.json()["probability_list"]
            lrf_data_with_prob_list = []
            for lrf_data_with_ans, probability_list in zip(lrf_data_with_ans_list, probability_lists):
                if len(lrf_data_with_ans) != len(probability_list):
                    logger.error(
                        f"""Difference in length between lrf data and
                         probability {len(lrf_data_with_ans)} {len(probability_list)}"""
                    )
                    raise ValueError(
                        f"""Difference in length between lrf data and
                         probability {len(lrf_data_with_ans)} {len(probability_list)}"""
                    )
                lrf_data_with_prob = []
                for data, prob in zip(lrf_data_with_ans, probability_list):
                    lrf_data_with_prob.append({**data, "Probability": prob})
                lrf_data_with_prob_list.append(lrf_data_with_prob)
            return lrf_data_with_prob_list
    else:
        return lrf_data_with_ans_list


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


# TODO: split this into two funtion: api request + ui update
# TODO: Seems unused?
def gen_lrf(
    model_id: str, output_dir: str, gen_lrf_type: str, threshold=None, top_k=None, key_number: int = 0, lot_id: str = ""
) -> None:
    if gen_lrf_type == "top_k":
        if top_k is not None and 1 <= top_k <= 999:
            if st.button(label=f"Generate new Model {model_id} lrf", key=f"gen_lrf_top_k_{key_number}"):
                request = request_top_k_lrf(output_dir=output_dir, top_k=top_k, lot_id=lot_id)

                if request.json().get("status") == "error":
                    code = request.json().get("code")
                    message = request.json().get("message")
                    st.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                    logger.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                else:
                    st.success(f"New .lrf file (top_k: {top_k}) generated at {output_dir}!")
                    logger.info(f"New .lrf file (top_k: {top_k}) generated at {output_dir}!")
        else:
            st.error(f"Top-k setting: {top_k} is invalid. Should be between 1 and 999 !")
    elif gen_lrf_type == "threshold":
        if threshold is not None and 0.0 <= threshold <= 1.0:
            if st.button(f"Generate new Model {model_id} lrf", key=f"gen_lrf_threshold_{key_number}"):
                request = request_threshold_lrf(output_dir=output_dir, confidence_threshold=threshold, lot_id=lot_id)

                if request.json().get("status") == "error":
                    code = request.json().get("code")
                    message = request.json().get("message")
                    st.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                    logger.error(f".lrf file not generated!\nError code: {code}\nError message: {message}")
                else:
                    st.success(f"New .lrf file (threshold: {threshold}) generated at {output_dir}!")
                    logger.info(f"New .lrf file (threshold: {threshold}) generated at {output_dir}!")
        else:
            st.error(f"Threshold setting: {threshold} is invalid. Should be between 0.0 and 1.0 !")
    else:
        raise NotImplementedError(f"gen_lrf_type {gen_lrf_type} is not implemented.")


def split_lrf(lrf_path: str, partitions: int, output_dir: str) -> requests.Response:
    r = requests.post(
        API_ROOT + "lrf_split",
        json={
            "lrf_path": lrf_path,
            "partitions": partitions,
            "output_dir": output_dir,
        },
        timeout=TIMEOUT,
    )

    if r.json()["status"] == "completed":
        logger.info("LRF split completed successfully!")
    else:
        logger.error(f"Error occurred when calling lrf_split API: {r.json()['message']}")

    return r


def merge_lrf(output_dir: str) -> requests.Response:
    r = requests.post(
        API_ROOT + "lrf_merge",
        json={
            "output_dir": output_dir,
        },
        timeout=TIMEOUT,
    )

    if r.json()["status"] == "completed":
        logger.info("LRF merge completed successfully!")
    else:
        logger.error(f"Error occurred when calling lrf_merge API: {r.json()['message']}")

    return r


def filter_lrf(lrf_path: str, output_dir: str, keep_defects_in_filter: bool, defect_filters: str) -> requests.Response:
    r = requests.post(
        API_ROOT + "lrf_filter",
        json={
            "lrf_path": lrf_path,
            "output_dir": output_dir,
            "keep_defects_in_filter": keep_defects_in_filter,
            "defect_filters": defect_filters,
        },
        timeout=TIMEOUT,
    )

    if r.json()["status"] == "completed":
        logger.info("LRF filter completed successfully!")
    else:
        logger.error(f"Error occurred when calling lrf_filter API: {r.json()['message']}")

    return r


def relabel_lrf(
    lrf_path: str, output_dir: str, relabel_mode: Literal["No/UniqueID", "ClassType"], relabel_map: dict[str, int]
) -> requests.Response:
    r = requests.post(
        API_ROOT + "lrf_relabel",
        json={
            "lrf_path": lrf_path,
            "output_dir": output_dir,
            "relabel_mode": relabel_mode,
            "relabel_map": relabel_map,
        },
        timeout=TIMEOUT,
    )

    if r.json()["status"] == "completed":
        logger.info("LRF re-label completed successfully!")
    else:
        logger.error(f"Error occurred when calling lrf_relabel API: {r.json()['message']}")

    return r


def get_model_data(
    output_dir: str,
) -> tuple[dict[str, Any], tuple[list[int], list[float], list[int]]] | tuple[None, None]:
    try:
        model_data_list = get_model_data_list(output_dir)
        db_metadata = model_data_list[0]
        defect_id_lists = model_data_list[1][0]
        probability_list = model_data_list[1][1]
        answer_list = model_data_list[1][2]
        return db_metadata[0], (defect_id_lists[0], probability_list[0], answer_list[0])
    except Exception as e:
        logger.warning(f"Error getting model data from {output_dir}! {e}")
        return None, None


class MultiLotModelData:
    def __init__(
        self,
        model_metadata_list: list[dict[str, Any]],
        defect_id_lists: list[list[int]],
        probability_lists: list[list[float]],
        answer_lists: list[list[int]],
    ):
        self.model_metadata_list = model_metadata_list
        self.defect_id_lists = defect_id_lists
        self.probability_lists = probability_lists
        self.answer_lists = answer_lists

    def __repr__(self) -> str:
        return f"""MultiLotModelData(model_metadata_list={self.model_metadata_list},
                defect_id_lists={self.defect_id_lists},
                probability_lists={self.probability_lists},
                answer_lists={self.answer_lists})"""


def get_multilot_model_data(
    output_dir: str,
) -> Optional[MultiLotModelData]:
    try:
        db_metadata = get_db_metadata_lists(output_dir=output_dir)
        defect_id_lists = get_defect_id_lists(output_dir=output_dir)
        probability_lists = get_probability(output_dir, defect_id_lists)
        answer_lists = get_answer(output_dir, defect_id_lists)
        assert len(defect_id_lists) == len(probability_lists), (
            f"IDs: {len(defect_id_lists)} Prob: {len(probability_lists)}"
        )
        assert len(defect_id_lists) == len(answer_lists), f"IDs: {len(defect_id_lists)} Ans: {len(answer_lists)}"

        return MultiLotModelData(db_metadata, defect_id_lists, probability_lists, answer_lists)
    except Exception as e:
        logger.warning(f"Error getting model data from {output_dir}! {type(e)} {e}")
        return None


def get_model_data_list(
    output_dir: str,
) -> list[tuple[dict[str, Any], tuple[list[int], list[float], list[int]]]] | list[tuple[None, None]]:
    try:
        db_metadata = get_db_metadata_lists(output_dir)
        defect_id_list = get_defect_id_lists(output_dir)
        probability_list = get_probability(output_dir, defect_id_list)
        answer_list = get_answer(output_dir, defect_id_list)
        return db_metadata, (defect_id_list, probability_list, answer_list)
    except Exception as e:
        logger.warning(f"Error getting model data from {output_dir}! {e}")
        return None, None


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


#####################################################################################################
# Inference / Multilot Inference                                                                    #
#####################################################################################################


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
def request_paginated_inference_status(page_size: int, current_page: int) -> pd.DataFrame:
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
                lambda x: f"{(x.components.days * 24 + x.components.hours):02}:{x.components.minutes:02}:{x.components.seconds:02}"
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
        paged_statuses_df["lot_info"] = paged_statuses_df["lot_info"].apply(
            lambda data_paths: pformat([per_lot.get("lot_id", "") for per_lot in data_paths["data_paths"]])
            if isinstance(data_paths, dict) and "data_paths" in data_paths and data_paths["data_paths"] is not None
            else None
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
                lambda x: f"{(x.components.days * 24 + x.components.hours):02}:{x.components.minutes:02}:{x.components.seconds:02}"
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
                lambda x: f"{(x.components.days * 24 + x.components.hours):02}:{x.components.minutes:02}:{x.components.seconds:02}"
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
                lambda x: f"{(x.components.days * 24 + x.components.hours):02}:{x.components.minutes:02}:{x.components.seconds:02}"
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
            "lot_count",
            "children_job_id",
            # 'Review Link',
            "output_dir",
            "gen_optimized_recipe",
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
def request_finetune(ff_core_training_request: "FFCoreTrainingRequest") -> requests.Response:
    """Call ff_core fine-tuning api.

    Args:
        ff_core_training_request: Refer to `FFCoreTrainingRequest` for details.

    Returns:
        The reponse of the API request.

    """
    # TODO: Check multilot_config is valid structure

    r = requests.post(
        f"{API_ROOT}{APIGroup.TRAINING}finetune", json=ff_core_training_request.model_dump(), timeout=TIMEOUT
    )

    status = r.json()["status"]

    if status == "started":
        logger.info("Model fine-tuning started running successfully!")
    else:
        logger.error(f"Error occurred when calling fine-tuning API: {r.json()['message']}")

    return r


def request_basetrain(ff_core_training_request: "FFCoreTrainingRequest") -> requests.Response:
    """Call ff_core base-training api.

    Args:
        ff_core_training_request: Refer to `FFCoreTrainingRequest` for details.

    Returns:
        The reponse of the API request.

    """
    # TODO: Check multilot_config is valid structure

    r = requests.post(
        f"{API_ROOT}{APIGroup.TRAINING}basetrain", json=ff_core_training_request.model_dump(), timeout=TIMEOUT
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
                lambda x: f"{(x.components.days * 24 + x.components.hours):02}:{x.components.minutes:02}:{x.components.seconds:02}"
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
def request_training_job_record(training_id: str) -> requests.Response:
    """Get training job records of the given `training_id`.

    Args:
        training_id: The training id of the target training job.

    Returns:
        The response of the API request.

    """
    r = requests.get(f"{API_ROOT}{APIGroup.TRAINING}job-record/{training_id}", timeout=TIMEOUT)

    return r.json()


@st.cache_data(ttl="1s")
def request_training_job_records(training_ids: list[str]) -> pd.DataFrame:
    """Get training job records of the given `training_ids`.

    Args:
        training_ids: List of training id to get record for.

    Returns:
        The response of the API request

    """
    training_job_records = {}
    for training_id in training_ids:
        training_job_records[training_id] = request_training_job_record(training_id)

    return format_training_job_records(pd.DataFrame.from_dict(training_job_records).T).T


def format_training_job_records(training_job_records_df: pd.DataFrame) -> pd.DataFrame:
    """Format and sort the training job records dataframe.

    Args:
        training_job_records_df: Dataframe containing raw training job status details.

    Returns:
        A processed training job record dataframe with adjusted timezones and formatted details.

    """
    if not training_job_records_df.empty:
        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        training_job_records_df["start_time"] = pd.to_datetime(
            training_job_records_df["start_time"], unit="s"
        ).dt.floor("s")
        training_job_records_df["start_time"] = (
            training_job_records_df["start_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
        )

        # Convert model name to user-readable format
        training_job_records_df["base_model_name"] = training_job_records_df["base_model_name"].apply(format_model_name)
        training_job_records_df["output_model_name"] = training_job_records_df["output_model_name"].apply(
            format_model_name
        )

        ### Readability format
        # Multilot_config (from yaml config)
        training_job_records_df["multilot_config"] = training_job_records_df["multilot_config"].map(pformat)

        # Epoch loss and validation loss
        if "debug" in training_job_records_df.columns:
            training_job_records_df["debug"] = training_job_records_df["debug"].map(pformat)

        training_job_records_df["hyper_params"] = training_job_records_df["hyper_params"].map(pformat)
        training_job_records_df["model_params"] = training_job_records_df["model_params"].map(pformat)

        # Rename index column so that detailed status table will show 'inference_id' instead of 'index'
        training_job_records_df = training_job_records_df.rename(columns={"index": "inference_id"})

        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        if "end_time" in training_job_records_df.columns:
            training_job_records_df["end_time"] = pd.to_datetime(
                training_job_records_df["end_time"], unit="s"
            ).dt.floor("s")
            training_job_records_df["end_time"] = (
                training_job_records_df["end_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")
            )

            # Calculate runtime only for rows that have end_time
            training_job_records_df["runtime"] = training_job_records_df.apply(
                lambda row: row["end_time"] - row["start_time"] if pd.notnull(row["end_time"]) else None, axis=1
            )

            training_job_records_df["runtime"] = training_job_records_df["runtime"].apply(
                lambda x: f"{(x.components.days * 24 + x.components.hours):02}:{x.components.minutes:02}:{x.components.seconds:02}"
                if pd.notnull(x)
                else None
            )

    # Change ordering
    sorted_training_job_records_df = training_job_records_df.reindex(
        columns=[
            "status",
            "progress",
            "start_time",
            "end_time",
            "runtime",
            "base_model_name",
            "model_params",
            "site",
            # "tool",
            "tech_layer",
            "layer_group",
            "output_model_name",
            "current_epoch",
            # "epochs",
            # "batch_size",
            # "learning_rate",
            # "optimizer_type",
            # "optimizer_params",
            # "loss_type",
            # "loss_params",
            # "lr_scheduler_type",
            # "lr_scheduler_params",
            "hyper_params",
            "multilot_config",
            "skip_particle_mode_defects",
            "debug",
            "message",
            "error_message",
        ]
    )

    # For columns not included above, just add them to the back.
    for column in training_job_records_df.columns:
        if column not in sorted_training_job_records_df.columns:
            sorted_training_job_records_df[column] = training_job_records_df[column]

    return sorted_training_job_records_df.astype(str)


@st.cache_data(ttl="1s")
def request_stop_job(job_id: str) -> str:
    r = requests.post(f"{API_ROOT}stop_job?job_id={job_id}", timeout=TIMEOUT)

    return f"{r.status_code}: {r.json().get('message', 'message not found...')}"


#####################################################################################################
# Data yaml                                                                                         #
#####################################################################################################
@st.cache_data(ttl="1s")
def is_valid_yaml_config(yaml_config: Any, mode: Literal["recipe", "lots"]) -> dict[str, Any]:
    r = requests.post(
        API_ROOT + "is_valid_yaml_format",
        json={
            "yaml_config": yaml_config,
            "mode": mode,
        },
        timeout=TIMEOUT,
    )

    return r.json()


@st.cache_data(ttl="1s")
def lrf_list_to_yaml(df: pd.DataFrame, original_str: str, replace_str: str) -> dict[str, Any]:
    r = requests.post(
        API_ROOT + "lrf_list_to_yaml",
        json={
            "lrf_list_raw_data": df.to_json(),
            "original_str": original_str,
            "replace_str": replace_str,
        },
        timeout=TIMEOUT,
    )

    return r.json()


@st.cache_data(ttl="1s")
def filter_data_yaml(
    original_data_yaml: dict[str, Any],
    filters: dict[str, Any],
    filter_lots_list: list[str],
    keep_lots: bool,
) -> dict[str, Any]:
    r = requests.post(
        API_ROOT + "filter_data_yaml",
        json={
            "original_data_yaml": original_data_yaml,
            "filters": filters,
            "filter_lots_list": filter_lots_list,
            "keep_lots": keep_lots,
        },
        timeout=TIMEOUT,
    )

    return r.json()


@st.cache_data(ttl="1s")
def parse_data_yaml(data_yaml: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(
        API_ROOT + "parse_data_yaml",
        json={
            "data_yaml": data_yaml,
        },
        timeout=TIMEOUT,
    )

    return r.json()


@st.cache_data(ttl="1s")
def generate_golden_set_from_data_yaml(
    data_yaml: dict[str, Any], output_dir: str, output_suffix: str
) -> dict[str, Any]:
    r = requests.post(
        API_ROOT + "generate_golden_set_from_data_yaml",
        json={
            "data_yaml": data_yaml,
            "output_dir": output_dir,
            "output_suffix": output_suffix,
        },
        timeout=TIMEOUT,
    )

    return r.json()


#####################################################################################################
# Model conversion                                                                                  #
#####################################################################################################
def convert_model(model_details: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(
        API_ROOT + "convert_model",
        json={
            "model_details": model_details,
        },
        timeout=TIMEOUT,
    )

    return r.json()
