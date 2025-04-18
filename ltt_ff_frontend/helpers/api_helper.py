from typing import Any, Optional, Union

import numpy as np
import requests
import streamlit as st
from loguru import logger
from ltt_ff_frontend.constant import API_ROOT, BLANK_MODEL, TIMEOUT

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

def calculate_recipe_filtered_results(output_dir: str, recipe: dict[str, Any]) -> dict[str, Any]:
    defect_id_list = get_defect_id_lists(output_dir)
    answer_list = get_answer(output_dir=output_dir, defect_id=defect_id_list)[0]
    prediction_list = get_predictions(output_dir=output_dir, recipe=recipe, defect_list=defect_id_list)[0]

    positive = answer_list.count(1)
    negative = answer_list.count(0)
    unlabeled = answer_list.count(-1)
    true_positive = sum(1 for pred, ans in zip(prediction_list, answer_list) if pred == 1 and ans == 1)
    false_positive = sum(1 for pred, ans in zip(prediction_list, answer_list) if pred == 1 and ans == 0)
    true_negative = sum(1 for pred, ans in zip(prediction_list, answer_list) if pred == 0 and ans == 0)
    false_negative = sum(1 for pred, ans in zip(prediction_list, answer_list) if pred == 0 and ans == 1)
    filtered_unlabeled_defect_count = sum(
        1 for pred, ans in zip(prediction_list, answer_list) if pred == 1 and ans == -1
    )

    as_is_defect_count = positive + negative + unlabeled
    to_be_defect_count = true_positive + false_positive + filtered_unlabeled_defect_count

    capture_rate = true_positive / positive if positive > 0 else -1
    false_filter_rate = true_negative / negative if negative > 0 else -1
    filter_rate = 1 - (to_be_defect_count / as_is_defect_count) if as_is_defect_count > 0 else -1

    return {
        "as_is_defect_count": as_is_defect_count,
        "to_be_defect_count": to_be_defect_count,
        "filter_rate": filter_rate,
        "as_is_true_defect_count": positive,
        "to_be_true_defect_count": true_positive,
        "capture_rate": capture_rate,
        "as_is_non_defect_count": negative,
        "to_be_non_defect_count": false_positive,
        "false_filter_rate": false_filter_rate,
        "unlabeled": unlabeled,
        "filtered_unlabeled_defect_count": filtered_unlabeled_defect_count,
    }

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

# TODO: split this into two funtion: api request + ui update
def gen_lrf(
    model_id: str,
    output_dir: str,
    gen_lrf_type: str,
    threshold=None,
    top_k=None,
    key_number: int = 0,
    lot_id: str = ""
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
                request = request_threshold_lrf(
                    output_dir=output_dir, confidence_threshold=threshold, lot_id=lot_id
                )

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

def get_model_data(
    output_dir: str
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
        answer_lists: list[list[int]]
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
) -> Union[MultiLotModelData, None]:
    try:
        db_metadata = get_db_metadata_lists(output_dir=output_dir)
        defect_id_lists = get_defect_id_lists(output_dir=output_dir)
        probability_lists = get_probability(output_dir, defect_id_lists)
        answer_lists = get_answer(output_dir, defect_id_lists)
        assert len(defect_id_lists) == len(probability_lists), f"IDs: {len(defect_id_lists)} Prob: {len(probability_lists)}"
        assert len(defect_id_lists) == len(answer_lists), f"IDs: {len(defect_id_lists)} Ans: {len(answer_lists)}"

        return MultiLotModelData(db_metadata, defect_id_lists, probability_lists, answer_lists)
    except Exception as e:
        logger.warning(f"Error getting model data from {output_dir}! {type(e)} {e}")
        return None, None

def get_model_data_list(
    output_dir: str
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
