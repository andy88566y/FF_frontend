import glob
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import API_ROOT


def gap(size: int) -> None:
    '''
    Simple function to space out Streamlit UI elements.

    Args:
        size: The number of newlines.
    '''
    for _ in range(size):
        st.write('')


def return_status_style(status: str) -> tuple[str, int]:
    if status == "starting":
        return ("green", 10)
    elif status == "running":
        return ("green", 70)
    elif status == "completed":
        return ("green", 100)
    elif status == "error":
        return ("red", 0)
    else:
        return ("grey", 0)


@st.cache_data(ttl='10s')
def generate_defect_list(db_path: str, confidence_threshold: float) -> list[str]:
    '''
    From the inference result, filter images with probability higher than confidence threshold.
    These images are considered to be defects.

    Args:
        db_path: Absolute path to database containing inference results
        confidence_threshold: Images with defect probability lower than confidence threshold
                                is considered defective.

    Returns a list of ID numbers of the defect images.
    '''

    results = requests.post(API_ROOT+'filter_threshold', json={
                        "db_path": db_path,
                        "confidence_threshold": confidence_threshold,
                    }, timeout=10)

    status = results.json()['status']

    if status == 'started':
        logger.info("Threshold filter successfully started!")
    else:
        logger.error(f"Error occurred when calling inference API: {results.json()['message']}")

    return results.json()['defect_list']


@st.cache_data(ttl='5s')
def request_lrf(db_path: str,
                lot_id: str,
                output_dir: str,
                confidence_threshold: float) -> requests.Response:
    '''
    Calls FalseFilter API with use_cache=True.

    Args:
        lot_id: Name of the lot
        confidence_threshold: Images with defect probability lower than confidence threshold
                                is considered defective.

    Returns the reponse of the API request.
    '''
    r = requests.post(API_ROOT+'generate_lrf', json={
                        "db_path": db_path,
                        "lot_id": lot_id,
                        "output_dir": output_dir,
                        "threshold": confidence_threshold,
                    }, timeout=10)

    status = r.json()['status']

    if status == 'started':
        logger.info(".lrf generation requested successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return r

@st.cache_data(ttl='10s')
def request_inference(model_path: str, model_key_path: str, image_dir: str, lrf_path: str, lot_id: str, output_dir: str,
                      inference_batch_size: int, confidence_threshold: float, overwrite: bool) -> requests.Response:
    '''
    Calls FalseFilter API with use_cache=False.

    Args:
        lot_id: Name of the lot
        confidence_threshold: Images with defect probability lower than confidence threshold
                                is considered defective.
        overwrite: Whether to overwrite the output directory.

    Returns the reponse of the API request.
    '''
    r = requests.post(API_ROOT+'inference', json={
                        "model_path": model_path,
                        "model_key_path": model_key_path,
                        "image_dir": image_dir,
                        "output_dir": output_dir,
                        "lrf_path": lrf_path,
                        "lot_id": lot_id,
                        "threshold": confidence_threshold,
                        "batch_size": inference_batch_size,
                        "overwrite": overwrite,
                        "use_cache": False
                    }, timeout=10)

    status = r.json()['status']

    if status == 'started':
        logger.info("Inference started running successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return r


def get_files_by_format(format: str) -> list[str]:
    '''
    Search for all file names that end with the input format.

    Args:
        format: File extension (e.g. .db, .txt)

    Returns a list of absolute paths to the files that match the format.
    '''
    return glob.glob(f"{st.session_state.global_output_dir}/**/*{format}", recursive=True)


def get_all_lots() -> list[str]:
    '''
    Returns a list of files and directories in the output directory.
    '''
    return os.listdir(st.session_state.global_input_dir)


@st.cache_data(ttl='1s')
def request_inference_status(inference_id: str) -> requests.Response:
    '''
    Gets inference status by calling FalseFilter API

    Args:
      inference_id: Name of the inference job

    Returns the response of the API request
    '''
    r = requests.get(f"{API_ROOT}inference/status/{inference_id}", timeout=10)

    status = r.json()['status']

    # TODO: The check is NOT working as intended...
    if status == 'starting':
        logger.info(f"Inference status request for {inference_id} was successful!")
    elif status == 'running':
        logger.info(f"Inference status request for {inference_id} is ongoing...")
    elif status == 'completed':
        logger.info(f"Inference status request for {inference_id} is completed.")
    else:
        logger.error(f"Unknown status for {inference_id}: {r.json()['message']}")

    return r

@st.cache_data(ttl='1s')
def request_all_inference_statuses() -> str:
    '''
    Gets inference status for all inference jobs by calling FalseFilter API

    Returns the a list of statuses for all inference jobs.
    '''
    r = requests.get(f"{API_ROOT}inference/get_all_status", timeout=10)

    if r.json()['status'] == 'error':
        logger.error(r.json()['message'])
        return {}
    else:
        all_statuses = r.json()['result']

        for inference_job in all_statuses:
            logger.info(f'Status of inference request for {inference_job}: {all_statuses[inference_job]}')

        return all_statuses

@st.cache_data(ttl='10s')
def read_database(db_path: str) -> requests.Response:
    '''
    Calls DB API to read a database file.

    Args:
        db_dir: Directory where the .db file is stored.
        lot_id: The name of the lot of images.

    Returns the reponse of the API request, including the data read from the database,
    as a pandas dataframe.
    '''

    r = requests.post(API_ROOT+'read_database', json={
                        "db_path": db_path,
                    }, timeout=10)

    status = r.json()['status']

    if status == 'started':
        logger.info("DB read started running successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return pd.DataFrame.from_dict(r.json()['results'])

@st.cache_data(ttl='1s')
def get_prc_data(db_path: str, return_curve: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Get the data needed to draw a PRC curve.

    Args:
        db_path: Absolute path to the .db file containing inference results for
                desired lot of defect images.
        return_curve: If false, just return the area under the curve (AUPRC)
    '''
    r = requests.get(API_ROOT+'get_prc_data', json={
                        "db_path": db_path,
                        "return_curve": return_curve,
                    }, timeout=10)

    prc_data_list = r.json()['prc_data']
    prc_data_ndarray = tuple(np.array(data_list) for data_list in prc_data_list)

    return prc_data_ndarray

@st.cache_data(ttl='1s')
def get_roc_data(db_path: str, return_curve: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Get the data needed to draw a PRC curve.

    Args:
        db_path: Absolute path to the .db file containing inference results for
                desired lot of defect images.
        return_curve: If false, just return the area under the curve (AUPRC)
    '''
    r = requests.get(API_ROOT+'get_roc_data', json={
                        "db_path": db_path,
                        "return_curve": return_curve,
                    }, timeout=10)

    roc_data_list = r.json()['roc_data']
    roc_data_ndarray = tuple(np.array(data_list) for data_list in roc_data_list)

    return roc_data_ndarray

@st.cache_data(ttl='1s')
def get_defect_id(db_path: str) -> list[int]:
    '''
    Get list of defect IDs from a database.

    Args:
        db_path: Absolute path to the .db file containing inference results for
            desired lot of defect images.

        Returns:
            A list of the defect IDs of a lot of images.
    '''
    r = requests.get(API_ROOT+'get_defect_id', json={
                        "db_path": db_path,
                    }, timeout=10)

    return r.json()['defect_id_list']

@st.cache_data(ttl='1s')
def get_probability(db_path: str, defect_id: list[int]) -> list[float]:
    """
    Read a list of the defect probabilities from a database.

    Args:
        db_path: Absolute path to the .db file containing inference results for
            desired lot of defect images.
        defect_id: ID of the defect images

    Returns:
        A list of the defect probabilities of a lot of images.
    """
    r = requests.get(API_ROOT+'get_probability', json={
                        "db_path": db_path,
                        "defect_id_list": defect_id,
                    }, timeout=10)

    return r.json()['probability_list']

@st.cache_data(ttl='1s')
def get_answer(db_path: str, defect_id: list[int]) -> list[int]:
    """
    Read a list of the ground truths from a database.

    Args:
        db_path: Absolute path to the .db file containing inference results for
            desired lot of defect images.
        defect_id: ID of the defect images

    Returns:
        A list of the ground truths of a lot of images.
    """
    r = requests.get(API_ROOT+'get_answer', json={
                        "db_path": db_path,
                        "defect_id_list": defect_id,
                    }, timeout=10)

    return r.json()['answer_list']

@st.cache_data(ttl='10s')
def request_finetune(batch_size: int, epochs: int, lr: float, output_dir: str,
                     overwrite: bool, train_dir: str, train_lrf_path: str,
                     val_dir: str, val_lrf_path: str) -> requests.Response:
    '''
    Calls FFA model fine-tuning.

    Args:
        batch_size: Training batch size.
        epochs: Number of training epochs.
        lr: Learning rate.
        output_dir: Directory to save the fine-tuned model.
        overwrite: Whether to overwrite the existing output directory.
        train_dir: Directory containing the training data.
        val_dir: Directory containing the validation data.

    Returns the reponse of the API request.
    '''
    r = requests.post(API_ROOT+'train', json={
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "lr": lr,
                        "output_dir": output_dir,
                        "overwrite": overwrite,
                        "train_dir": train_dir,
                        "train_lrf_path": train_lrf_path,
                        "val_dir": val_dir,
                        "val_lrf_path": val_lrf_path
                    }, timeout=10)

    status = r.json()['status']

    if status == 'started':
        logger.info("Model fine-tuning started running successfully!")
    else:
        logger.error(f"Error occurred when calling fine-tuning API: {r.json()['message']}")

    return r

@st.cache_data(ttl='1s')
def request_all_finetuning_statuses():
    '''
    Gets finetuning status for all finetuning jobs by calling FalseFilter API

    Returns the a list of statuses for all finetuning jobs.
    '''
    r = requests.get(f"{API_ROOT}train/get_all_status", timeout=10)

    if r.json()['status'] == 'error':
        logger.error(r.json()['message'])
        return {}
    else:
        all_statuses = r.json()['result']

        for training_job in all_statuses:
            logger.info(f'Status of finetuning request for {training_job}: {all_statuses[training_job]["status"]}')

        return all_statuses

def get_base_models():
    return ["Model#1", "Model#2", "Model#3"]
