import glob
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import API_ROOT, ALLOWED_LRF_TYPES, TIMEOUT


def gap(size: int) -> None:
    '''
    Simple function to space out Streamlit UI elements.

    Args:
        size: The number of newlines.
    '''
    for _ in range(size):
        st.write('')


def return_status_style(status: str) -> int:
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

def return_finetune_status_style(status: str, current_epoch: int, total_epochs: int) -> int:
    if status == "starting":
        return 10
    elif status == "running":
        progress = ((current_epoch / total_epochs) * 90) + 10
        return int(progress)
    elif status == "completed":
        return 100
    elif status == "error":
        return 0
    else:
        return 0

def format_model_name(name: str) -> str:
    if name == '' or '/' not in name:
        logger.warning("Trying to format empty string or a string without /, returning empty string...")
        return ''
    model_type, model_name = name.split("/")
    return f"[{model_type}] {model_name.replace('.encrypted', '').replace('.pth', '').replace('#', ' ')}"

def filter_details(name: str, detail: str) -> str:
    split_model_name = name.split(' ')
    if detail == 'site':
        return split_model_name[1]
    elif detail == 'tool':
        return split_model_name[2]
    elif detail == 'tech_layer':
        return split_model_name[3]
    elif detail == 'layer_group':
        return split_model_name[4]

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
                    }, timeout=TIMEOUT)

    status = results.json()['status']

    if status == 'started':
        logger.info("Threshold filter successfully started!")
    else:
        logger.error(f"Error occurred when calling inference API: {results.json()['message']}")

    return results.json()['defect_list']


@st.cache_data(ttl='5s')
def request_lrf(output_dir: str,
                lot_id: str,
                model_name: str,
                confidence_threshold: float) -> requests.Response:
    '''
    Calls FalseFilter API with use_cache=True.

    Args:
        output_dir: Output root directory. The generated lrf will be stored in output_dir/LRF/
        lot_id: Name of the lot
        model_name: Name of the inference model.
        confidence_threshold: Images with defect probability lower than confidence threshold
                                is considered defective.

    Returns the reponse of the API request.
    '''
    r = requests.post(API_ROOT+'generate_lrf', json={
                        "output_dir": output_dir,
                        "lot_id": lot_id,
                        "model_name": model_name,
                        "threshold": confidence_threshold,
                    }, timeout=TIMEOUT)

    status = r.json()['status']

    if status == 'started':
        logger.info(".lrf generation requested successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return r

@st.cache_data(ttl='10s')
def request_inference(base_model: str, image_dir: str, lrf_path: str, lot_id: str, output_dir: str,
                      inference_batch_size: int = 32, confidence_threshold: float = 0.174, overwrite: bool = False) -> requests.Response:
    '''
    Calls FalseFilter API to run inference.

    Args:
        model_name: Name of inference model.
        image_dir: Directory containing defect images.
        lrf_path: Absolute path to the .lrf file for the defect images.
        lot_id: Name of the lot of defect images.
        output_dir: Directory to store the generated database file and filtered .lrf file.
        inference_batch_size: Inference batch size. Higher batch size: faster but requires more memory.
        confidence_threshold: Images with defect probability higher than confidence threshold
                                is considered defective.
        overwrite: Whether to overwrite the existing .db and .lrf files of the same name.

    Returns the reponse of the API request.
    '''
    r = requests.post(API_ROOT+'inference', json={
                        "model_name": base_model,
                        "image_dir": image_dir,
                        "lrf_path": lrf_path,
                        "lot_id": lot_id,
                        "output_dir": output_dir,
                        "threshold": confidence_threshold,
                        "batch_size": inference_batch_size,
                        "overwrite": overwrite,
                        "use_cache": False
                    }, timeout=TIMEOUT)

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
    r = requests.get(f"{API_ROOT}inference/status/{inference_id}", timeout=TIMEOUT)

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
    r = requests.get(f"{API_ROOT}inference/get_all_status", timeout=TIMEOUT)

    if r.json()['status'] == 'error':
        logger.error(r.json()['message'])
        return {}
    else:
        all_statuses = r.json()['result']

        for inference_job in all_statuses:
            logger.info(f'Status of inference request for {inference_job}: {all_statuses[inference_job]}')

        return all_statuses

@st.cache_data(ttl='1s')
def request_paginated_inference_status(page_size: int, current_page: int) -> str:
    '''
    Gets pagainated inference status by calling FalseFilter API

    Args:
        page_size : the number of entries to be shown on the dataframe
        current_page : the page that is current requested

    Returns the response of the API request
    '''
    r = requests.get(f"{API_ROOT}inference/get_paginated_status?page_size={page_size}&current_page={current_page}", timeout=TIMEOUT)
    logger.debug(r)
    if r.json()['status'] == 'error':
        logger.error(r.json()['message'])
        return {}
    else:
        paged_statuses = r.json()['result']

        for inference_job in paged_statuses:
            logger.info(f'Status of inference request for {inference_job}: {paged_statuses[inference_job]}')

        return paged_statuses

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
                    }, timeout=TIMEOUT)

    status = r.json()['status']

    if status == 'started':
        logger.info("DB read started running successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return pd.DataFrame.from_dict(r.json()['results'])

@st.cache_data(ttl='1s')
def get_prc_data(output_dir: str, lot_id: str, model_name:str, return_curve: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Get the data needed to draw a PRC curve.

    Args:
        output_dir: Root output directory of inference resuits.
        lot_id: Name of the lof of defect images.
        model_name: Name of inference results.
        return_curve: If false, just return the area under the curve (AUPRC)
    '''
    r = requests.get(API_ROOT+'get_prc_data', json={
                        "output_dir": output_dir,
                        "lot_id": lot_id,
                        "model_name": model_name,
                        "return_curve": return_curve,
                    }, timeout=TIMEOUT)

    prc_data_list = r.json()['prc_data']
    prc_data_ndarray = tuple(np.array(data_list) for data_list in prc_data_list)

    return prc_data_ndarray

@st.cache_data(ttl='1s')
def get_roc_data(output_dir: str, lot_id: str, model_name:str, return_curve: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Get the data needed to draw an ROC curve.

    Args:
        output_dir: Root output directory of inference resuits.
        lot_id: Name of the lof of defect images.
        model_name: Name of inference results.
        return_curve: If false, just return the area under the curve (AUROC)
    '''
    r = requests.get(API_ROOT+'get_roc_data', json={
                        "output_dir": output_dir,
                        "lot_id": lot_id,
                        "model_name": model_name,
                        "return_curve": return_curve,
                    }, timeout=TIMEOUT)

    roc_data_list = r.json()['roc_data']
    roc_data_ndarray = tuple(np.array(data_list) for data_list in roc_data_list)

    return roc_data_ndarray

@st.cache_data(ttl='1s')
def get_defect_id(output_dir: str, lot_id: str, model_name: str) -> list[int]:
    '''
    Get list of defect IDs from a database.

    Args:
        output_dir: Root output directory where inference results were stored.
        lot_id: Name of the lot of defect images.
        model_name: Name of model used to run inference.

        Returns:
            A list of the defect IDs of a lot of images.
    '''
    r = requests.get(API_ROOT+'get_defect_id', json={
                        "output_dir": output_dir,
                        "lot_id": lot_id,
                        "model_name": model_name,
                    }, timeout=TIMEOUT)

    if r.json()['status'] == 'completed':
        logger.info("DB read started running successfully!")
        return r.json()['defect_id_list']
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")

@st.cache_data(ttl='1s')
def get_probability(output_dir: str, lot_id: str, model_name: str, defect_id: list[int]) -> list[float]:
    """
    Read a list of the defect probabilities from a database.

    Args:
        output_dir: Root output directory where inference results were stored.
        lot_id: Name of the lot of defect images.
        model_name: Name of model used to run inference.
        defect_id: ID of the defect images

    Returns:
        A list of the defect probabilities of a lot of images.
    """
    r = requests.get(API_ROOT+'get_probability', json={
                        "output_dir": output_dir,
                        "lot_id": lot_id,
                        "model_name": model_name,
                        "defect_id_list": defect_id,
                    }, timeout=TIMEOUT)

    if r.json()['status'] == 'completed':
        logger.info("DB read started running successfully!")
        return r.json()['probability_list']
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")

@st.cache_data(ttl='1s')
def get_answer(output_dir: str, lot_id: str, model_name: str, defect_id: list[int]) -> list[int]:
    """
    Read a list of the ground truths from a database.

    Args:
        output_dir: Root output directory where inference results were stored.
        lot_id: Name of the lot of defect images.
        model_name: Name of model used to run inference.
        defect_id: ID of the defect images

    Returns:
        A list of the ground truths of a lot of images.
    """
    r = requests.get(API_ROOT+'get_answer', json={
                        "output_dir": output_dir,
                        "lot_id": lot_id,
                        "model_name": model_name,
                        "defect_id_list": defect_id,
                    }, timeout=TIMEOUT)

    if r.json()['status'] == 'completed':
        logger.info("DB read started running successfully!")
        return r.json()['answer_list']
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")
        raise ValueError(f"Error occurred when calling inference API: {r.json()['message']}")

@st.cache_data(ttl='10s')
def request_finetune(base_model: str,
                     model_naming: tuple[str, str, str, str],
                     multilot_config: dict,
                     epochs: int,
                     lr: float) -> requests.Response:
    '''
    Calls FFA model fine-tuning.

    Args:
        base_model: Name of model to finetune.
        model_naming: Details to be used for re-trained model (site, tool, tech layer, layer group)
        multilot_config: Dict containing training data info (lot id, lrf path, image dir)
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns the reponse of the API request.
    '''
    r = requests.post(API_ROOT+'train', json={
                        "base_model_name": base_model,
                        "batch_size": 32,
                        "epochs": epochs,
                        "learning_rate": lr,
                        "model_naming": model_naming,
                        "training_info": multilot_config,
                    }, timeout=TIMEOUT)

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
    r = requests.get(f"{API_ROOT}train/get_all_status", timeout=TIMEOUT)

    if r.json()['status'] == 'error':
        logger.error(r.json()['message'])
        return {}
    else:
        all_statuses = r.json()['result']

        for training_job in all_statuses:
            logger.info(f'Status of finetuning request for {training_job}: {all_statuses[training_job]["status"]}')

        return all_statuses

@st.cache_data(ttl='1s')
def get_base_models() -> list[str]:
    '''
    Returns a list of all available models to be used for inference or fine-tuning.
    '''
    r = requests.get(f'{API_ROOT}get_model_list', timeout=TIMEOUT)

    if r.json()['status'] == 'error':
        logger.error(r.json()['message'])
        return []
    else:
        base_model_list = r.json()['model_list']
        logger.info(f'List of base models: {base_model_list}')
        return base_model_list

def request_paginated_finetuning_status(page_size: int, current_page: int) -> dict[str, Any]:
    '''
    Gets pagainated inference status by calling FalseFilter API

    Args:
        page_size : the number of entries to be shown on the dataframe
        current_page : the page that is current requested

    Returns the response of the API request
    '''
    r = requests.get(f"{API_ROOT}train/get_paginated_status?page_size={page_size}&current_page={current_page}", timeout=TIMEOUT)
    logger.debug(r)
    if r.json()['status'] == 'error':
        logger.error(r.json()['message'])
        return {}
    else:
        paged_statuses = r.json()['result']

        for training_job in paged_statuses:
            logger.info(f'Status of finetuning request for {training_job}: {paged_statuses[training_job]}')

        return paged_statuses

def check_valid_lrf_in_yaml(yaml_config: dict) -> bool:
    '''
    Ensures all .lrf files listed in the yaml config file are valid
    (i.e. exists, has correct file extension, is lableled)

    Args:
        yaml_config: Dict containing training info such as lrf path, lot id, training image dir

    Returns true if there are no invalid lrf files found (check passed), and returns false if
    an invalid lrf file is found (check failed).
    '''
    # Extract all lrf paths from the yaml config
    lrf_paths = [batch['lrf_path'] for batch in yaml_config['data_paths']]

    for lrf_path in lrf_paths:
        lrf_string = os.path.basename(lrf_path)

        # Check .lrf file extension
        if re.search(r".lrf$", lrf_string):
            lrf_string = re.sub(r".lrf$", "", lrf_string)
        else:
            logger.error(f'Invalid lrf path found in .yaml config file: {lrf_path}. Check file extension.')
            raise ValueError(f'Invalid lrf path found in .yaml config file: {lrf_path}. Check file extension.')

        # Check invalid lrf types (lrf is not an allowed type and lrf is not 'filtered' type)
        if not lrf_string.endswith(tuple(ALLOWED_LRF_TYPES)) and re.search(r'_filtered_\d{6}$', lrf_string) is None:
            logger.error(f'''Invalid lrf path found in .yaml config file: {lrf_path}.
                         Check that it belongs to one of these types: {ALLOWED_LRF_TYPES}''')
            raise ValueError(f'''Invalid lrf path found in .yaml config file: {lrf_path}.
                         Check that it belongs to one of these types: {ALLOWED_LRF_TYPES}''')

        return True
