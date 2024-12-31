import glob
import os

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


@st.cache_data(ttl='10s')
def generate_defect_list(inference_result: pd.DataFrame, confidence_threshold: float) -> list[str]:
    '''
    From the inference result, filter images with probability higher than confidence threshold.
    These images are considered to be defects.

    Args:
        inference_result: A pd dataframe containing inference results
        confidence_threshold: Images with defect probability lower than confidence threshold
                                is considered defective.

    Returns a list of ID numbers of the defect images.
    '''
    inference_result_dict = inference_result.to_dict()

    results = requests.post(API_ROOT+'filter_threshold', json={
                        "inference_result": inference_result_dict,
                        "confidence_threshold": confidence_threshold,
                    }, timeout=10)

    status = results.json()['status']

    if status == 'started':
        logger.info("Threshold filter successfully started!")
    else:
        logger.error(f"Error occurred when calling inference API: {results.json()['message']}")

    filtered_df = pd.DataFrame(results.json()['results'])
    filtered_df = filtered_df.reset_index()
    filtered_df['index'] = filtered_df['index'] + 1
    defect_list = filtered_df.loc[filtered_df["probabilities"] > confidence_threshold]
    defect_list = defect_list['index'].tolist()
    return defect_list


@st.cache_data(ttl='5s')
def request_lrf(lot_id: str,
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
def request_inference(image_dir: str, lrf_path: str, lot_id: str, output_dir: str,
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
def request_all_statuses() -> str:
    '''
    Gets inference status for all inference jobs by calling FalseFilter API

    Returns the a list of statuses for all inference jobs.
    '''
    r = requests.get(f"{API_ROOT}inference/get_all_status", timeout=10)

    all_statuses = r.json()
    for status in all_statuses:
        logger.info(f'Status of inference request for {status["inference_id"]}: {status["status"]}')

    return r.json()

@st.cache_data(ttl='10s')
def read_database(db_dir: str, lot_id: str) -> requests.Response:
    '''
    Calls DB API to read a database file.

    Args:
        db_dir: Directory where the .db file is stored.
        lot_id: The name of the lot of images.

    Returns the reponse of the API request, including the data read from the database,
    as a pandas dataframe.
    '''

    r = requests.post(API_ROOT+'read_database', json={
                        "db_dir": db_dir,
                        "lot_id": lot_id,
                    }, timeout=10)

    status = r.json()['status']

    if status == 'started':
        logger.info("DB read started running successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return pd.DataFrame.from_dict(r.json()['results'])
