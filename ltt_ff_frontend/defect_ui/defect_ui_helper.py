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
def generate_defect_list(output_dir: str, lot_id: str, confidence_threshold: float) -> list[str]:
    '''
    From the inference result, filter images with probability lower than confidence threshold.
    These images are considered to be defects.

    Args:
        inference_result: A pd dataframe containing inference results
        confidence_threshold: Images with defect probability lower than confidence threshold
                                is considered defective.

    Returns a list of ID numbers of the defect images.
    '''
    # TODO: Remember to also change input params
    # inference_result = DBManager.ReadFromDB(output_dir, lot_id)
    # results = DBManager.FilterThreshold(inference_result, confidence_threshold)
    results = {}
    filtered_df = pd.DataFrame(results)
    filtered_df = filtered_df.reset_index()
    filtered_df['index'] = filtered_df['index'] + 1
    defect_list = filtered_df.loc[filtered_df["probabilities"] > confidence_threshold]
    defect_list = defect_list['index'].tolist()
    return defect_list


@st.cache_data(ttl='5s')
def request_lrf(image_dir:str,
                lot_id: str,
                lrf_path: str,
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
    r = requests.post(API_ROOT+'inference', json={
                        "root_dir": image_dir,
                        "output_dir": output_dir,
                        "lrf_path": lrf_path,
                        "lot_id": lot_id,
                        "threshold": confidence_threshold,
                        "batch_size": 4,
                        "overwrite": True,
                        "use_cache": True
                    }, timeout=600)

    status = r.json()['status']

    if status == 'started':
        logger.info("Inference started running successfully!")
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
                        "root_dir": image_dir,
                        "output_dir": output_dir,
                        "lrf_path": lrf_path,
                        "lot_id": lot_id,
                        "threshold": confidence_threshold,
                        "batch_size": inference_batch_size,
                        "overwrite": overwrite,
                        "use_cache": False
                    }, timeout=600)

    status = r.json()['status']

    if status == 'started':
        logger.info("Inference started running successfully!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return r


def get_files_by_format(fformat: str) -> list[str]:
    '''
    Search for all file names that end with the input format.

    Args:
        fformat: File extension (e.g. .db, .txt)

    Returns a list of absolute paths to the files that match the format.
    '''
    return glob.glob(f"{st.session_state.global_output_dir}/**/*{fformat}", recursive=True)


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
    r = requests.get(f"{API_ROOT}inference/status/{inference_id}", timeout=1000)

    status = r.json()['status']

    # TODO: The check is NOT working as intended...
    if status == 'starting':
        logger.info("Inference status request was successful!")
    else:
        logger.error(f"Error occurred when calling inference API: {r.json()['message']}")

    return r
