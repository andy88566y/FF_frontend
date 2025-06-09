from typing import Any, Literal

import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


"""
Shows a summary of OOS lots in a result directory.
The format of this summary mirrors the run test excel table done by the PE team.
"""


def calculate_avg_ffr_per_lot(filtered_inference_results: list[dict[str, Any]]) -> float:
    ffr_per_lot = []
    for lot in filtered_inference_results:
        true_negative = lot["true_negative"]
        false_defects = lot["as_is_non_defect_count"]
        false_filter_rate = true_negative / false_defects if false_defects > 0 else -1
        ffr_per_lot.append(false_filter_rate)
    return sum(ffr for ffr in ffr_per_lot) / len(ffr_per_lot)


# TODO: move calculations to backend
def calculate_oos_summary(
    inference_results: list[dict[str, Any]], mode: Literal["ALL", ">=150", "<150"]
) -> dict[str, Any]:
    logger.warning(inference_results[0])
    # for lot in inference_results:
    #     logger.warning(lot)

    if mode == ">=150":
        filtered_inference_results = [lot for lot in inference_results if lot["as_is_true_defect_count"] >= 150]
        logger.warning(len(filtered_inference_results))
    elif mode == "<150":
        filtered_inference_results = [lot for lot in inference_results if lot["as_is_true_defect_count"] < 150]
        logger.warning(len(filtered_inference_results))
    else:
        filtered_inference_results = inference_results
        logger.warning(f"LOT: {len(filtered_inference_results)}")

    as_is = sum(lot["as_is_defect_count"] for lot in filtered_inference_results)
    logger.warning(f"As-is: {as_is}")

    to_be = sum(lot["to_be_defect_count"] for lot in filtered_inference_results)
    logger.warning(f"To-be: {to_be}")

    # Calculate CR%
    true_positives = sum(lot["to_be_true_defect_count"] for lot in filtered_inference_results)
    true_defects = sum(lot["as_is_true_defect_count"] for lot in filtered_inference_results)
    capture_rate = true_positives / true_defects if true_defects > 0 else -1
    logger.warning(f"CR%: ({true_positives}/{true_defects}) {capture_rate * 100}%")

    # Calculate FFR%
    true_negative = sum(lot["true_negative"] for lot in filtered_inference_results)
    false_defects = sum(lot["as_is_non_defect_count"] for lot in filtered_inference_results)
    false_filter_rate = true_negative / false_defects if false_defects > 0 else -1
    logger.warning(f"FFR%: ({true_negative}/{false_defects}) {false_filter_rate * 100}%")

    avg_ffr_per_lot = calculate_avg_ffr_per_lot(filtered_inference_results)
    logger.warning(f"FFR per Lots: {avg_ffr_per_lot * 100}%")

    miss_catch = true_defects - true_positives
    logger.warning(f"A. Miss Catch: {miss_catch}")

    high_false = sum(1 for lot in filtered_inference_results if lot["to_be_non_defect_count"] > 150)
    logger.warning(f"B. High False: {high_false}")

    success_lots = sum(
        1
        for lot in filtered_inference_results
        if (lot["result"]["Capture Rate"] != "OOS" and lot["result"]["False Filter Rate"] != "OOS")
    )
    logger.warning(f"Success lots: {success_lots}")

    return {}


def gen(
    recipe: dict[str, Any],
    inference_result_dir: str,
) -> None:
    inference_results = api_helper.get_recipe_filtered_results_from_api(inference_result_dir, recipe=recipe)
    # for lot_result in inference_results:
    #     logger.warning(f"\n{lot_result}")

    calculate_oos_summary(inference_results, "ALL")
    # calculate_oos_summary(inference_results, ">=150")
    # calculate_oos_summary(inference_results, "<150")
    # df = pd.DataFrame(columns=["Lot ID", "Missed defects"])
    # for missed_defect_per_lot in missed_defects:
    #     for k, v in missed_defect_per_lot.items():
    #         df.loc[len(df)] = pd.Series({"Lot ID": k, "Missed defects": v})
    # st.dataframe(df)
