
import itertools
import os
import re
import requests
import pandas as pd
import streamlit as st
from loguru import logger
from typing import Any
from collections import defaultdict

from ltt_ff_frontend.constant import API_ROOT
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import helper
from ltt_ff_frontend.regression_test_ui.regression_constant import (
    LAYERS,
    RUN_MODES,
    LAYERS_FABS,
    RULE_LAYERS_FABS
)


CR_COL = "(CR OOS lots)"
FFR_COL = "(FFR OOS lots)\nExclude flush"
AVG_FFR_COL = "avg ffr"
TABLE_COLUMNS = [CR_COL, FFR_COL]
RULE_TABLE_COLUMNS = [CR_COL, FFR_COL, AVG_FFR_COL]

ffr_check_memo: dict[str, Any] = defaultdict(list)

def save_false_filter_rates(test_case: str, resp: requests.Response) -> list:
    false_filter_rates = {
        stat["lot_id"]: stat["false_filter_rate"]
        for stat in resp.json()["filtered_stats"]
        if stat["false_filter_rate"] != -1
    }
    ffr_check_memo[test_case] = false_filter_rates
    return list(false_filter_rates.values())

def app() -> None:
    logger.debug("Loading Regression Result Viewer...")
    st.title("Regression Test Result Viewer")
    st.caption("Visualize Regression Test Result")
    r1_col1, r1_col2, r1_col3 = st.columns([3, 3, 1])

    output_dir_default = "/mnt/dbpc/xxx"
    with r1_col1:
        regression_result_dir = st.text_input("Regression Result Directory", value=output_dir_default)
        previous_week = st.text_input("Previous Week")
    with r1_col2:
        run_mode = st.segmented_control("Run Mode", RUN_MODES, default="Normal")
        if run_mode is None:
            st.error("Run Mode Option can not be None!")
            return
        current_week = st.text_input("Current Week")
    if previous_week and current_week:
        weeks = [f"W{previous_week}", f"W{current_week}"]
        rule_weeks = [f"W{current_week}", f"W{current_week}RULE"]
        rule_only_weeks = [f"W{current_week}RULEONLY"]
    else:
        return

    if run_mode == "Normal":
        prefixes = [f"{w}_{lf}" for lf, w in itertools.product(LAYERS_FABS, weeks)]
        regression_test_cases = defaultdict(list)
        for _dir in os.listdir(regression_result_dir):
            key = "_".join(_dir.split("_")[:3])
            regression_test_cases[key].append(_dir)
        regression_test_results = defaultdict(list)
        for i, prefix in enumerate(prefixes):
            total_lot_count = 0
            ffr_total_lot_count = 0
            cr_oos_count = 0
            ffr_oos_count = 0
            ffr_non_critical_count = 0

            if prefix not in regression_test_cases:
                st.error(
                    f"No test results with prefix '{prefix}' were found. "
                    "Please check if all the inference jobs were run and finished."
                )
            matched_test_cases = sorted(regression_test_cases[prefix])
            for test_case in matched_test_cases:
                inference_result_dir = os.path.join(regression_result_dir, test_case)
                r = requests.get(
                    API_ROOT + "result/get_filtered_stats",
                    json={"inference_result_dir": inference_result_dir},
                    timeout=15,
                )
                if r.ok:
                    save_false_filter_rates(test_case, r)
                    results = [filtered_stat["result"] for filtered_stat in r.json()["filtered_stats"]]

                    count = len(results)
                    total_lot_count += count

                    cr_oos_count += sum(1 for r in results if r["Capture Rate"] == "OOS")
                    if not re.match(r".*_flush$", test_case):
                        ffr_total_lot_count += count
                        ffr_oos_count += sum(1 for r in results if r["False Filter Rate"] == "OOS")
                        ffr_non_critical_count += sum(1 for r in results if r["False Filter Rate"] == "OOS, Non-Critical")
                else:
                    st.error(f" ======== {inference_result_dir} ======== \n")
            critical_and_non_critical_count = ffr_oos_count + ffr_non_critical_count

            week, layer, fab = prefix.split("_")
            regression_test_results[f"{week}#{layer}#{CR_COL}"].append(f"{fab}: {cr_oos_count}/{total_lot_count}")
            regression_test_results[f"{week}#{layer}#{FFR_COL}"].append(
                f"{fab}: {ffr_oos_count}({critical_and_non_critical_count})/{ffr_total_lot_count}"
            )
        table_summary = []
        headers = [f"{week} {col}" for col in TABLE_COLUMNS for week in weeks]
        for layer in LAYERS:
            row = []
            for col in TABLE_COLUMNS:
                for week in weeks:
                    numerator1, numerator2, denominator = 0, 0, 0
                    for info in regression_test_results[f"{week}#{layer}#{col}"]:
                        n, d = info.split()[-1].split("/")
                        denominator += int(d)
                        if col == CR_COL:
                            numerator1 += int(n)
                        else:
                            assert col == FFR_COL
                            n1, n2 = n[:-1].split("(")
                            numerator1 += int(n1)
                            numerator2 += int(n2)
                    if col == CR_COL:
                        row.append(f"{numerator1}/{denominator}")
                    else:
                        assert col == FFR_COL
                        row.append(f"{numerator1}({numerator2})/{denominator}")
            table_summary.append(row)
        st.subheader("Regression Results Summary")
        st.dataframe(pd.DataFrame(table_summary, columns=headers, index=LAYERS))

