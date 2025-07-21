# ruff: noqa: F841

import streamlit as st

from ltt_ff_frontend.regression_test_ui import recipe_converter, regression_runner


def app() -> None:
    st.title("Regression Test Dashboard")
    st.caption("Configure and run regression tests")
    mode = st.segmented_control(
        label="Mode", options=["Regression Runner", "Recipe Converter"], default="Regression Runner"
    )
    if mode is None:
        st.error("Please select a mode!")
        return

    if mode == "Regression Runner":
        regression_runner.app()
    elif mode == "Recipe Converter":
        recipe_converter.app()
