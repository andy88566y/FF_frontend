import os

import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_review import (
    diff_image_view,
    init_defect_review,
    label_mapping,
    list_view,
    map_view,
    recipe_view,
)


def app() -> None:
    st.title("Defect Review")

    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            init_defect_review.app()
        except ValueError as e:
            st.error(e)
            return

        ###Label info
        # TODO: don't hardcode this, instead get lrf type from st.session_state.selected_lot_id
        label_mapping.app(lrf_type="LTTSWADC")
        map_view.app()

    with col2:
        diff_image_view.app()
        recipe_view.app()

        ### List view
        if st.session_state.result_dir:
            if not os.path.isdir(st.session_state.result_dir):
                raise ValueError(f"Input Result directory in text field is invalid: {st.session_state.result_dir}")

            logger.info("Input field params encoded and stored in URL.")
            list_view.app(
                result_dir=st.session_state.result_dir,
                selected_lot_id=st.session_state.selected_lot_id,
                models_name=st.session_state.models_name,
                lrf_ext=st.session_state.selected_lot_lrf_ext,
                df=st.session_state.list_view_df,
            )
        else:
            list_view.app(
                result_dir="",
                selected_lot_id=st.session_state.selected_lot_id,
                models_name=[],
                lrf_ext=st.session_state.selected_lot_lrf_ext,
                df=st.session_state.list_view_df,
            )
