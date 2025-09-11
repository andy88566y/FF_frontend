import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_review.defect_diff_viewer import draw_diff_img_plotly


def app() -> None:
    if st.session_state.defect_number and st.session_state.data_yaml:
        if st.session_state.selected_lot_lrf_ext == "lrf":
            draw_diff_img_plotly(
                data_yaml_path=st.session_state.data_yaml,
                lot_id=st.session_state.selected_lot_id,
                defect_id=st.session_state.defect_number,
                norm=st.session_state.norm,
                diff_clip=0.3,
            )
        else:
            row_index = st.session_state.filtered_df.index[
                st.session_state.filtered_df["No"] == int(st.session_state.defect_number)
            ]
            unique_id = st.session_state.filtered_df.loc[row_index[0], "UniqueID"]
            draw_diff_img_plotly(
                data_yaml_path=st.session_state.data_yaml,
                lot_id=st.session_state.selected_lot_id,
                defect_id=unique_id,
                norm=st.session_state.norm,
                diff_clip=0.3,
            )
