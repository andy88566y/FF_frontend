import json

import streamlit as st

from ltt_ff_frontend.constant import INFERENCE_DEFAULT_RESULT_DIR, ResultViewerComponents
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import (
    helper,
    prob_2d_distribution_fig,
    prob_distribution_fig,
    roc_fig,
    upset_plot,
    venn_diagram,
)


UPSET_SORT_OPTIONS = ["default", "ascending", "descending"]


def get_require_component_lists(length: int) -> dict[str, bool]:
    if length == 0:
        return []
    elif length == 1:
        return {ResultViewerComponents.ONE_D_DEFECT_DISTRIBUTION_CHART.value: True, ResultViewerComponents.CR_FFR_CURVE.value: True}
    else:
        return {ResultViewerComponents.MULTI_MODEL_PROBABILITIES.value: True}


def app() -> None:
    result_dir = st.text_input("Inference Result Directory", value=INFERENCE_DEFAULT_RESULT_DIR)

    # Early return if any output dir is invalid
    if not helper.is_valid_output_dir(result_dir):
        st.error(f"Inference Result Directory is invalid: {result_dir}")
        return

    db_metadata_list = api_helper.get_db_metadata_lists(output_dir=result_dir)
    recipe_list = [metadata["recipe"] for metadata in db_metadata_list]
    if all(recipe == recipe_list[0] for recipe in recipe_list):
        db_recipe = json.loads(recipe_list[0])
    else:
        st.error("Not all lots use same recipe.")
        return
    select_col, start_col = st.columns([8, 2])
    model_map = helper.get_model_map(db_recipe)
    with select_col:
        selected = st.multiselect("Select Models", model_map.keys())

    selected_count = len(selected)
    if selected_count == 0:
        return

    rvc_setting = (result_dir, selected)
    if "rvc_result" not in st.session_state or st.session_state.rvc_result["setting"] != rvc_setting:
        with start_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if not st.button("Compare"):
                return
        try:
            st.session_state.rvc_result = {
                "result": api_helper.get_result_viewer_components(
                    inference_result_dir=result_dir,
                    required_components=get_require_component_lists(selected_count),
                    required_input={"model_ids": [model_map[model]["id"] for model in selected]},
                ),
                "setting": rvc_setting,
            }
        except ValueError as e:
            st.error(e)
            return
    result_viewer_components = st.session_state.rvc_result["result"]

    col1, col2, col3, _ = st.columns([1, 1, 1, 6])
    
    if selected_count == 1:
        with col1:
            st.markdown("<br>", unsafe_allow_html=True)
            split_lot = st.toggle(label="Results split by lots", value=False)
        col_1d_chart_column, col_roc_curve_column = st.columns(2)
        with col_1d_chart_column:
            st.plotly_chart(
                prob_distribution_fig.gen(
                    aggregated_lists=result_viewer_components["one_d_defect_distribution_chart"],
                    selected_model=model_map[selected[0]],
                    split_lot=split_lot,
                )
            )
        with col_roc_curve_column:
            roc_fig.gen(
                model=model_map[selected[0]], roc_data=result_viewer_components["cr_ffr_curve"], split_lot=split_lot
            )
    elif selected_count == 2:
        with col1:
            st.markdown("<br>", unsafe_allow_html=True)
            split_lot = st.toggle(label="Results split by lots", value=False)
        _, plot_container, _ = st.columns([1, 8, 1])
        with plot_container:
            st.plotly_chart(
                prob_2d_distribution_fig.gen(
                    aggregated_model_data=[result_viewer_components["multi_model_probabilities"]],
                    model_1=model_map[selected[0]],
                    model_2=model_map[selected[1]],
                    split_lot=split_lot,
                    mode="Combined"
                )
            )
    elif selected_count == 3:
        with col1:
            threeD_mode = st.selectbox(label="3D Mode", options=["Venn3", "Upset"])
        if threeD_mode == "Venn3":
            venn_diagram.gen(
                aggregated_model_data=result_viewer_components["multi_model_probabilities"],
                intersections=result_viewer_components["intersections"],
                model_names=selected,
                split_lot=False #temporally unused,
            )

        else:
            with col2:
                sort_by = st.selectbox(label="Sort By", options=UPSET_SORT_OPTIONS)
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                exclude_zero = st.toggle("Exclude Empty Subsets")

            upset_plot.gen(
                aggregated_model_data=result_viewer_components["multi_model_probabilities"],
                intersections=result_viewer_components["intersections"],
                model_names=selected,
                sort_by=sort_by,
                split_lot=False, #temporally unused
                exclude_zero=exclude_zero,
            )

    else:
        with col1:
            sort_by = st.selectbox(label="Sort By", options=UPSET_SORT_OPTIONS)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            exclude_zero = st.toggle("Exclude Empty Subsets")

        upset_plot.gen(
            aggregated_model_data=result_viewer_components["multi_model_probabilities"],
            intersections=result_viewer_components["intersections"],
            model_names=[model_map[select]["model_hash"] for select in selected],
            split_lot=False, #temporally unused
            sort_by=sort_by,
            exclude_zero=exclude_zero,
        )
