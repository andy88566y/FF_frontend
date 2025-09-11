import pandas as pd
import streamlit as st


def app() -> None:
    if st.session_state.result_dir and st.session_state.defect_number:
        row_number = st.session_state.filtered_df.index.get_loc(int(st.session_state.defect_number))
        selected_defect_prob = st.session_state.defect_prob["probability_list"][0][row_number]

        table_data = {
            "Model Name": st.session_state.models_name,
            "Defect Probability / Threshold": [
                f"{selected_defect_prob[i]:.3f}/{st.session_state.models_threshold[i]}"
                for i in range(st.session_state.model_count)
            ],
            "Defect Probability / Threshold_C": [
                f"{selected_defect_prob[i]:.3f}/{st.session_state.models_threshold_c[i]}"
                for i in range(st.session_state.model_count)
            ],
        }

        table_df = pd.DataFrame(table_data)

        # Apply styling and hide index
        styled_df = table_df.style.set_table_styles([{"selector": "th, td", "props": [("text-align", "center")]}]).hide(
            axis="index"
        )

        # Convert to HTML
        html_table = styled_df.to_html()

        # Display using Streamlit
        st.title("Defect Probability vs Threshold Table")
        st.markdown(html_table, unsafe_allow_html=True)
