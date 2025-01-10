import pandas as pd
import streamlit as st

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def dummy_get_progress(status):
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

def get_color(status):
    if status in ["starting", "running", "completed"]:
        return "green"
    elif status == "error":
        return "red"
    else:
        return "grey"

def app() -> None:
    st.title("BDDetector False Filtering API")
    st.caption("API for running inference and fine-tuning False Filtering models.")

    st.header("Fine-tuning")
    with st.expander("Run model fine-tuning"):
        st.markdown(":violet[Fine-tune a model by re-training it using images in the specified directory.]")

        st.header("Parameters")

        training_batch_size = st.select_slider("Inference batch size", [4, 8, 16, 32])
        epochs = st.number_input('Epochs', value=10)
        learning_rate = st.number_input('Learning rate', value=0.0001, step=0.0001, format="%0.4f")
        output_dir = st.text_input('Output directory', value='', help='The directory to store the re-trained model.')
        overwrite = st.toggle('Overwrite existing model', value=False, help='Overwrite the old model that has the same name as the re-trained model.')
        st.caption(f":red[If overwrite=True, everything in output_dir will be deleted. Will be fixed before V2.]")
        train_dir = st.text_input('Training images directory', value='', help='Directory containing training data.')
        train_lrf_path = st.text_input('Training images .lrf path', value='', help='Path to the .lrf file for training images.')
        val_dir = st.text_input('Validation images directory', value='', help='Directory containing validation data.')
        val_lrf_path = st.text_input('Validation images .lrf path', value='', help='Path to the .lrf file for validation images.')
        if st.button('Start fine-tuning', type='primary'):
            request = helper.request_finetune(batch_size=training_batch_size,
                                              epochs=epochs,
                                              lr=learning_rate,
                                              output_dir=output_dir,
                                              overwrite=overwrite,
                                              train_dir=train_dir,
                                              train_lrf_path=train_lrf_path,
                                              val_dir=val_dir,
                                              val_lrf_path=val_lrf_path)

            if request.json().get('status') == 'error':
                code = request.json().get('code')
                message = request.json().get('message')
                st.text(f'Error code: {code}\nError message: {message}')
            else:
                training_id = request.json().get('training_id')
                st.text(f'Training ID: {training_id}')

    if st.button('Check all fine-tuning jobs'):
        all_statuses = helper.request_all_finetuning_statuses()

        brief = ["training_id", "status",'progress_bar',"estimated_time_remaining","current_epoch","total_epochs","epoch_loss","val_loss"]
        reorder = ["training_id","start_time", "status", "progress", "current_epoch", "total_epochs", "estimated_time_remaining", "output_dir", "train_dir", "val_dir", "epoch_loss", "val_loss", "best_model_path", "message", "error_message"]

        detail_status_df = pd.DataFrame(columns = reorder)

        for training_id, status in all_statuses.items():
            status['training_id'] = training_id
            new_row = pd.DataFrame([status])
            detail_status_df = pd.concat([detail_status_df, new_row], ignore_index=True)

        for column in detail_status_df.columns:
            if detail_status_df[column].dtype == 'object':
                detail_status_df[column] = detail_status_df[column].astype(str)

        detail_status_df['start_time'] = pd.to_datetime(detail_status_df['start_time'], unit='s')
        detail_status_df['start_time'] = detail_status_df['start_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
        detail_status_df['progress_bar'] = detail_status_df['status'].apply(dummy_get_progress)
        detail_status_df['color'] = detail_status_df['status'].apply(get_color)
        detail_status_df = detail_status_df.iloc[::-1].reset_index(drop=True)
        status_df = detail_status_df[brief]

        st.session_state.status_df_fin = status_df
        st.session_state.detail_fin = detail_status_df[reorder]

    progress_column = st.column_config.ProgressColumn(
        label="progress_bar",
        min_value=0,
        max_value=100
    )

    if "status_df_fin" not in st.session_state:
        st.session_state.status_df_fin = pd.DataFrame(columns = ["status",'progress_bar',"estimated_time_remaining","current_epoch","total_epochs","epoch_loss","val_loss"])
    if "detail_fin" not in st.session_state:
        st.session_state.detail_fin = pd.DataFrame(columns = ["status", "progress", "current_epoch", "total_epochs", "estimated_time_remaining", "output_dir", "train_dir", "val_dir", "epoch_loss", "val_loss", "best_model_path", "message", "error_message"])

    start_index = 0
    end_index = 10
    page_size = 10

    # Pagination settings
    if not st.session_state.status_df_fin.empty:
        page_size = 10
        page_number = st.number_input('Page number', min_value=1, value=1, step=1)
        start_index = (page_number - 1) * page_size
        end_index = page_number * page_size

    # Selection to find more detail
    event_fin = st.dataframe(
        st.session_state.status_df_fin.iloc[start_index:end_index],
        key = "statuses_finetuning",
        on_select = "rerun",
        selection_mode = "multi-row",
        use_container_width=True,
        column_config={"progress_bar": progress_column}
    ) if not st.session_state.status_df_fin.empty else st.write("")

    if event_fin and event_fin.selection:
        if event_fin.selection["rows"]:
            selected_indices = [start_index + st.session_state.status_df_fin.index[i] for i in event_fin.selection["rows"]]
            selected_rows = st.session_state.detail_fin.loc[selected_indices]

            transposed_detail = selected_rows.T
            st.dataframe(transposed_detail, use_container_width= True )
