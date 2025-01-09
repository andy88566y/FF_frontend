import streamlit as st

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


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

        if not all_statuses:
            st.error('No ongoing training jobs!')
        else:
            for status in all_statuses:
                st.json(all_statuses[status])
