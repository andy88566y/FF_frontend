import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def app() -> None:
    logger.debug("Opening Inference API page")

    st.title("BDDetector False Filtering API")
    st.caption("API for running inference and fine-tuning False Filtering models.")

    st.header("Inference")
    with st.expander("Run inference on images"):

        st.markdown(":violet[Runs inference on images in the specified directory.]")

        st.header("Parameters")

        image_dir = st.text_input('Image directory', value='', help='The directory that contains the Images folder.')
        lrf_path = st.text_input('.lrf path', value='', help='Absolute path to the selected .lrf file.')
        lot_id = st.text_input('Lot ID', value='', help='Name of the lot of images to run inference on.')
        output_dir = st.text_input('Output directory', value='', help='The directory to store generated .lrf and .db')
        inference_batch_size = st.select_slider("Inference batch size", [4, 8, 16, 32])

        # selected_lot = st.selectbox(
        #     "Which lot do you want to run inference on?",
        #     helper.get_all_lots()
        # )
        # helper.gap(2)

        slider_threshold = st.slider("Select confidence threshold:", 0.0, 1.0, 0.5)
        st.caption(f"Probabilities above :blue[{slider_threshold}] will be considered defects.")
        helper.gap(2)

        # # add model selection here in the future
        # abs_model_paths = ...
        # model_basename: list[str] = []
        # selected_model = st.selectbox(
        #     "[Placeholder] Which model do you want to run?",
        #     model_basename
        # )
        # helper.gap(2)

        overwrite = st.toggle(label="Overwrite files in output directory", value=True)
        helper.gap(2)

        if st.button("Run inference", type='primary'):
            request = helper.request_inference(image_dir,
                                           lrf_path,
                                           lot_id,
                                           output_dir,
                                           inference_batch_size,
                                           slider_threshold,
                                           overwrite)
            if request.json().get('status') == 'error':
                code = request.json().get('code')
                message = request.json().get('message')
                st.text(f'Error code: {code}\nError message: {message}')
            else:
                inference_id = request.json().get('inference_id')
                st.text(f'Inference ID: {inference_id}')

            # st.divider()
            # # TODO: check with Carl on how to get inference progress
            # st.warning("TODO: Update progress bar")
            # progress_text = 'Inference in progress...'
            # progress_bar = st.progress(0.0, text=progress_text)

    with st.expander("Current jobs"):
        infer_id = st.text_input("Inference ID", help="The unique number generated after click Generate .lrf")
        if st.button("Check status"):
            status = helper.request_inference_status(infer_id)
            st.json(status.json())

    if st.button('Check all inference jobs'):
        all_statuses = helper.request_all_inference_statuses()
        for status in all_statuses:
            st.json(status)
