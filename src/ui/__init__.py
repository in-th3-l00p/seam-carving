import numpy as np
import streamlit as st
import cv2

from seam_carving import SeamCarving
from .components import caption as ui_caption, original_grid, resized_grid
from .state import init_session_state, reset_on_new_upload, store_carved_result


def run_app() -> None:
    st.set_page_config(
        page_title="seam carving demo",
        page_icon="🖼️",
        layout="centered",
    )

    ui_caption()
    init_session_state(st)

    uploaded = st.file_uploader(
        "upload image", type=["png", "jpg", "jpeg", "bmp", "webp"]
    )

    if uploaded is None:
        return

    reset_on_new_upload(st, uploaded)

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("could not read the uploaded image")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    st.subheader("original and next seams")
    original_grid(img_rgb)

    col1, col2 = st.columns(2)
    with col1:
        target_w = st.slider(
            "target width", min_value=1, max_value=int(w), value=int(max(1, w // 2))
        )
    with col2:
        target_h = st.slider(
            "target height", min_value=1, max_value=int(h), value=int(max(1, h // 2))
        )

    if target_w > w or target_h > h:
        st.warning("target size must be smaller than the original (only shrinking supported)")

    run = st.button("carve seams")
    if run and target_w <= w and target_h <= h:
        with st.spinner("carving seams..."):
            carver = SeamCarving(img_rgb)
            try:
                out = carver.shrink(int(target_w), int(target_h))
            except Exception as e:
                st.exception(e)
                out = None
        if out is not None:
            store_carved_result(st, out, (int(h), int(w)), (int(target_w), int(target_h)))

    carved = st.session_state.get("carved")
    if carved is not None:
        st.subheader("resized and next seams")
        resized_grid(carved)

