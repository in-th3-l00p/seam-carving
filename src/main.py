import io
from typing import Tuple

import numpy as np
import streamlit as st
import cv2

from seam_carving import SeamCarving

st.set_page_config(
    page_title="seam carving demo",
    page_icon="🖼️",
    layout="centered",
)

st.title("seam carving demo")
st.caption("upload an image, choose a smaller target size, and carve seams")

uploaded = st.file_uploader("upload image", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("could not read the uploaded image")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        st.subheader("original")
        st.image(img_rgb, channels="RGB", width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            target_w = st.number_input("target width", min_value=1, max_value=w, value=max(1, w // 2))
        with col2:
            target_h = st.number_input("target height", min_value=1, max_value=h, value=max(1, h // 2))

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
                st.subheader("result")
                st.image(out, channels="RGB", width="stretch")

                # enable download
                out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                ok, png_bytes = cv2.imencode(".png", out_bgr)
                if ok:
                    st.download_button(
                        label="download result (PNG)",
                        data=png_bytes.tobytes(),
                        file_name="seam_carved.png",
                        mime="image/png",
                    )
