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

# persist carved result across reruns (e.g., after pressing download)
if "carved" not in st.session_state:
    st.session_state["carved"] = None
    st.session_state["carved_png"] = None
    st.session_state["carved_meta"] = None
    st.session_state["upload_sig"] = None

uploaded = st.file_uploader("upload image", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded is not None:
    # reset stored result when a new file is uploaded
    upload_sig = (getattr(uploaded, "name", None), getattr(uploaded, "size", None))
    if upload_sig != st.session_state.get("upload_sig"):
        st.session_state["upload_sig"] = upload_sig
        st.session_state["carved"] = None
        st.session_state["carved_png"] = None
        st.session_state["carved_meta"] = None

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("could not read the uploaded image")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # row 1: original + next seams (preview)
        st.subheader("original and next seams")
        try:
            orig_carver = SeamCarving(img_rgb)
            img_v = orig_carver.show_vertical()
            img_h = orig_carver.show_horizontal()
        except Exception as e:
            st.exception(e)
            img_v, img_h = None, None

        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("original")
            st.image(img_rgb, channels="RGB", width="content")
        with c2:
            st.caption("next vertical seam")
            if img_v is not None:
                st.image(img_v, channels="RGB", width="content")
        with c3:
            st.caption("next horizontal seam")
            if img_h is not None:
                st.image(img_h, channels="RGB", width="content")

        # sliders to change target dimensions (must be <= original)
        col1, col2 = st.columns(2)
        with col1:
            target_w = st.slider("target width", min_value=1, max_value=int(w), value=int(max(1, w // 2)))
        with col2:
            target_h = st.slider("target height", min_value=1, max_value=int(h), value=int(max(1, h // 2)))

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
                # store result in session so downloads/reruns don't clear it
                st.session_state["carved"] = out
                out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                ok, png_bytes = cv2.imencode(".png", out_bgr)
                st.session_state["carved_png"] = png_bytes.tobytes() if ok else None
                st.session_state["carved_meta"] = {
                    "orig_shape": (int(h), int(w)),
                    "target": (int(target_w), int(target_h)),
                }

        # always render carved preview if available
        carved = st.session_state.get("carved")
        if carved is not None:
            st.subheader("resized and next seams")
            try:
                res_carver = SeamCarving(carved)
                out_v = res_carver.show_vertical()
                out_h = res_carver.show_horizontal()
            except Exception as e:
                st.exception(e)
                out_v, out_h = None, None

            r1, r2, r3 = st.columns(3)
            with r1:
                st.caption("resized")
                st.image(carved, channels="RGB", width="content")
                # download persists across reruns
                if st.session_state.get("carved_png") is not None:
                    st.download_button(
                        label="download resized (PNG)",
                        data=st.session_state["carved_png"],
                        file_name="seam_carved.png",
                        mime="image/png",
                    )
            with r2:
                st.caption("next vertical seam (resized)")
                if out_v is not None:
                    st.image(out_v, channels="RGB", width="content")
            with r3:
                st.caption("next horizontal seam (resized)")
                if out_h is not None:
                    st.image(out_h, channels="RGB", width="content")
