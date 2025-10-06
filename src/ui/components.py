import numpy as np
import streamlit as st

from seam_carving import SeamCarving


def caption() -> None:
    st.title("seam carving demo")
    st.caption(
        "upload an image, choose a smaller target size, and carve seams"
        " · by Tisca Catalin (intheloop) — "
        "[tiscacatalin.com](https://tiscacatalin.com)"
    )


def original_grid(img_rgb: np.ndarray) -> None:
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


def resized_grid(out_rgb: np.ndarray) -> None:
    try:
        res_carver = SeamCarving(out_rgb)
        out_v = res_carver.show_vertical()
        out_h = res_carver.show_horizontal()
    except Exception as e:
        st.exception(e)
        out_v, out_h = None, None

    r1, r2, r3 = st.columns(3)
    with r1:
        st.caption("resized")
        st.image(out_rgb, channels="RGB", width="content")
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

