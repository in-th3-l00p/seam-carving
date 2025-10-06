from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CarvedResult:
    image_rgb: np.ndarray
    png_bytes: Optional[bytes]
    orig_shape: Tuple[int, int]
    target_shape: Tuple[int, int]


def init_session_state(st) -> None:
    if "carved" not in st.session_state:
        st.session_state["carved"] = None
        st.session_state["carved_png"] = None
        st.session_state["carved_meta"] = None
        st.session_state["upload_sig"] = None


def reset_on_new_upload(st, uploaded) -> None:
    upload_sig = (getattr(uploaded, "name", None), getattr(uploaded, "size", None))
    if upload_sig != st.session_state.get("upload_sig"):
        st.session_state["upload_sig"] = upload_sig
        st.session_state["carved"] = None
        st.session_state["carved_png"] = None
        st.session_state["carved_meta"] = None


def store_carved_result(st, image_rgb: np.ndarray, orig_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> None:
    import cv2

    st.session_state["carved"] = image_rgb
    ok, png_bytes = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    st.session_state["carved_png"] = png_bytes.tobytes() if ok else None
    st.session_state["carved_meta"] = {
        "orig_shape": (int(orig_shape[0]), int(orig_shape[1])),
        "target": (int(target_shape[0]), int(target_shape[1])),
    }

