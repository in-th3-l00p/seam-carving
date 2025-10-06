import numpy as np

def remove_vertical_seam(img: np.ndarray, seam_x: np.ndarray) -> np.ndarray:
    """Return a copy of img with one vertical seam removed (per-row x)."""
    h, w = img.shape[:2]
    if w <= 1:
        raise ValueError("cannot remove vertical seam from width=1 image")
    if img.ndim == 2:
        out = np.empty((h, w - 1), dtype=img.dtype)
        for y in range(h):
            x = seam_x[y]
            out[y, :x] = img[y, :x]
            out[y, x:] = img[y, x + 1 :]
        return out
    else:
        c = img.shape[2]
        out = np.empty((h, w - 1, c), dtype=img.dtype)
        for y in range(h):
            x = seam_x[y]
            out[y, :x, :] = img[y, :x, :]
            out[y, x:, :] = img[y, x + 1 :, :]
        return out

def remove_horizontal_seam(img: np.ndarray, seam_y: np.ndarray) -> np.ndarray:
    """Return a copy of img with one horizontal seam removed (per-column y)."""
    h, w = img.shape[:2]
    if h <= 1:
        raise ValueError("cannot remove horizontal seam from height=1 image")
    if img.ndim == 2:
        out = np.empty((h - 1, w), dtype=img.dtype)
        for x in range(w):
            y = seam_y[x]
            out[:y, x] = img[:y, x]
            out[y:, x] = img[y + 1 :, x]
        return out
    else:
        c = img.shape[2]
        out = np.empty((h - 1, w, c), dtype=img.dtype)
        for x in range(w):
            y = seam_y[x]
            out[:y, x, :] = img[:y, x, :]
            out[y:, x, :] = img[y + 1 :, x, :]
        return out
