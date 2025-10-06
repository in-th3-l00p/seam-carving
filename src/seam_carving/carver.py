import numpy as np

from .energy import ImageEnergy
from .utils.dp import dp_vertical, dp_horizontal
from .utils.seams import remove_vertical_seam, remove_horizontal_seam

class SeamCarving:
    def __init__(self, image: np.ndarray):
        self._image = image
        self._energy = ImageEnergy(image)
        E = self._energy.matrix()
        self._vertical, self._vertical_steps = dp_vertical(E)
        self._horizontal, self._horizontal_steps = dp_horizontal(E)

    def _trace_min_vertical_seam(self) -> np.ndarray:
        seam_dp = self._vertical
        steps = self._vertical_steps
        h, w = seam_dp.shape
        x = int(np.argmin(seam_dp[-1, :]))
        seam_x = np.empty(h, dtype=np.int32)
        seam_x[-1] = x
        for y in range(h - 2, -1, -1):
            x = x + int(steps[y + 1, x])
            seam_x[y] = x
        return seam_x

    def _trace_min_horizontal_seam(self) -> np.ndarray:
        seam_dp = self._horizontal
        steps = self._horizontal_steps
        h, w = seam_dp.shape
        y = int(np.argmin(seam_dp[:, -1]))
        seam_y = np.empty(w, dtype=np.int32)
        seam_y[-1] = y
        for x in range(w - 2, -1, -1):
            y = y + int(steps[y, x + 1])
            seam_y[x] = y
        return seam_y

    def show_vertical(self, color=(255, 0, 0), thickness=1) -> np.ndarray:
        img = self._image.copy()
        h, _ = self._vertical.shape
        seam_x = self._trace_min_vertical_seam()
        for y in range(h):
            x = int(seam_x[y])
            x0 = max(0, x - thickness // 2)
            x1 = min(img.shape[1], x0 + thickness)
            img[y, x0:x1] = color
        return img

    def show_horizontal(self, color=(255, 0, 0), thickness=1) -> np.ndarray:
        img = self._image.copy()
        _, w = self._horizontal.shape
        seam_y = self._trace_min_horizontal_seam()
        for x in range(w):
            y = int(seam_y[x])
            y0 = max(0, y - thickness // 2)
            y1 = min(img.shape[0], y0 + thickness)
            img[y0:y1, x] = color
        return img

    def _recompute(self) -> None:
        self._energy = ImageEnergy(self._image)
        E = self._energy.matrix()
        self._vertical, self._vertical_steps = dp_vertical(E)
        self._horizontal, self._horizontal_steps = dp_horizontal(E)

    def pop_vertical(self) -> np.ndarray:
        seam_x = self._trace_min_vertical_seam()
        self._image = remove_vertical_seam(self._image, seam_x)
        self._recompute()
        return self._image

    def pop_horizontal(self) -> np.ndarray:
        seam_y = self._trace_min_horizontal_seam()
        self._image = remove_horizontal_seam(self._image, seam_y)
        self._recompute()
        return self._image

    def shrink(self, target_width: int, target_height: int) -> np.ndarray:
        h, w = self._image.shape[:2]
        if target_width > w or target_height > h:
            raise ValueError("shrink() only supports reducing size")
        if target_width <= 0 or target_height <= 0:
            raise ValueError("invalid target size")

        need_v = w - target_width
        need_h = h - target_height
        if need_v == 0 and need_h == 0:
            return self._image

        removed_v = removed_h = 0
        total = need_v + need_h
        for _ in range(total):
            if need_v > 0 and (need_h == 0 or (removed_v * need_h) <= (removed_h * need_v)):
                self.pop_vertical()
                removed_v += 1
            else:
                self.pop_horizontal()
                removed_h += 1

        return self._image
