import numpy as np

from .energy import ImageEnergy
from .utils.dp import dp_vertical, dp_horizontal
from .utils.seams import remove_vertical_seam, remove_horizontal_seam

# provides the implementation of the seam carving resizer using dp
class SeamCarving:
  def __init__(self, image):
    self.__image = image
    self.__energy = ImageEnergy(image)
    E = self.__energy.matrix()  # 2D (h, w) float

    self.__vertical, self.__vertical_steps = dp_vertical(E)
    self.__horizontal, self.__horizontal_steps = dp_horizontal(E)
  
  def _trace_min_vertical_seam(self) -> np.ndarray:
    """return x of the seam per row"""
    seam_dp = self.__vertical
    steps = self.__vertical_steps
    h, w = seam_dp.shape
    x = int(np.argmin(seam_dp[-1, :]))
    seam_x = np.empty(h, dtype=np.int32)
    seam_x[-1] = x
    for y in range(h-2, -1, -1):
        x = x + int(steps[y+1, x])
        seam_x[y] = x
    return seam_x

  def show_vertical(self, color=(255, 0, 0), thickness=1) -> np.ndarray:
    img = self.__image.copy()
    h, _ = self.__vertical.shape
    seam_x = self._trace_min_vertical_seam()

    for y in range(h):
        x = int(seam_x[y])
        x0 = max(0, x - thickness // 2)
        x1 = min(img.shape[1], x0 + thickness)
        img[y, x0:x1] = color  # paint pixels
    return img
  
  def _trace_min_horizontal_seam(self) -> np.ndarray:
      """return y of the seam per row"""
      seam_dp = self.__horizontal
      steps = self.__horizontal_steps
      h, w = seam_dp.shape
      y = int(np.argmin(seam_dp[:, -1]))
      seam_y = np.empty(w, dtype=np.int32)
      seam_y[-1] = y
      for x in range(w-2, -1, -1):
          y = y + int(steps[y, x+1])
          seam_y[x] = y
      return seam_y

  def show_horizontal(self, color=(255, 0, 0), thickness=1) -> np.ndarray:
    img = self.__image.copy()
    _, w = self.__horizontal.shape
    seam_y = self._trace_min_horizontal_seam()

    for x in range(w):
        y = int(seam_y[x])
        y0 = max(0, y - thickness // 2)
        y1 = min(img.shape[0], y0 + thickness)
        img[y0:y1, x] = color
    return img
  
  def show_horizontal(self, color=(255, 0, 0), thickness=1) -> np.ndarray:
    img = self.__image.copy()
    _, w = self.__horizontal.shape
    seam_y = self._trace_min_horizontal_seam()
    for x in range(w):
        y = int(seam_y[x])
        y0 = max(0, y - thickness // 2)
        y1 = min(img.shape[0], y0 + thickness)
        img[y0:y1, x] = color
    return img

  def _recompute(self):
    """recompute energy + DP tables from current image"""
    self.__energy = ImageEnergy(self.__image)
    E = self.__energy.matrix()
    self.__vertical, self.__vertical_steps = dp_vertical(E)
    self.__horizontal, self.__horizontal_steps = dp_horizontal(E)

  def pop_vertical(self) -> np.ndarray:
    """remove the minimal vertical seam; returns the updated image"""
    seam_x = self._trace_min_vertical_seam()
    self.__image = remove_vertical_seam(self.__image, seam_x)
    self._recompute()
    return self.__image

  def pop_horizontal(self) -> np.ndarray:
    """remove the minimal horizontal seam; returns the updated image"""
    seam_y = self._trace_min_horizontal_seam()
    self.__image = remove_horizontal_seam(self.__image, seam_y)
    self._recompute()
    return self.__image
  
  def shrink(self, target_width: int, target_height: int) -> np.ndarray:
    """
    carve seams until the image reaches (target_width, target_height)
    only shrinking is supported here
    returns the new image
    """
    h, w = self.__image.shape[:2]
    if target_width > w or target_height > h:
        raise ValueError("shrink() only supports reducing size")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("invalid target size")

    need_v = w - target_width      # vertical seams to remove
    need_h = h - target_height     # horizontal seams to remove
    if need_v == 0 and need_h == 0:
        return self.__image

    # interleave removals to limit distortion (Bresenham-like scheduler)
    # greedy demo implementation
    removed_v = removed_h = 0
    total = need_v + need_h
    for i in range(total):
        # decide which seam to remove next to match the ratio need_v:need_h
        # remove vertical if vertical progress is behind its quota
        if need_v > 0 and (need_h == 0 or (removed_v * need_h) <= (removed_h * need_v)):
            self.pop_vertical()
            removed_v += 1
        else:
            self.pop_horizontal()
            removed_h += 1

    return self.__image