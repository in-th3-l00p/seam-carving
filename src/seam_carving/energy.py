import numpy as np
import cv2

class ImageEnergy:
    def __init__(self, image: np.ndarray):
        self._energy = np.zeros(image.shape[:2], dtype=np.float64)
        for channel in range(3):
            sobel_x = cv2.Sobel(image[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
            self._energy += np.abs(sobel_x) + np.abs(sobel_y)

    def get(self, x: int, y: int) -> float:
        return float(self._energy[y, x])

    def matrix(self) -> np.ndarray:
        return self._energy
