"""
Модель объектива: пиксель кадра -> сантиметры вдоль конвейера.

Камера — «рыбий глаз»: у краёв кадра картинка сжата (1 пиксель ~ 0.7-0.9 см),
в центре — нет (1 пиксель ~ 0.26 см). Модель объектива это учитывает.
Масштаб задаёт красный маркер (его длина — в config.toml).
Параметры (calibration.json) получены в репозитории без нейросети (analization, calibrate_motion.py).
"""

import json

import numpy as np

W, H = 2688, 1520                         # размер кадра камеры
CY = H / 2
NORM = np.hypot(W, H) / 2
MARKER_PX = ((1089, 515), (1471, 515))    # концы маркера на кадре


class Lens:
    def __init__(self, k1, k2, cx, marker_length_cm):
        self.k1, self.k2, self.cx = float(k1), float(k2), float(cx)
        m = self.undistort(np.array(MARKER_PX, float))
        self.cm_per_unit = float(marker_length_cm) / np.linalg.norm(m[1] - m[0])

    def undistort(self, pts):
        d = np.asarray(pts, float) - (self.cx, CY)
        r2 = (d ** 2).sum(-1, keepdims=True) / NORM ** 2
        return d / (1 + self.k1 * r2 + self.k2 * r2 ** 2)

    def s(self, x, y):
        """Точка кадра (x, y) -> сантиметры вдоль конвейера от центра объектива.
        x и y — числа или массивы одинаковой формы."""
        pts = np.stack(np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float)), -1)
        return self.undistort(pts)[..., 0] * self.cm_per_unit

    @classmethod
    def load(cls, path, marker_length_cm):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["k1"], d["k2"], d["cx"], marker_length_cm)
