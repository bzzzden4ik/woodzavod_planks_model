"""
Поиск досок обученной моделью (YOLO, сегментация).

Модель работает с полосой конвейера (строки BAND_TOP..BAND_TOP+BAND_H кадра) —
так же, как при обучении. Для каждой доски возвращается её контур и осевая линия.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

BAND_TOP, BAND_H = 450, 700      # полоса конвейера в кадре (как в tools/build_dataset.py)
IMGSZ = 1280                     # ширина входа модели (как при обучении)
MIN_CONF = 0.25                  # слабее — не показываем вовсе
MIN_COLS = 200                   # объект уже 200 пикселей — не доска


@dataclass
class Detection:
    conf: float
    x0: int                      # левый и правый край по x (пиксели кадра)
    x1: int
    xs: np.ndarray               # столбцы x0..x1
    yc: np.ndarray               # осевая линия доски в каждом столбце (y кадра, NaN — нет доски)
    thick: np.ndarray            # толщина доски в столбце, пикс
    contour: np.ndarray          # контур для рисования (координаты кадра)

    @property
    def width(self):
        return self.x1 - self.x0

    def y_at_right(self):
        """y осевой линии у правого торца."""
        return float(np.nanmedian(self.yc[-15:]))


def pick_device(device="auto"):
    """'auto' -> '0' (видеокарта NVIDIA), если PyTorch её видит, иначе 'cpu'."""
    device = str(device).strip().lower()
    if device != "auto":
        return device
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def describe_device(device="auto"):
    """Понятное название устройства для вывода на экран."""
    d = pick_device(device)
    if d == "cpu":
        return "процессор"
    import torch
    if not torch.cuda.is_available():
        return f"видеокарта {d} — НО PyTorch не видит видеокарту (установлен вариант для процессора?)"
    return f"видеокарта {torch.cuda.get_device_name(int(d.split(',')[0]))}"


class Detector:
    def __init__(self, model_path, device="auto"):
        self.model = YOLO(str(model_path))
        self.device = pick_device(device)

    def __call__(self, frame):
        band = frame[BAND_TOP:BAND_TOP + BAND_H]
        r = self.model.predict(band, imgsz=IMGSZ, rect=True, conf=MIN_CONF, retina_masks=True,
                               device=self.device, verbose=False)[0]
        out = []
        if r.masks is None:
            return out
        rows = np.arange(BAND_H)[:, None]
        for mk, c in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.conf.cpu().numpy()):
            cols = np.nonzero(mk.any(0))[0]
            if len(cols) < MIN_COLS:
                continue
            x0, x1 = int(cols.min()), int(cols.max())
            sub = mk[:, x0:x1 + 1]
            cnt = sub.sum(0)
            yc = np.where(cnt > 0, (sub * rows).sum(0) / np.maximum(cnt, 1), np.nan) + BAND_TOP
            cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(cs, key=cv2.contourArea) + (0, BAND_TOP)
            out.append(Detection(float(c), x0, x1, np.arange(x0, x1 + 1), yc, cnt.astype(float), contour))
        return out
