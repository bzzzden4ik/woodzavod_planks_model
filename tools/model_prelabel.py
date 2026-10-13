"""
Черновая разметка кадров обученной моделью (для круга 2 в Label Studio).

Модель видит полосу конвейера (строки BAND), найденные контуры переводятся
обратно в координаты полного кадра.

Запуск (из папки machine_learning):
    .venv\\Scripts\\python.exe tools\\model_prelabel.py --model runs/segment/draft_r1/weights/best.pt
        --tasks labeling/tasks_round2.json --out labeling/tasks_round2_model.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from build_dataset import BAND, FRAMES

ROOT = Path(__file__).resolve().parent.parent      # папка проекта (machine_learning), откуда бы ни запускали


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25, help="порог уверенности (ниже — больше находок)")
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    model = YOLO(args.model)
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    preview = ROOT / "labeling" / "preview_model"
    preview.mkdir(parents=True, exist_ok=True)
    n_boards = 0
    for i, t in enumerate(tasks):
        name = t["data"]["image"].split("frames/")[-1]
        img = cv2.imread(str(FRAMES / name))
        h, w = img.shape[:2]
        band = img[BAND[0]:BAND[1]]
        res = model.predict(band, imgsz=args.imgsz, conf=args.conf, rect=True, verbose=False, retina_masks=True)[0]
        result = []
        if res.masks is not None:
            for poly in res.masks.xy:                       # контуры в пикселях полосы
                if len(poly) < 3:
                    continue
                c = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2), 3.0, True).reshape(-1, 2)
                if len(c) < 3:
                    continue
                c[:, 1] += BAND[0]
                result.append({
                    "from_name": "board", "to_name": "image", "type": "polygonlabels",
                    "original_width": w, "original_height": h,
                    "value": {"points": [[float(x) / w * 100, float(y) / h * 100] for x, y in c],
                              "polygonlabels": ["доска"]}})
        n_boards += len(result)
        t["predictions"] = [{"model_version": Path(args.model).parent.parent.name, "result": result}]
        if i % 20 == 0:
            vis = img.copy()
            for r in result:
                pts = np.array([[p[0] * w / 100, p[1] * h / 100] for p in r["value"]["points"]], np.int32)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 4)
            cv2.imwrite(str(preview / name), cv2.resize(vis[BAND[0]:BAND[1]], None, fx=0.5, fy=0.5))
    Path(args.out).write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    print(f"Кадров: {len(tasks)}, найдено досок: {n_boards} -> {args.out}")


if __name__ == "__main__":
    main()
