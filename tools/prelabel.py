"""
Черновая разметка кадров для Label Studio.

  Доски  — контуры (многоугольники): сравнение с пустым конвейером + длинные
           горизонтальные полосы (как в analization/process_video.py).
  Людей не размечаем: контур доски и так покрывает только видимую часть доски.

"Пустой конвейер" для кадра — медиана соседних по времени кадров того же видео.

Результат:
  labeling/tasks.json           — задания для импорта в Label Studio (с черновой разметкой)
  labeling/preview/*.jpg        — несколько кадров с нарисованной разметкой для проверки

Запуск (из папки machine_learning):
    .venv\\Scripts\\python.exe tools\\prelabel.py
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # папка проекта (machine_learning), откуда бы ни запускали

FRAMES = ROOT / "data" / "frames"
OUT = ROOT / "labeling"
BAND = (600, 1080)          # строки, где ищем доски (с запасом: доску кладут наискосок)
DIFF = 25
H_RUN = 81                  # доска — горизонтальная полоса длиннее этого, пикс
MIN_BOARD_W = 500           # в черновик — только уверенные (длинные) контуры
MAX_BOARD_H = 140
MIN_FILL = 0.5              # доска — сплошная полоса, а не "дырявое" пятно
CONVEYOR_Y = (680, 1000)    # центр доски должен быть в зоне конвейера
NEIGHBOURS = 7              # для "пустого конвейера" берём по 7 соседних кадров с каждой стороны


def board_polygons(gray, empty):
    y0, y1 = BAND
    band, ref = gray[y0:y1], empty[y0:y1]
    m = ((cv2.absdiff(band, ref) > DIFF) & (band > 70)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, H_RUN), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 41), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    polys = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w < MIN_BOARD_W or h > MAX_BOARD_H or area / (w * h) < MIN_FILL:
            continue
        if not CONVEYOR_Y[0] <= y0 + y + h / 2 <= CONVEYOR_Y[1]:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 3.0, True).reshape(-1, 2) + (0, y0)
        if len(c) >= 3:
            polys.append(c)
    return polys


def main():
    meta = json.loads((FRAMES / "frames.json").read_text(encoding="utf-8"))
    by_video = defaultdict(list)
    for m in meta:
        by_video[m["video"]].append(m)

    tasks = []
    (OUT / "preview").mkdir(parents=True, exist_ok=True)
    n_boards = 0
    for video, items in by_video.items():
        items = sorted(items, key=lambda m: m["second"])
        grays = [cv2.imread(str(FRAMES / m["file"]), cv2.IMREAD_GRAYSCALE)[BAND[0]:BAND[1]] for m in items]
        for j, m in enumerate(items):
            # "Пустой конвейер" — медиана соседних по времени кадров: учитывает освещение
            # в это время, а движущиеся доски в медиане пропадают.
            near = grays[max(0, j - NEIGHBOURS):j + NEIGHBOURS + 1]
            empty = np.zeros((1520, 2688), np.uint8)
            empty[BAND[0]:BAND[1]] = np.median(np.array(near), axis=0).astype(np.uint8)
            img = cv2.imread(str(FRAMES / m["file"]))
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = []
            for poly in board_polygons(gray, empty):
                result.append({
                    "from_name": "board", "to_name": "image", "type": "polygonlabels",
                    "original_width": w, "original_height": h,
                    "value": {"points": [[float(x) / w * 100, float(y) / h * 100] for x, y in poly],
                              "polygonlabels": ["доска"]}})
                n_boards += 1
            tasks.append({"data": {"image": f"/data/local-files/?d=frames/{m['file']}",
                                   "video": video, "second": m["second"], "kind": m["kind"]},
                          "predictions": [{"model_version": "черновик", "result": result}]})
            if j % 25 == 0:                 # картинка для проверки
                vis = img.copy()
                for r in result:
                    pts = np.array([[p[0] * w / 100, p[1] * h / 100] for p in r["value"]["points"]], np.int32)
                    cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
                cv2.imwrite(str(OUT / "preview" / m["file"]), cv2.resize(vis, None, fx=0.5, fy=0.5))
        print(f"{video}: кадров {len(items)}")

    (OUT / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    print(f"\nЗаданий: {len(tasks)}, черновых досок: {n_boards}")
    print(f"Сохранено: {OUT / 'tasks.json'}, примеры: {OUT / 'preview'}/")


if __name__ == "__main__":
    main()
