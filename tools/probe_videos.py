"""
Проверка исходных видео: разрешение, FPS, настоящая длительность, целостность.
Читает каждый файл до конца (параллельно), сохраняет кадры в data/probe/.

Запуск (из папки machine_learning):
    .venv\\Scripts\\python.exe tools\\probe_videos.py
"""

import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent      # папка проекта (machine_learning), откуда бы ни запускали

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "probe"
SAVE_EVERY_S = 300          # сохранять кадр раз в 5 минут


def probe(path):
    started = time.time()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"file": path.name, "error": "не открылся"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = int(SAVE_EVERY_S * fps)
    n, failed_in_row = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            cv2.imwrite(str(OUT / f"{path.stem}_{int(n / fps // 60):03d}m.jpg"), frame)
        n += 1
    cap.release()
    return {"file": path.name, "size_mb": round(path.stat().st_size / 2**20),
            "width": w, "height": h, "fps": fps, "frames": n,
            "minutes": round(n / fps / 60, 1), "decode_s": round(time.time() - started)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("*.mp4"))
    if not files:
        print(f"Нет видео в {RAW}")
        return
    with Pool(len(files)) as pool:
        results = pool.map(probe, files)
    for r in results:
        print(r)
    (OUT / "probe.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
