"""
Выбор кадров для разметки из часовых видео (без ручной перемотки).

Один проход по каждому видео (видео обрабатываются параллельно):
  1. Раз в STEP_S секунд сохраняем кадр-кандидат и уменьшенную полосу конвейера.
  2. После прохода: "пустой конвейер" = медиана всех полос (доски лежат меньшую часть времени).
     Для каждого кандидата считаем:
       присутствие — доля пикселей полосы, отличающихся от пустого конвейера;
       движение    — отличие от предыдущего кандидата.
  3. Выбираем ~ACTIVE кадров "с действием" (доски, люди) и ~IDLE "спокойных"
     (пустой конвейер, разный свет), равномерно по времени и не подряд.
  4. Выбранные копируем в data/frames/, остальных кандидатов удаляем.

Запуск (из папки machine_learning):
    .venv\\Scripts\\python.exe tools\\select_frames.py
"""

import json
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # папка проекта (machine_learning), откуда бы ни запускали

RAW = ROOT / "data" / "raw"
CANDIDATES = ROOT / "data" / "candidates"
FRAMES = ROOT / "data" / "frames"
STEP_S = 3                    # кандидат раз в 3 секунды
BAND = (600, 1050)            # строки с конвейером
SMALL = 4                     # уменьшение полосы для подсчётов
ACTIVE, IDLE = 70, 30         # сколько кадров выбрать на видео
MIN_GAP = 3                   # между выбранными кадрами хотя бы 3 кандидата (9 с)


def scan(path):
    out = CANDIDATES / path.stem
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = int(STEP_S * fps)
    bands, names, k = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k % step == 0:
            sec = int(k / fps)
            name = f"{path.stem}_{sec // 60:02d}m{sec % 60:02d}s.jpg"
            cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            g = cv2.cvtColor(frame[BAND[0]:BAND[1]], cv2.COLOR_BGR2GRAY)
            bands.append(cv2.resize(g, None, fx=1 / SMALL, fy=1 / SMALL, interpolation=cv2.INTER_AREA))
            names.append(name)
        k += 1
    cap.release()

    bands = np.array(bands, np.int16)
    empty = np.median(bands, axis=0)
    presence = (np.abs(bands - empty) > 30).mean(axis=(1, 2))
    motion = np.r_[0, (np.abs(np.diff(bands, axis=0)) > 30).mean(axis=(1, 2))]
    active = (presence > 0.04) | (motion > 0.02)

    def pick(indices, count):
        """Равномерно по времени, не ближе MIN_GAP друг к другу."""
        chosen = []
        for i in np.linspace(0, len(indices) - 1, min(count * 3, len(indices))).astype(int):
            idx = indices[i]
            if all(abs(idx - c) >= MIN_GAP for c in chosen):
                chosen.append(idx)
        if len(chosen) > count:
            chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, count).astype(int)]
        return chosen

    act_idx = list(np.nonzero(active)[0])
    idle_idx = list(np.nonzero(~active)[0])
    chosen_act = pick(act_idx, ACTIVE)
    chosen_idle = pick(idle_idx, IDLE + max(0, ACTIVE - len(chosen_act)))

    FRAMES.mkdir(parents=True, exist_ok=True)
    meta = []
    for idx, kind in [(i, "active") for i in chosen_act] + [(i, "idle") for i in chosen_idle]:
        shutil.copy(out / names[idx], FRAMES / names[idx])
        meta.append({"file": names[idx], "video": path.name, "second": int(idx) * STEP_S, "kind": kind,
                     "presence": round(float(presence[idx]), 3), "motion": round(float(motion[idx]), 3)})
    shutil.rmtree(out)
    return {"video": path.name, "candidates": len(names), "active_candidates": len(act_idx),
            "chosen_active": len(chosen_act), "chosen_idle": len(chosen_idle), "frames": meta}


def main():
    files = sorted(RAW.glob("*.mp4"))
    with Pool(len(files)) as pool:
        results = pool.map(scan, files)
    meta = []
    for r in results:
        print(f"{r['video']}: кандидатов {r['candidates']}, с действием {r['active_candidates']}, "
              f"выбрано: с действием {r['chosen_active']}, спокойных {r['chosen_idle']}")
        meta += r["frames"]
    (FRAMES / "frames.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Всего кадров для разметки: {len(meta)} -> {FRAMES}/")


if __name__ == "__main__":
    main()
