"""
Перевод разметки Label Studio в датасет YOLO (сегментация).

Модель видит только полосу конвейера (строки BAND кадра), а не весь кадр:
в 2 раза меньше вычислений, доски в уменьшенном изображении толще.
Контуры переносятся в координаты полосы (часть контура вне полосы обрезается).

Деление на обучение и проверку — ПО ВИДЕО (кадры одного видео целиком в одной части),
чтобы проверка была на видео, которое модель не видела.

Запуск (из папки machine_learning):
    .venv\\Scripts\\python.exe tools\\build_dataset.py --exports labeling/export_round1.json --val hiv00003 --name boards_r1
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # папка проекта (machine_learning), откуда бы ни запускали

FRAMES = ROOT / "data" / "frames"
BAND = (450, 1150)          # строки кадра с конвейером


def board_polygons(task):
    """Контуры досок задания в пикселях полного кадра (последняя аннотация)."""
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None                         # задание не размечено
    out = []
    for r in anns[-1]["result"]:
        if r["type"] != "polygonlabels":
            continue
        w, h = r["original_width"], r["original_height"]
        out.append(np.array([[p[0] * w / 100, p[1] * h / 100] for p in r["value"]["points"]], np.float32))
    return out


def to_band(poly, frame_w):
    """Контур полного кадра -> контур в полосе (через маску, чтобы корректно обрезать)."""
    mask = np.zeros((BAND[1] - BAND[0], frame_w), np.uint8)
    cv2.fillPoly(mask, [np.round(poly - (0, BAND[0])).astype(np.int32)], 1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    return cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", nargs="+", required=True, help="файлы экспорта Label Studio (JSON)")
    ap.add_argument("--val", nargs="+", required=True, help="видео для проверки, например hiv00003")
    ap.add_argument("--name", required=True, help="имя датасета (папка в datasets/)")
    args = ap.parse_args()

    root = ROOT / "datasets" / args.name
    if root.exists():
        shutil.rmtree(root)
    for part in ("train", "val"):
        (root / "images" / part).mkdir(parents=True)
        (root / "labels" / part).mkdir(parents=True)

    tasks = {}
    for e in args.exports:                       # если кадр есть в нескольких файлах — берём последний
        for t in json.loads(Path(e).read_text(encoding="utf-8")):
            tasks[t["data"]["image"].split("frames/")[-1]] = t

    stats = {"train": [0, 0], "val": [0, 0]}
    for name, task in sorted(tasks.items()):
        polys = board_polygons(task)
        if polys is None:
            continue
        part = "val" if any(name.startswith(v) for v in args.val) else "train"
        img = cv2.imread(str(FRAMES / name))
        h, w = img.shape[:2]
        band = img[BAND[0]:BAND[1]]
        cv2.imwrite(str(root / "images" / part / name), band, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for p in polys:
            bp = to_band(p, w)
            if bp is None:
                continue
            bh = BAND[1] - BAND[0]
            coords = " ".join(f"{x / w:.6f} {y / bh:.6f}" for x, y in bp)
            lines.append(f"0 {coords}")
        (root / "labels" / part / (Path(name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        stats[part][0] += 1
        stats[part][1] += len(lines)

    (root / "data.yaml").write_text(
        f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: board\n",
        encoding="utf-8")
    for part, (n_img, n_obj) in stats.items():
        print(f"{part}: кадров {n_img}, досок {n_obj}")
    print(f"Датасет: {root}/  (настройки: {root / 'data.yaml'})")


if __name__ == "__main__":
    main()
