r"""
Обучение модели сегментации досок (дообучение готовой YOLO, не с нуля).

Вход — полоса конвейера 2688x700, модель работает с ней в размере IMGSZ по ширине
(rect=True: без квадратных полей, в 2-3 раза меньше вычислений).

Запуск (из любой папки; пути к датасету — от текущей папки или полные):
    на видеокарте NVIDIA (см. TRAIN_ON_GPU.md):
        .venv\Scripts\python.exe tools\train.py --data datasets/boards_v1/data.yaml --name boards_v2_s --device 0 --epochs 300 --batch 16
    на процессоре (медленно, только для пробы):
        .venv\Scripts\python.exe tools\train.py --data datasets/boards_v1/data.yaml --name test --device cpu --epochs 1

Результат: runs/segment/<name>/weights/best.pt и графики в runs/segment/<name>/
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent      # папка проекта (machine_learning), откуда бы ни запускали
RUNS = ROOT / "runs" / "segment"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml датасета")
    ap.add_argument("--name", required=True, help="имя запуска (папка в runs/segment/)")
    ap.add_argument("--model", default="yolo26s-seg.pt",
                    help="исходная модель; YOLO26 — у YOLO11 рамка не шире ~75%% входа, длинные доски обрезались")
    ap.add_argument("--device", default="cpu", help="cpu или 0 (первая видеокарта)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280, help="ширина входа модели")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    # Абсолютный путь: иначе Ultralytics сохраняет результаты в свою папку вне проекта.
    project = str(RUNS.resolve())
    # Датасет мог переехать на другой компьютер: подставляем в data.yaml его текущую папку.
    data_yaml = Path(args.data).resolve()
    lines = [f"path: {data_yaml.parent.as_posix()}" if l.startswith("path:") else l
             for l in data_yaml.read_text(encoding="utf-8").splitlines()]
    local_yaml = data_yaml.with_name("data_local.yaml")
    local_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.data = str(local_yaml)
    model = YOLO(args.model)                 # готовая модель, обученная на COCO
    model.train(
        data=args.data,
        name=args.name,
        project=project,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        rect=True,                           # полоса 2688x700 без квадратных полей
        patience=30,                         # остановка, если 30 эпох нет улучшения
        workers=2,
        # Аугментации: свет и цвет меняются в течение дня; зеркало по горизонтали допустимо.
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,
        degrees=0.0, translate=0.05, scale=0.2,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    metrics = model.val(data=args.data, imgsz=args.imgsz, rect=True, device=args.device, verbose=False,
                        project=project, name=args.name + "_val", exist_ok=True)
    print(f"\nКачество на проверочных кадрах: mAP50 маски = {metrics.seg.map50:.3f}, "
          f"mAP50-95 маски = {metrics.seg.map:.3f}")
    print(f"Модель: runs/segment/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
