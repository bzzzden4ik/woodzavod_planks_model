"""
Делит кадры на два круга разметки.

  Круг 1 (labeling/tasks_round1.json): ROUND1_PER_VIDEO кадров из каждого видео,
      равномерно по времени — размечаются вручную, на них учится черновая модель.
  Круг 2 (labeling/tasks_round2.json): остальные кадры — их сначала размечает
      черновая модель, человек только проверяет.

Берёт labeling/tasks.json (результат prelabel.py) — черновые контуры сохраняются.

Запуск (из папки machine_learning):
    .venv\\Scripts\\python.exe tools\\make_rounds.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # папка проекта (machine_learning), откуда бы ни запускали

LABELING = ROOT / "labeling"
ROUND1_PER_VIDEO = 20


def main():
    tasks = json.loads((LABELING / "tasks.json").read_text(encoding="utf-8"))
    by_video = defaultdict(list)
    for t in tasks:
        by_video[t["data"]["video"]].append(t)
    round1, round2 = [], []
    for video, items in sorted(by_video.items()):
        items.sort(key=lambda t: t["data"]["second"])
        pick = set(np.linspace(0, len(items) - 1, ROUND1_PER_VIDEO).astype(int).tolist())
        for i, t in enumerate(items):
            (round1 if i in pick else round2).append(t)
    # Перемешиваем порядок видео в первом круге, чтобы условия чередовались.
    round1.sort(key=lambda t: (t["data"]["second"], t["data"]["video"]))
    (LABELING / "tasks_round1.json").write_text(json.dumps(round1, ensure_ascii=False), encoding="utf-8")
    (LABELING / "tasks_round2.json").write_text(json.dumps(round2, ensure_ascii=False), encoding="utf-8")
    print(f"Круг 1: {len(round1)} кадров -> labeling/tasks_round1.json")
    print(f"Круг 2: {len(round2)} кадров -> labeling/tasks_round2.json")


if __name__ == "__main__":
    main()
