"""
Подсчёт досок и коротких досок (выбросов) на видео конвейера.

Запуск (из папки machine_learning):
    .venv\\Scripts\\python.exe measure_boards.py data\\raw\\hiv00000.mp4
    .venv\\Scripts\\python.exe measure_boards.py data\\sample.mp4 --no-video
    .venv\\Scripts\\python.exe measure_boards.py видео.mp4 --start 60 --end 360

Результат — в папке results\\<имя видео>\\:
    boards.csv     — таблица досок (открывается в Excel)
    summary.json   — итог: сколько досок и выбросов
    annotated.mp4  — видео с разметкой (если не указан --no-video)

Как работает (подробнее — README.md):
  1. Модель ищет доски 5 раз в секунду (models/best.pt).
  2. Новая доска — длинный объект у входа в станок, который лежит неподвижно.
  3. Сдвиг доски D (см) считается в каждом кадре по рисунку древесины (tracker.py).
  4. Правый торец ловится, когда идёт через центр кадра; длина = торец + D - вход в станок
     (measure.py).
  5. Доска, которая так и не поехала в станок, не считается.
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detector import BAND_H, BAND_TOP, Detector, describe_device
from lens import Lens
from measure import END_CONF, END_ZONE, MIN_LINE_POINTS, AFTER_S, length_from_motion
from settings import HERE, load_settings
from tracker import DisplacementTracker, ProfileSampler

# --- как узнаём новую доску ---
NEW_MAX_X0 = 300           # левый конец доски у входа в станок (пикс)
LEFT_OK = (40, 200)        # левый торец в покое должен быть здесь (x, пикс): левее — уже в станке,
                           # правее — доска ещё не у входа; тогда длину не меряем
NEW_MIN_WIDTH = 1200       # доска 5 м и длиннее занимает больше 1200 пикс
NEW_MIN_CONF = 0.7
STILL_S = 1.5              # доска должна лежать неподвижно столько секунд
STILL_PX = 25              # «неподвижно» — правый край гуляет не больше чем на столько пикс
REST_MATCH_PX = 40         # объект в покое — та же доска, если правый край отличается меньше
NEXT_AFTER_S = 3.0         # новая доска не раньше, чем через 3 с после того, как прошлая поехала
NEXT_JUMP_PX = 200         # и правее места, где сейчас торец прошлой доски
FOLLOW_PX = 250            # слежение за доской: правый край сдвинулся не больше чем на столько
REST_BEFORE_S = 3.0        # точки торца берём не раньше чем за 3 с до конца покоя
MEASURE_EVERY_S = 1.0      # как часто пробовать посчитать длину
MOVED_CM = 30.0            # доска «поехала»: сдвиг по рисунку больше 30 см
MOVED_PX = 300             # или её правый край ушёл влево на 300 пикс

FONTS = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


@dataclass
class Board:
    start_t: float
    rest_x1: float                       # правый край в покое (пикс)
    last_rest_t: float
    last_right: float                    # где правый край сейчас (пикс)
    rest_y: float                        # высота осевой линии у правого торца в покое (пикс)
    rest_x0: float                       # левый торец в покое (пикс)
    number: int = 0                      # номер дают, когда доска поехала в станок
    end_points: list = field(default_factory=list)   # (t, s_торца)
    tracker: DisplacementTracker = field(default_factory=DisplacementTracker)
    result: dict = field(default_factory=dict)
    length_cm: float | None = None
    method: str = ""
    moved: bool = False                  # доска поехала в станок
    done: bool = False                   # длина посчитана, дальше не следим
    end_t: float = 0.0
    last_try_t: float = -1e9


class BoardCounter:
    def __init__(self, st, lens):
        self.st = st
        self.lens = lens
        self.p_left = float(lens.s(st.machine_entry_x, st.machine_entry_y))   # вход в станок, см
        self.sampler = ProfileSampler(lens, y_offset=BAND_TOP)
        self.boards = []                 # подтверждённые доски
        self.unfinished = []             # не успели доехать до конца видео
        self.dropped = 0                 # объекты, которые не поехали в станок
        self.cur = None
        self.cur_det = None
        self.still = []                  # (t, x1) кандидата в новую доску

    # --- ответы модели (каждый N-й кадр) ---
    def on_detections(self, t, dets):
        self._check_new_board(t, dets)
        cur = self.cur
        self.cur_det = None
        if cur is None:
            self.sampler.ready = None
            return
        near = [d for d in dets if d.conf >= 0.4 and d.x0 < 400 and d.width > 800
                and abs(d.x1 - cur.last_right) < FOLLOW_PX]
        if near:
            self.cur_det = min(near, key=lambda d: abs(d.x1 - cur.last_right))
            cur.last_right = self.cur_det.x1
            if cur.last_right < cur.rest_x1 - MOVED_PX:
                cur.moved = True
            self.sampler.prepare(self.cur_det)
        else:
            self.sampler.ready = None
        if cur.done:
            return
        for d in dets:
            if d.conf >= END_CONF and END_ZONE[0] < d.x1 < END_ZONE[1]:
                cur.end_points.append((t, float(self.lens.s(d.x1, d.y_at_right()))))
        if t - cur.last_try_t >= MEASURE_EVERY_S:
            cur.last_try_t = t
            self._try_measure(cur, t)

    def _check_new_board(self, t, dets):
        cands = [d for d in dets if d.x0 < NEW_MAX_X0 and d.width > NEW_MIN_WIDTH and d.conf > NEW_MIN_CONF]
        best = max(cands, key=lambda d: d.x1) if cands else None
        self.still.append((t, best.x1 if best else None, best.x0 if best else None))
        self.still = [p for p in self.still if p[0] > t - STILL_S - 0.01]
        xs = [p[1] for p in self.still]
        if None in xs or self.still[-1][0] - self.still[0][0] < STILL_S - 0.25 or max(xs) - min(xs) > STILL_PX:
            return
        x1 = float(np.median(xs))
        cur = self.cur
        if cur is not None and not cur.moved and abs(x1 - cur.rest_x1) < REST_MATCH_PX:
            cur.last_rest_t = t                      # прежняя доска всё ещё лежит
            return
        # у всех досок 10 м правый край в покое на одном месте, поэтому новую доску
        # отличаем так: прежняя уже поехала, и новая лежит правее её торца
        if cur is None or (cur.moved and t - cur.last_rest_t > NEXT_AFTER_S
                           and x1 > cur.last_right + NEXT_JUMP_PX):
            if cur is not None:
                self._finish(cur, t)
            self.cur = Board(start_t=self.still[0][0], rest_x1=x1, last_rest_t=t, last_right=x1,
                             rest_y=best.y_at_right(), rest_x0=float(np.median([p[2] for p in self.still])))

    # --- каждый кадр ---
    def on_frame(self, t, gray):
        cur = self.cur
        if cur is not None and not cur.done:
            cur.tracker.add(t, self.sampler.sample(gray))
            if cur.tracker.D > MOVED_CM:
                cur.moved = True

    # --- длина ---
    def _try_measure(self, b, t, final=False):
        if final:
            b.tracker.finish_rest()
        pts = [p for p in b.end_points if p[0] >= b.last_rest_t - REST_BEFORE_S]
        r = length_from_motion(pts, b.tracker.history, self.p_left)
        b.result = r
        if r["n_line"] >= MIN_LINE_POINTS and not b.number:
            b.number = len(self.boards) + 1          # поехала в станок — это доска
            self.boards.append(b)
        if not LEFT_OK[0] <= b.rest_x0 <= LEFT_OK[1]:
            if final or (r["line_start"] is not None and t >= r["line_start"] + AFTER_S):
                where = "уже в станке" if b.rest_x0 < LEFT_OK[0] else "не у входа в станок"
                b.method = f"левый торец в покое {where} (x={b.rest_x0:.0f})"
                b.done = True
            return
        if r["length_cm"] is not None and (final or t >= r["line_start"] + AFTER_S):
            b.length_cm = r["length_cm"] + self.st.length_correction_cm
            b.method = "движение: торец + сдвиг"
            b.done = True
        elif final and b.number and END_ZONE[0] < b.rest_x1 < END_ZONE[1]:
            # короткая доска: правый торец был виден, пока она лежала
            b.length_cm = float(self.lens.s(b.rest_x1, b.rest_y)) - self.p_left + self.st.length_correction_cm
            b.method = "в покое: торец виден"
            b.done = True

    def _finish(self, b, t, end_of_video=False):
        if not b.done:
            self._try_measure(b, t, final=True)
        if b.number and b.length_cm is None and not b.method:
            b.method = ("видео закончилось раньше, чем торец прошёл центр кадра" if end_of_video
                        else "мало данных: торец и сдвиг доски не удалось связать")
        b.end_t = t
        if not b.number:
            if end_of_video:
                self.unfinished.append(b)
            else:
                self.dropped += 1
                print(f"  {fmt_t(b.start_t)}: объект не поехал в станок — не считается")

    def close(self, t):
        if self.cur is not None:
            self._finish(self.cur, t, end_of_video=True)
            self.cur = None

    # --- итог ---
    def status(self, b):
        if b.length_cm is None:
            return "не измерена"
        return "выброс" if b.length_cm <= self.st.outlier_limit_cm else "норма"

    def counts(self):
        st = [self.status(b) for b in self.boards]
        return dict(boards=len(self.boards), outliers=st.count("выброс"), not_measured=st.count("не измерена"))


def fmt_t(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}" if t >= 3600 else f"{int(t // 60)}:{t % 60:04.1f}"


class Overlay:
    """Разметка на видео: контуры досок, номер и длина, счётчики."""

    def __init__(self, scale):
        self.scale = scale
        fonts = list(FONTS)
        try:                       # matplotlib (ставится вместе с ultralytics) несёт шрифт с кириллицей
            import matplotlib
            fonts.append(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"))
        except ImportError:
            pass
        path = next((p for p in fonts if Path(p).exists()), None)
        self.font = ImageFont.truetype(path, 22) if path else ImageFont.load_default()
        self.small = ImageFont.truetype(path, 17) if path else self.font

    def draw(self, frame, t, dets, counter):
        s = self.scale
        img = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cur, cur_det, limit = counter.cur, counter.cur_det, counter.st.outlier_limit_cm
        for d in dets or []:
            if d is cur_det:
                continue
            color = (0, 200, 0) if d.conf >= END_CONF else (150, 150, 150)
            cv2.polylines(img, [(d.contour * s).astype(np.int32)], True, color, 1)
        label = None
        if cur is not None and cur_det is not None:
            color = (0, 220, 255)                                     # жёлтая — измеряется
            if cur.length_cm is not None:
                color = (0, 0, 255) if cur.length_cm <= limit else (0, 220, 0)
            cv2.polylines(img, [(cur_det.contour * s).astype(np.int32)], True, color, 3)
            if not cur.number:
                text = "Новая доска: ждём, поедет ли в станок"
            elif cur.length_cm is None:
                text = f"Доска №{cur.number}: измеряется..."
            else:
                text = f"Доска №{cur.number}: {cur.length_cm:.0f} см ({counter.status(cur)})"
            ys = cur_det.contour[:, 0, 1]
            label = (int(cur_det.x0 * s) + 20, int(ys.min() * s) - 34, text)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil, "RGBA")
        c = counter.counts()
        lines = [f"Время видео: {fmt_t(t)}",
                 f"Досок: {c['boards']}   Выбросов: {c['outliers']}   Не измерено: {c['not_measured']}",
                 f"N = {counter.st.normal_length_cm:.0f} см, выброс если ≤ {limit:.0f} см"]
        last = [b for b in counter.boards if b.length_cm is not None][-3:]
        if last:
            lines.append("Последние: " + ", ".join(f"№{b.number} {b.length_cm:.0f}" for b in last))
        dr.rectangle((8, 8, 640, 16 + 30 * len(lines)), fill=(0, 0, 0, 160))
        for i, line in enumerate(lines):
            dr.text((16, 14 + 30 * i), line, font=self.font, fill=(255, 255, 255))
        if label:
            x, y, text = label
            y = max(y, 16 + 30 * len(lines) + 4)
            dr.rectangle((x - 4, y - 2, x + 12 * len(text), y + 26), fill=(0, 0, 0, 160))
            dr.text((x, y), text, font=self.font, fill=(255, 255, 0))
        dr.text((16, pil.height - 28), "Жёлтый контур — доска измеряется, зелёный — норма, красный — выброс, "
                "тонкий — другие ответы модели", font=self.small, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def analyze_video(video, out_dir, write_video=True, start_s=0.0, end_s=None, scale=0.5, device=None):
    """Обработать одно видео. Возвращает итог: {'boards': ..., 'outliers': ..., ...}.
    device: None — как в config.toml; 'auto', 'cpu' или '0' (видеокарта)."""
    st = load_settings()
    if st.model_url and not st.model_path.exists():
        from download_model import download
        download(st.model_url, st.model_path, st.model_sha256)
    device = device or st.device
    problems = st.problems()
    if problems:
        raise SystemExit("ОШИБКА в config.toml:\n  " + "\n  ".join(problems))
    lens = Lens.load(st.calibration_path, st.marker_length_cm)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"ОШИБКА: не открылось видео {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 5 < fps < 121:
        fps = 25.0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Нейросеть считается на: {describe_device(device)}")
    detector = Detector(st.model_path, device)
    counter = BoardCounter(st, lens)
    overlay = Overlay(scale) if write_video else None
    writer = None
    dets = []

    k = 0
    while k < int(start_s * fps) and cap.grab():          # пропуск начала без декодирования в картинку
        k += 1
    t_wall = time.time()
    k_first = k
    t = k / fps
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = k / fps
        if end_s is not None and t > end_s:
            break
        if (k - k_first) % st.every_n_frames == 0:
            dets = detector(frame)
            counter.on_detections(t, dets)
        counter.on_frame(t, cv2.cvtColor(frame[BAND_TOP:BAND_TOP + BAND_H], cv2.COLOR_BGR2GRAY))
        if write_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (int(w * scale), int(h * scale)))
            writer.write(overlay.draw(frame, t, dets, counter))
        if (k - k_first) % int(fps * 60) == 0 and k > k_first:
            speed = (k - k_first) / (time.time() - t_wall)
            c = counter.counts()
            print(f"  {fmt_t(t)}  досок: {c['boards']}, выбросов: {c['outliers']}  "
                  f"({speed:.1f} кадров/с, x{speed / fps:.2f} от реального времени)", flush=True)
        k += 1
    counter.close(t)
    cap.release()
    if writer is not None:
        writer.release()

    with open(out_dir / "boards.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["номер", "появилась", "появилась_с", "длина_см", "статус", "как_измерена",
                     "точек_торца", "кадров_торец_и_сдвиг", "скорость_см_с"])
        for b in counter.boards + counter.unfinished:
            r = b.result or {}
            wr.writerow([b.number or "", fmt_t(b.start_t), f"{b.start_t:.1f}",
                         "" if b.length_cm is None else f"{b.length_cm:.0f}",
                         counter.status(b) if b.number else "не завершена (конец видео)",
                         b.method, r.get("n_line", 0), r.get("n_overlap", 0),
                         "" if r.get("speed") is None else f"{-r['speed']:.1f}"])
    summary = dict(video=str(video), **counter.counts(), unfinished=len(counter.unfinished),
                   dropped_objects=counter.dropped, normal_length_cm=st.normal_length_cm,
                   outlier_limit_cm=st.outlier_limit_cm,
                   lengths_cm=[None if b.length_cm is None else round(b.length_cm, 1) for b in counter.boards],
                   processing_s=round(time.time() - t_wall, 1), video_s=round(t - start_s, 1))
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Подсчёт досок и выбросов на видео конвейера")
    ap.add_argument("video", help="путь к видеофайлу")
    ap.add_argument("--out", help="папка для результатов (по умолчанию results/<имя видео>)")
    ap.add_argument("--no-video", action="store_true", help="не записывать видео с разметкой (быстрее)")
    ap.add_argument("--start", type=float, default=0.0, help="начать с этой секунды")
    ap.add_argument("--end", type=float, default=None, help="закончить на этой секунде")
    ap.add_argument("--device", default=None,
                    help="auto, cpu или 0 (видеокарта NVIDIA); по умолчанию — как в config.toml")
    args = ap.parse_args()

    video = Path(args.video)
    out = Path(args.out) if args.out else HERE / "results" / video.stem
    print(f"Видео: {video}")
    print(f"Результаты: {out}")
    s = analyze_video(video, out, write_video=not args.no_video, start_s=args.start, end_s=args.end,
                      device=args.device)
    print()
    print(f"Готово за {s['processing_s'] / 60:.1f} мин (видео {s['video_s'] / 60:.1f} мин).")
    print(f"Досок: {s['boards']}, выбросов (≤ {s['outlier_limit_cm']:.0f} см): {s['outliers']}, "
          f"не измерено: {s['not_measured']}")
    if s["unfinished"]:
        print(f"Не доехали до конца видео (не считаются): {s['unfinished']}")
    print(f"Таблица: {out / 'boards.csv'}")
    if not args.no_video:
        print(f"Видео:   {out / 'annotated.mp4'}")


if __name__ == "__main__":
    main()
