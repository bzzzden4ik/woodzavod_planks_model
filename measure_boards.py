r"""
Подсчёт досок и коротких досок (выбросов) на видео конвейера.

Запуск (из папки machine_learning):
    .venv\Scripts\python.exe measure_boards.py data\raw\hiv00000.mp4
    .venv\Scripts\python.exe measure_boards.py data\sample.mp4 --no-video
    .venv\Scripts\python.exe measure_boards.py видео.mp4 --start 60 --end 360

Результат — в папке results\<имя видео>\:
    boards.csv     — таблица досок (открывается в Excel)
    summary.json   — итог: сколько досок и выбросов
    annotated.mp4  — видео с разметкой (если не указан --no-video)

Как работает (подробнее — README.md):
  1. Модель ищет доски 5 раз в секунду (models/best.pt).
  2. Слежение: уверенный объект, правый конец которого двигается только влево (в станок).
     Новая доска — длинный объект у входа в станок правее всех, за которыми уже следим.
     Доска засчитывается, когда проехала в станок.
  3. Длина — по положению доски в начале (когда её положили):
     - видны оба торца — длина прямо по кадру;
     - левый торец уже в станке (за краем кадра) — длина известна в пределах
       [видимая часть; видимая часть + max_hidden_left_cm];
     - правый торец за правым краем кадра (длинная доска) — правый торец ловится, когда
       идёт через центр кадра, а сдвиг доски D считается по рисунку древесины (tracker.py).
  4. Выброс — если длина (или её верхняя граница) <= N * (1 - погрешность). Если по
     границам решить нельзя — «не измерена».
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
from measure import END_CONF, END_ZONE, length_from_motion
from settings import HERE, load_settings
from tracker import DisplacementTracker, ProfileSampler

# --- слежение за досками ---
TRACK_CONF = 0.65          # слабее — не используем (бывают «призраки» доски от видеокодека)
TRACK_MIN_WIDTH = 300      # уже — обрезки, куски доски, ушедшей в станок
MATCH_TOL_PX = 80          # правый край той же доски может «дрогнуть» на столько вправо
MAX_SPEED_PX = 450         # и уехать влево не быстрее, пикс/с
LOST_AFTER_S = 8.0         # доску не видно дольше — она ушла (у длинной доски «провал» бывает до 6 с)
# --- новая доска ---
NEW_MIN_WIDTH = 800        # новая доска на кадре длиннее 800 пикс (доска 5 м — ~1500)
NEW_MAX_X0 = 300           # и левым концом у входа в станок или уже в нём
NEW_RIGHT_OF_PX = 150      # и правее всех досок, за которыми следим (её кладут после)
# --- когда засчитываем ---
COUNT_TRAVEL_PX = 400      # правый край уехал влево хотя бы на столько
COUNT_MIN_DETS = 5         # и доску видели хотя бы в 5 ответах модели (~1 с)
# --- начало (где доску положили) ---
START_S = 0.6              # первые 0.6 с слежения
START_MOVE_PX = 40         # и пока правый край сдвинулся не больше чем на столько
EDGE_X0 = 20               # левый край контура у края кадра — доска уходит за кадр
LEFT_SAME = 0.85           # левее контура яркость как у доски — доска продолжается (в станок)
STILL_S = 1.0              # длинная доска: сдвиг D считаем от момента, когда она пролежала
STILL_PX = 25              # неподвижно 1 с (правый край гуляет не больше чем на столько пикс)
FOLLOW_CONF = 0.4          # для рисунка доски годится и неуверенный контур
FOLLOW_PX = 250            # если его правый край рядом с прошлым (пикс)
MEASURE_EVERY_S = 1.0      # как часто пересчитывать длину для показа на видео

FONTS = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def left_end_hidden(gray, det):
    """Левый торец не виден: контур у края кадра, или левее контура доска продолжается
    (модель не размечает доску за концом роликов). gray — полоса кадра с BAND_TOP."""
    if det.x0 < EDGE_X0:
        return True
    ok = ~np.isnan(det.yc[:80]) & (det.thick[:80] > 10)
    if ok.sum() < 10 or det.x0 < 60:
        return False
    y = int(np.median(det.yc[:80][ok])) - BAND_TOP
    h = max(int(np.median(det.thick[:80][ok]) / 4), 3)
    board = np.median(gray[y - h:y + h, det.x0 + 30:det.x0 + 300])
    left = np.median(gray[y - h:y + h, max(det.x0 - 50, 0):det.x0])
    return bool(left >= LEFT_SAME * board)


@dataclass
class Board:
    first_t: float
    first_x1: float
    last_t: float
    last_x1: float
    needs_shift: bool                    # правый торец за кадром — нужен сдвиг D
    number: int = 0                      # номер дают, когда доска поехала в станок
    max_x1: float = 0.0
    min_x1: float = 1e9
    n_dets: int = 0
    start: list = field(default_factory=list)        # (x0, x1, y_левого, y_правого, левый скрыт)
    end_points: list = field(default_factory=list)   # (t, s_торца) — для прямой
    tracker: DisplacementTracker = field(default_factory=DisplacementTracker)
    sampler: ProfileSampler = None
    last_det: object = None
    result: dict = field(default_factory=dict)
    length_cm: float | None = None       # оценка длины
    length_min: float | None = None      # длина точно не меньше
    length_max: float | None = None      # и не больше
    method: str = ""
    shift_started: bool = False          # доска легла неподвижно — сдвиг D считаем отсюда
    shift_done: bool = False             # сдвиг больше не нужен
    recent: list = field(default_factory=list)       # (t, x1) последних ответов модели
    follow_x1: float = 0.0               # правый край по любому ответу модели (для рисунка доски)
    closed: bool = False
    last_try_t: float = -1e9


class BoardCounter:
    def __init__(self, st, lens):
        self.st = st
        self.lens = lens
        self.p_left = float(lens.s(st.machine_entry_x, st.machine_entry_y))   # вход в станок, см
        self.all = []                    # все объекты, за которыми следили
        self.boards = []                 # засчитанные доски (по порядку номеров)
        self.unfinished = []             # видео кончилось раньше, чем доска поехала
        self.dropped = 0                 # объекты, которые не поехали в станок

    def active(self, t):
        return [b for b in self.all if not b.closed]

    # --- ответы модели (каждый N-й кадр) ---
    def on_detections(self, t, dets, gray):
        for b in self.active(t):
            b.last_det = None
        cands = sorted([d for d in dets if d.conf >= TRACK_CONF and d.width >= TRACK_MIN_WIDTH],
                       key=lambda d: -d.x1)
        used = set()
        for d in cands:
            best = None
            for b in self.active(t):
                if id(b) in used:
                    continue
                lo = b.last_x1 - (t - b.last_t) * MAX_SPEED_PX - MATCH_TOL_PX
                if lo <= d.x1 <= b.last_x1 + MATCH_TOL_PX and (
                        best is None or abs(d.x1 - b.last_x1) < abs(d.x1 - best.last_x1)):
                    best = b
            if best is None:
                right_of = max([b.last_x1 for b in self.active(t)], default=-1e9)
                if d.width >= NEW_MIN_WIDTH and d.x0 < NEW_MAX_X0 and d.x1 > right_of + NEW_RIGHT_OF_PX:
                    best = Board(first_t=t, first_x1=d.x1, last_t=t, last_x1=d.x1,
                                 needs_shift=d.x1 >= END_ZONE[1],
                                 sampler=ProfileSampler(self.lens, y_offset=BAND_TOP))
                    self.all.append(best)
                else:
                    continue
            used.add(id(best))
            self._update(best, t, d, gray)
        for b in self.active(t):
            if b.needs_shift and b.shift_started and not b.shift_done:
                self._follow(b, dets)
            if t - b.last_t > LOST_AFTER_S:
                self._close(b, t)

    def _follow(self, b, dets):
        """Контур для рисунка доски: любой ответ модели рядом с ней, даже неуверенный.
        Уверенный контур на длинной доске пропадает на секунды, а доска за это время уезжает."""
        near = [d for d in dets if d.conf >= FOLLOW_CONF and d.x0 < 400 and d.width > 800
                and abs(d.x1 - b.follow_x1) < FOLLOW_PX]
        if near:
            d = min(near, key=lambda d: abs(d.x1 - b.follow_x1))
            b.follow_x1 = d.x1
            b.sampler.prepare(d)
        else:
            b.sampler.ready = None

    def _update(self, b, t, d, gray):
        b.last_t, b.last_x1, b.last_det = t, d.x1, d
        b.max_x1, b.min_x1 = max(b.max_x1, d.x1), min(b.min_x1, d.x1)
        b.n_dets += 1
        if b.needs_shift and not b.shift_started:
            # длинная доска: начало — когда она легла неподвижно (раньше её ещё двигают руками)
            b.recent = [(tt, x) for tt, x in b.recent if tt > t - STILL_S] + [(t, d.x1)]
            xs = [x for _, x in b.recent]
            if t - b.recent[0][0] >= STILL_S - 0.25 and max(xs) - min(xs) <= STILL_PX:
                b.shift_started = True
                b.first_t, b.first_x1, b.start = t, d.x1, []
                b.follow_x1 = d.x1
        if (b.shift_started or not b.needs_shift) and t <= b.first_t + START_S                 and d.x1 >= b.first_x1 - START_MOVE_PX:
            b.start.append((d.x0, d.x1, float(np.nanmedian(d.yc[:15])), d.y_at_right(),
                            left_end_hidden(gray, d)))
        if END_ZONE[0] < d.x1 < END_ZONE[1]:
            b.end_points.append((t, float(self.lens.s(d.x1, d.y_at_right()))))
        if not b.number and b.max_x1 - b.min_x1 >= COUNT_TRAVEL_PX and b.n_dets >= COUNT_MIN_DETS:
            b.number = len(self.boards) + 1          # поехала в станок — это доска
            self.boards.append(b)
        if b.number and t - b.last_try_t >= MEASURE_EVERY_S:
            b.last_try_t = t
            self._measure(b)

    # --- каждый кадр: сдвиг длинных досок ---
    def on_frame(self, t, gray):
        for b in self.active(t):
            if b.needs_shift and b.shift_started and not b.shift_done:
                b.tracker.add(t, b.sampler.sample(gray))

    # --- длина ---
    def _measure(self, b, final=False):
        if not b.start:
            b.method = "нет уверенных кадров в начале"
            return
        st = np.array(b.start, float)
        x0, x1, yl, yr = (float(np.median(st[:, i])) for i in range(4))
        hidden = st[:, 4].mean() >= 0.5
        s_edge = float(self.lens.s(0, yl))                      # левый край кадра, см
        H = self.st.max_hidden_left_cm
        corr = self.st.length_correction_cm
        if not b.needs_shift:                                   # правый торец виден сразу
            p_right = float(self.lens.s(x1, yr))
            if not hidden:
                L = p_right - float(self.lens.s(x0, yl)) + corr
                b.length_cm = b.length_min = b.length_max = L
                b.method = "оба торца в кадре"
            else:
                lo = p_right - s_edge + corr
                b.length_min, b.length_max, b.length_cm = lo, lo + H, lo
                b.method = "левый торец в станке: длина в пределах"
        else:                                                   # длинная: торец + сдвиг
            if final:
                b.tracker.finish_rest()
            r = length_from_motion(b.end_points, b.tracker.history, s_edge if hidden else self.p_left)
            b.result = r
            if r["length_cm"] is None:
                if final:
                    b.method = "мало данных: торец и сдвиг доски не удалось связать"
                return
            L = r["length_cm"] + corr
            if not hidden:
                b.length_cm = b.length_min = b.length_max = L
                b.method = "движение: торец + сдвиг"
            else:
                b.length_min, b.length_max, b.length_cm = L, L + H, L
                b.method = "движение, левый торец в станке: длина в пределах"
            if b.end_points and b.end_points[-1][0] > r["line_start"] + 1.0:
                b.shift_done = True                             # дальше сдвиг не нужен

    def _close(self, b, t, end_of_video=False):
        b.closed = True
        if b.number:
            self._measure(b, final=True)
        elif end_of_video and b.n_dets >= COUNT_MIN_DETS:
            self.unfinished.append(b)
        else:
            self.dropped += 1

    def close(self, t):
        for b in self.active(t):
            self._close(b, t, end_of_video=True)

    # --- итог ---
    def status(self, b):
        limit = self.st.outlier_limit_cm
        if b.length_max is not None and b.length_max <= limit:
            return "выброс"
        if b.length_min is not None and b.length_min > limit:
            return "норма"
        return "не измерена"

    def counts(self):
        st = [self.status(b) for b in self.boards]
        return dict(boards=len(self.boards), outliers=st.count("выброс"), not_measured=st.count("не измерена"))


def fmt_t(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}" if t >= 3600 else f"{int(t // 60)}:{t % 60:04.1f}"


def fmt_len(b):
    if b.length_min is None:
        return "?"
    if b.length_max - b.length_min < 1:
        return f"{b.length_cm:.0f}"
    return f"{b.length_min:.0f}–{b.length_max:.0f}"


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
        tracked = {id(b.last_det): b for b in counter.active(t) if b.last_det is not None}
        labels = []
        for d in dets or []:
            b = tracked.get(id(d))
            if b is None:
                cv2.polylines(img, [(d.contour * s).astype(np.int32)], True, (150, 150, 150), 1)
                continue
            st = counter.status(b) if b.number else ""
            color = {"выброс": (0, 0, 255), "норма": (0, 220, 0)}.get(st, (0, 220, 255))
            cv2.polylines(img, [(d.contour * s).astype(np.int32)], True, color, 3)
            if not b.number:
                text = "Новая доска: засчитаю, когда поедет в станок"
            elif b.length_min is None:
                text = f"№{b.number}: измеряется..."
            else:
                text = f"№{b.number}: {fmt_len(b)} см ({st})"
            labels.append((int(d.x0 * s) + 20, int(d.contour[:, 0, 1].min() * s) - 34, text))

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil, "RGBA")
        c = counter.counts()
        limit = counter.st.outlier_limit_cm
        lines = [f"Время видео: {fmt_t(t)}",
                 f"Досок: {c['boards']}   Выбросов: {c['outliers']}   Не измерено: {c['not_measured']}",
                 f"N = {counter.st.normal_length_cm:.0f} см, выброс если ≤ {limit:.0f} см"]
        last = [b for b in counter.boards if b.length_min is not None][-3:]
        if last:
            lines.append("Последние: " + ", ".join(f"№{b.number} {fmt_len(b)}" for b in last))
        dr.rectangle((8, 8, 640, 16 + 30 * len(lines)), fill=(0, 0, 0, 160))
        for i, line in enumerate(lines):
            dr.text((16, 14 + 30 * i), line, font=self.font, fill=(255, 255, 255))
        for x, y, text in labels:
            y = max(y, 16 + 30 * len(lines) + 4)
            dr.rectangle((x - 4, y - 2, x + 12 * len(text), y + 26), fill=(0, 0, 0, 160))
            dr.text((x, y), text, font=self.font, fill=(255, 255, 0))
        dr.text((16, pil.height - 28), "Жёлтый контур — доска измеряется, зелёный — норма, красный — выброс, "
                "серый — не доска или неуверенный ответ модели", font=self.small, fill=(255, 255, 255))
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
        gray = cv2.cvtColor(frame[BAND_TOP:BAND_TOP + BAND_H], cv2.COLOR_BGR2GRAY)
        if (k - k_first) % st.every_n_frames == 0:
            dets = detector(frame)
            counter.on_detections(t, dets, gray)
        counter.on_frame(t, gray)
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
        wr.writerow(["номер", "появилась", "появилась_с", "длина_см", "длина_мин_см", "длина_макс_см",
                     "статус", "как_измерена", "точек_торца", "кадров_торец_и_сдвиг", "скорость_см_с"])
        r0 = lambda v: "" if v is None else f"{v:.0f}"
        for b in counter.boards + counter.unfinished:
            r = b.result or {}
            wr.writerow([b.number or "", fmt_t(b.first_t), f"{b.first_t:.1f}",
                         r0(b.length_cm), r0(b.length_min), r0(b.length_max),
                         counter.status(b) if b.number else "не завершена (конец видео)",
                         b.method, r.get("n_line", len(b.end_points)), r.get("n_overlap", ""),
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
        print(f"Не поехали в станок до конца видео (не считаются): {s['unfinished']}")
    print(f"Таблица: {out / 'boards.csv'}")
    if not args.no_video:
        print(f"Видео:   {out / 'annotated.mp4'}")


if __name__ == "__main__":
    main()
