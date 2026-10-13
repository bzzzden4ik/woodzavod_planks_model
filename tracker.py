"""
Сдвиг доски D: насколько сантиметров доска уехала в станок с момента, когда лежала.

Как меряется:
  1. Вдоль осевой линии доски (её даёт модель) берётся яркость — «рисунок древесины»
     (сучки, волокна), пересчитанная в сантиметры по модели объектива.
  2. Пока доска лежит, запоминается её рисунок (опорный).
  3. В каждом следующем кадре рисунок в центре кадра сравнивается с опорным:
     на сколько сантиметров он сдвинулся — это и есть D.
  4. Опорный рисунок обновляется каждые несколько сантиметров (по цепочке),
     потому что на движущейся доске картинка смазывается видеокодеком.
Сравнение идёт только в центре кадра: там «рыбий глаз» почти не искажает.
"""

import warnings

import numpy as np

STEP_CM = 0.5                         # шаг рисунка вдоль доски, см
GRID = np.arange(-170, 320, STEP_CM)  # где берём рисунок (см от центра объектива)
COMPARE = (-150, 250)                 # где сравниваем рисунок (центр кадра), см
HP_WINDOW = 40                        # убрать плавные перепады яркости длиннее 20 см
MIN_OVERLAP = 300                     # минимум общих точек рисунка (150 см)

REST_FRAMES = 50                      # кадров покоя для опорного рисунка (2 с)
MIN_SCORE = 0.3                       # совпадение рисунка хуже — кадр пропускаем
RELINK_CM = 6.0                       # после такого сдвига опора переносится на текущий кадр


def highpass(p):
    """Убрать плавные изменения яркости (тени, свет), оставить рисунок. NaN — нет доски."""
    v = ~np.isnan(p)
    q = np.where(v, p, 0.0)
    k = np.ones(HP_WINDOW) / HP_WINDOW
    mq = np.convolve(q, k, "same")
    mc = np.convolve(v.astype(float), k, "same")
    return np.where(v & (mc > 0.5), p - mq / np.maximum(mc, 1e-6), np.nan)


def find_shift(ref, cur, lo_cm, hi_cm):
    """Сдвиг D (см, > 0 — доска уехала влево), при котором cur(u) похоже на ref(u + D).
    Перебор от lo_cm до hi_cm. Возвращает (D, совпадение от -1 до 1) или (None, -1)."""
    res = []
    for n in range(int(np.floor(lo_cm / STEP_CM)), int(np.ceil(hi_cm / STEP_CM)) + 1):
        a, b = (ref[n:], cur[:len(cur) - n]) if n >= 0 else (ref[:n], cur[-n:])
        v = ~np.isnan(a) & ~np.isnan(b)
        if v.sum() < MIN_OVERLAP:
            res.append(-2.0)
            continue
        x = a[v] - a[v].mean()
        y = b[v] - b[v].mean()
        res.append(float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9)))
    res = np.array(res)
    i = int(np.argmax(res))
    if res[i] <= -1:
        return None, -1.0
    f = float(i)
    if 0 < i < len(res) - 1 and res[i - 1] > -1 and res[i + 1] > -1:   # уточнение между отсчётами
        l, c, r = res[i - 1], res[i], res[i + 1]
        den = l - 2 * c + r
        if den != 0:
            f = i + 0.5 * (l - r) / den
    return np.floor(lo_cm / STEP_CM) * STEP_CM + f * STEP_CM, float(res[i])


class ProfileSampler:
    """Берёт рисунок доски из кадра по осевой линии, которую нашла модель."""

    def __init__(self, lens, y_offset=0):
        self.lens = lens
        self.y_offset = y_offset          # с какой строки кадра начинается переданная картинка
        self.ready = None

    def prepare(self, det):
        """Запомнить, откуда брать рисунок (по последнему ответу модели)."""
        ok = (det.thick >= 12) & ~np.isnan(det.yc) & (det.xs < det.x1 - 10)
        if ok.sum() < 100:
            self.ready = None
            return
        xs, yc = det.xs[ok], det.yc[ok]
        s = self.lens.s(xs, yc)
        keep = (s >= GRID[0] - 5) & (s <= GRID[-1] + 5)
        if keep.sum() < 100:
            self.ready = None
            return
        xs, yc, s = xs[keep], yc[keep], s[keep]
        order = np.argsort(s)
        xs, yc, s = xs[order], yc[order], s[order]
        half = max(int(np.median(det.thick[ok]) / 6), 2)         # средняя треть толщины доски
        rows = np.round(yc).astype(int)[None, :] + np.arange(-half, half + 1)[:, None] - self.y_offset
        cols = np.broadcast_to(xs, rows.shape)
        inside = (GRID >= s[0]) & (GRID <= s[-1])
        self.ready = (rows, cols, s, inside)

    def sample(self, gray):
        """gray — полутоновая картинка (кадр или полоса кадра, начиная со строки y_offset)."""
        if self.ready is None:
            return None
        rows, cols, s, inside = self.ready
        rows = np.clip(rows, 0, gray.shape[0] - 1)
        vals = gray[rows, cols].astype(np.float32).mean(0)
        p = np.full(len(GRID), np.nan, np.float32)
        p[inside] = np.interp(GRID[inside], s, vals)
        return highpass(p)


class DisplacementTracker:
    """D(t) от покоя: первые REST_FRAMES кадров — опорный рисунок, дальше — цепочка сравнений."""

    def __init__(self):
        self.buffer = []
        self.ref = None
        self.ref_D = 0.0
        self.D = 0.0
        self.missed = 0
        self.history = []                     # (t, D, совпадение)
        self.outside = (GRID < COMPARE[0]) | (GRID > COMPARE[1])

    def add(self, t, prof):
        if prof is None:
            self.missed += 1
            return
        if self.ref is None:
            self.buffer.append((t, prof))
            if len(self.buffer) >= REST_FRAMES:
                self._start()
            return
        self._step(t, prof)

    def finish_rest(self):
        """Покой оказался короче REST_FRAMES — начать с тем, что есть."""
        if self.ref is None and len(self.buffer) >= 10:
            self._start()

    def _start(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # столбцы без доски
            self.ref = np.nanmedian(np.stack([p for _, p in self.buffer]), 0)
        buf, self.buffer = self.buffer, []
        for t, p in buf:
            self._step(t, p)

    def _step(self, t, prof):
        cur = prof.copy()
        cur[self.outside] = np.nan
        base = self.D - self.ref_D
        d, score = find_shift(self.ref, cur, base - 3, base + 8 + 4 * self.missed)
        if d is None or score < MIN_SCORE:
            self.missed += 1
            return
        self.missed = 0
        self.D = self.ref_D + d
        self.history.append((t, self.D, score))
        if d > RELINK_CM:
            self.ref, self.ref_D = prof, self.D
