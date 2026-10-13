"""
Длина доски.

Когда доска лежит перед станком, её левый торец у входа в станок (P_L), а правый
у 10-метровой доски — за краем кадра. Поэтому длина считается в движении:

    положение правого торца в момент покоя  P_R = s_торца(t) + D(t)
    длина = P_R - P_L

  s_торца(t) — где правый торец в момент t (см). Берётся прямая по уверенным ответам
               модели, пока торец идёт через центр кадра (станок тянет доску ровно,
               ~70 см/с). Прямая сглаживает ошибки отдельных кадров.
  D(t)       — на сколько доска уехала с момента покоя (tracker.py).
Берутся моменты, где известны оба: прямая продлевается назад не больше чем на BACK_S.

Если доска короткая и её правый торец виден, пока она лежит, длина = s_торца - P_L.
"""

import numpy as np

END_CONF = 0.65            # ответы модели слабее — не используем (бывают «призраки» доски)
END_ZONE = (250, 2380)     # где торец виден надёжно (x кадра): не у краёв кадра
MIN_LINE_POINTS = 8        # столько точек торца на прямой, чтобы ей верить
SPEED_RANGE = (-150, -30)  # допустимая скорость торца, см/с (минус — влево, к станку)
LINE_TOL_CM = 8.0          # точка дальше от прямой — ошибка модели
BACK_S = 3.5               # насколько назад можно продлить прямую, с
AFTER_S = 1.0              # и вперёд, с
MIN_D_CM = 20.0            # доска должна уже ехать
MIN_OVERLAP = 3            # минимум общих моментов «прямая + D»


def fit_line(t, s, seed=0):
    """Прямая s = a*t + b по точкам торца, устойчивая к выбросам (RANSAC).
    Возвращает ((a, b), маска точек на прямой) или (None, None)."""
    if len(t) < MIN_LINE_POINTS:
        return None, None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(400):
        i, j = rng.choice(len(t), 2, replace=False)
        if t[i] == t[j]:
            continue
        a = (s[j] - s[i]) / (t[j] - t[i])
        if not SPEED_RANGE[0] < a < SPEED_RANGE[1]:
            continue
        inl = np.abs(s - (s[i] + a * (t - t[i]))) < LINE_TOL_CM
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < MIN_LINE_POINTS:
        return None, None
    return tuple(np.polyfit(t[best], s[best], 1)), best


def length_from_motion(end_points, d_history, p_left):
    """end_points: [(t, s_торца)], d_history: [(t, D, совпадение)].
    Возвращает словарь с результатом; length_cm = None, если данных пока не хватает."""
    out = dict(length_cm=None, line=None, line_start=None, speed=None, n_line=0, n_overlap=0)
    if len(end_points) < MIN_LINE_POINTS:
        return out
    e = np.array(end_points, float)
    line, inl = fit_line(e[:, 0], e[:, 1])
    if line is None:
        return out
    t0 = float(e[inl, 0].min())
    out.update(line=line, line_start=t0, speed=float(line[0]), n_line=int(inl.sum()))
    if not d_history:
        return out
    h = np.array(d_history, float)
    sel = (h[:, 0] >= t0 - BACK_S) & (h[:, 0] <= t0 + AFTER_S) & (h[:, 1] > MIN_D_CM)
    out["n_overlap"] = int(sel.sum())
    if sel.sum() >= MIN_OVERLAP:
        p_right = np.polyval(line, h[sel, 0]) + h[sel, 1]
        out["length_cm"] = float(np.median(p_right)) - p_left
    return out
