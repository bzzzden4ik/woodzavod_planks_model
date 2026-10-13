# Обучение модели на видеокарте

Дообучение готовой нейросети YOLO26 на размеченных кадрах (не с нуля).
Время: установка ~15–20 минут (скачивается ~3 ГБ), обучение на RTX 5070 — 10–30 минут.
Все команды — в PowerShell, **из папки проекта** `machine_learning`.

## 0. Что нужно

- Компьютер с видеокартой NVIDIA.
- Этот репозиторий (`git clone ...` или архив «Source code» со страницы Releases).
- **Датасет** — папка `datasets\boards_v1` (кадры + разметка). В git его нет (большой).
  Два способа получить:
  - собрать из кадров и разметки: нужны кадры в `data\frames` (их выбирает
    `tools\select_frames.py` из часовых видео в `data\raw`) и файлы экспорта разметки
    из `labeling\`:
    ```powershell
    .venv\Scripts\python.exe tools\build_dataset.py --exports labeling\export_round1.json labeling\export_round2.json --val hiv00003 --name boards_v1
    ```
  - или скопировать готовую папку `datasets\boards_v1` с компьютера, где её собрали.

## 1. Драйвер видеокарты

Нужен свежий драйвер NVIDIA (для RTX 50-й серии — **570 или новее**). Проверка:

```powershell
nvidia-smi
```

В правом верхнем углу таблицы — `CUDA Version`: должно быть 12.8 или выше. Если команда не
найдена или версия старая — обновите драйвер (приложение NVIDIA или nvidia.com).

## 2. Python 3.12

Скачать с python.org, при установке отметить **«Add python.exe to PATH»**.
Проверка: `py -3.12 --version`.

## 3. Окружение и библиотеки (один раз)

```powershell
cd путь\к\проекту\machine_learning
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch ставится **именно со ссылкой `.../whl/cu128`** — это сборка с поддержкой видеокарты
(в том числе RTX 50-й серии). Обычный `pip install torch` поставит версию без видеокарты.
Это же окружение подходит и для запуска программы (`measure_boards.py`) на видеокарте.

Проверка, что видеокарта видна:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Должно быть `True` и название видеокарты.

## 4. Обучение

```powershell
.venv\Scripts\python.exe tools\train.py --data datasets\boards_v1\data.yaml --name boards_v2_s --model yolo26s-seg.pt --device 0 --epochs 300 --batch 16
```

- `yolo26s-seg` — модель среднего размера. Именно **YOLO26**: у YOLO11 рамка объекта не может
  быть шире ~75% входа, и длинные доски обрезались.
- Обучение само остановится раньше, если 30 эпох подряд нет улучшения.
- Ошибка нехватки памяти видеокарты (`CUDA out of memory`) — повторить с `--batch 8`.
- В конце: `Качество на проверочных кадрах: mAP50 маски = ...` (модель v2: 0.958).

Результат — `runs\segment\boards_v2_s\weights\best.pt` и графики `runs\segment\boards_v2_s\results.png`.

## 5. Новая модель — в работу

1. Скопировать `runs\segment\<имя>\weights\best.pt` в `models\best.pt`.
2. Проверить на тестовом видео с известными длинами (README.md, раздел 7):
   ```powershell
   .venv\Scripts\python.exe measure_boards.py data\sample.mp4 --no-video
   ```
3. Если результат лучше прежнего — выложить новый релиз и обновить `model.url` и `model.sha256`
   в `config.toml` (README.md, раздел 2).
