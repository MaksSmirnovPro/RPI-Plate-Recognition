# Шлагбаум с распознаванием автомобильных номеров

Проект для Raspberry Pi 5 + Camera Module 2 + сервопривод SG-90. Камера периодически получает кадр, YOLO находит область номерного знака, OCR извлекает символы, после проверки по локальному списку разрешённых номеров сервопривод открывает шлагбаум.

> Важно: обычные модели YOLO, обученные на COCO, не распознают номерные знаки как отдельный класс. Нужна уже обученная модель детекции номеров. В проекте предусмотрен готовый YOLO11n license-plate detector в ONNX; дополнительное обучение не требуется. YOLO выполняет **детекцию рамки номера**, а чтение букв и цифр выполняет отдельный OCR (Tesseract).

## Зависимости

### Аппаратная часть

- Raspberry Pi 5 (желательно 4 ГБ RAM или больше), 64-bit Raspberry Pi OS.
- Raspberry Pi Camera Module 2, подключённая к CSI-разъёму.
- SG-90 (180°), подключённый к PWM GPIO.
- Отдельный стабилизированный источник 5 В для SG-90 (ориентировочно 1 А или больше с запасом по пусковому току). Земля источника питания сервопривода должна быть соединена с GND Raspberry Pi.
- Не питать SG-90 от GPIO. Не подключать сигнальный провод к 5 В. Для сигнала удобно использовать GPIO18 (физический pin 12), поддерживающий аппаратный PWM.

### Системное ПО и системные пакеты

Предполагается 64-битная Raspberry Pi OS с рабочим стеком `libcamera` (на образе марта 2026 года проверьте фактический выпуск ОС и Python). Docker не требуется.

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y \
  python3 python3-venv python3-pip python3-dev \
  python3-picamera2 python3-libcamera \
  python3-gpiozero python3-lgpio \
  tesseract-ocr tesseract-ocr-eng \
  libcamera-dev
```

Если для используемого языка/региона номеров нужен другой набор символов Tesseract, установите соответствующий пакет `tesseract-ocr-<код-языка>`. Для большинства номерных знаков начните с `eng`.

Проверьте камеру:

```bash
rpicam-hello --list-cameras
```

Для `Picamera2` создавайте виртуальное окружение с доступом к системным пакетами: так Python из venv увидит установленный через APT модуль `picamera2`.

```bash
cd ~/barrier
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Python-зависимости

Полный список находится в `requirements.txt`:

- `numpy` — массивы и обработка кадров.
- `opencv-python-headless` — подготовка изображения и обработка рамок без GUI.
- `onnxruntime` — запуск ONNX-модели на CPU ARM64.
- `pytesseract` — Python-обёртка для системного Tesseract OCR.
- `gpiozero` — управление SG-90 через GPIO/PWM.

`picamera2`, `libcamera`, `gpiozero` и `lgpio` устанавливаются системными пакетами APT. Не устанавливайте поверх них случайные версии из PyPI: на Raspberry Pi важно сохранить совместимость с системным стеком камеры и GPIO.

### Модель YOLO (готовые веса, без обучения)

Используйте компактную **YOLO11n License Plate Detection** модель, уже дообученную автором репозитория для детекции номерных знаков:

- Репозиторий модели: <https://huggingface.co/morsetechlab/yolov11-license-plate-detection>
- Файл для устройства: `license-plate-finetune-v1n.onnx`
- Сохранить в проект как: `models/license_plate_yolo11n.onnx`
- Модельный репозиторий также содержит более крупные варианты; для Raspberry Pi 5 начинайте с варианта `n` (nano).
- Перед эксплуатацией проверьте лицензию и условия использования именно выбранных весов в карточке модели/репозитории.

Создайте каталог и скачайте модель с Hugging Face Hub:

```bash
mkdir -p models
python -m pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

model = hf_hub_download(
    repo_id="morsetechlab/yolov11-license-plate-detection",
    filename="license-plate-finetune-v1n.onnx",
)
Path("models/license-plate-finetune-v1n.onnx").write_bytes(Path(model).read_bytes())
print("Модель сохранена:", Path("models/license-plate-finetune-v1n.onnx").resolve())
PY
```

`huggingface_hub` нужен только для первоначальной загрузки модели и не входит в обязательный runtime `requirements.txt`. Его можно установить отдельной командой, как показано выше. Альтернативно скачайте файл вручную со страницы Files and versions этой модели.

**Совместимость модели:** до запуска убедитесь, что в ONNX-файле есть вход/выход, ожидаемые используемым кодом детектора. Размер входа, формат RGB/BGR, нормализация и постобработка (включая NMS) должны соответствовать экспорту конкретной модели. Не следует считать любой файл с расширением `.onnx` взаимозаменяемым с любым YOLO. Если автор обновит имя файла, берите точное имя из вкладки Files and versions.

### Настройка и программы для разработки

- Python 3 и `venv` — среда приложения.
- `rpicam-hello` — проверка Camera Module 2.
- Tesseract OCR — системная программа, запускаемая через `pytesseract`.
- ONNX Runtime — инференс модели без PyTorch.
- Git — рекомендуется для версионирования исходного кода (установка: `sudo apt install git`).

### Рекомендуемый режим нагрузки

Не запускайте YOLO на каждом кадре видеопотока. Камера может работать с небольшим preview/FPS, а детектор запускать по таймеру, например раз в **1–2 секунды**. Начальные параметры для настройки:

- `CAPTURE_INTERVAL_SEC = 1.5` — интервал между кадрами для анализа.
- `INFERENCE_WIDTH = 640` — начальная ширина кадра; при необходимости уменьшить.
- `YOLO_CONFIDENCE = 0.45` — начальный порог уверенности, подбирается на реальных кадрах.
- Сначала обнаружить номер, затем выполнять OCR только на вырезанной области номера.
- Для открытия требовать совпадения номера с локальным whitelist и несколько последовательных подтверждений (например, 2 распознавания подряд).
- После открытия включать cooldown, чтобы не дёргать сервопривод повторно.
- Значения выше — стартовые настройки, а не гарантированные оптимальные параметры. Измерьте фактическую задержку и точность на месте установки.

### Важные ограничения и безопасность

1. Детектор находит номер, но сам по себе не определяет, кому разрешён въезд. Для этого нужен whitelist (например, `config/allowed_plates.txt`) и точная нормализация OCR-строки.
2. Точность зависит от освещения, угла, расстояния, загрязнения номера и регионального формата. Проверьте модель на кадрах именно вашей камеры и места установки.
3. Не открывайте шлагбаум только по наличию любой рамки номера: используйте OCR, whitelist, временную фильтрацию и защиту от повторных срабатываний.
4. Предусмотрите ручное аварийное управление и безопасное положение шлагбаума при сбое питания/программы. SG-90 — демонстрационный привод, он может быть недостаточно мощным для реальной стрелы шлагбаума.
5. Проверьте ток питания, механические упоры и диапазон PWM SG-90 до установки рычага. Не удерживайте сервопривод в механически заблокированном положении.
6. ONNX Runtime на CPU уменьшает зависимости по сравнению с PyTorch, но не гарантирует определённую частоту кадров. Замерьте производительность на вашей версии ОС и настройте интервал обработки.

## Структура

### Реализованная структура (barrier/ — Raspberry Pi runtime)

```text
barrier/
├── config/
│   ├── settings.yaml              # все настройки: камера, детектор, OCR, доступ, шлагбаум, processing
│   └── allowed_plates.txt         # whitelist: по одному номеру на строку (6 номеров)
├── models/
│   ├── license-plate-finetune-v1n.onnx   # 10.5 MB — рекомендуется для Pi 5
│   ├── license-plate-finetune-v1n.pt
│   ├── license-plate-finetune-v1s.onnx   # 37.8 MB
│   ├── license-plate-finetune-v1m.onnx   # 80.3 MB
│   ├── license-plate-finetune-v1m.pt
│   └── license-plate-finetune-v1l.onnx   # 101 MB
├── src/
│   ├── __init__.py
│   ├── main.py                    # LicensePlateSystem — главный orquestратор + CLI
│   ├── camera.py                  # Camera — Picamera2 wrapper (RGB888, 640x480)
│   ├── detector.py                # PlateDetector — YOLO11n ONNX через ONNX Runtime
│   ├── ocr.py                     # PlateOCR — Tesseract с 8 вариантами предобработки + голосование
│   ├── access_control.py          # AccessControl — whitelist, подтверждения, cooldown
│   ├── barrier.py                 # Barrier — SG-90 через gpiozero.AngularServo
│   └── logger.py                  # setup_logger — RotatingFileHandler + console
├── tests/
│   ├── test_access_control.py     # whitelist, подтверждения, сброс счётчика
│   └── test_plate_normalization.py # normalize() тесты для AccessControl и PlateOCR
└── data/                          # создаётся автоматически при запуске
    ├── captures/                  # диагностические кадры (не используется в текущей версии)
    └── logs/
        └── barrier.log            # ротирующие логи (2MB, 3 бэкапа)
```

### Дополнительная структура (barrier/windows/ — оценка качества)

```text
barrier/windows/
├── config/settings.yaml           # настройки для оценки на датасете
├── dataset/
│   ├── images/                    # 4 тестовых изображения
│   └── labels.csv                 # разметка: image, plate, [x1,y1,x2,y2]
├── results/                       # генерируются автоматически
│   ├── annotated/                 # изображения с нарисованными GT и детекциями
│   ├── crops/                     # вырезанные области номерных знаков
│   └── rectification/             # результаты выпрямления номеров
├── src/
│   ├── main.py                    # CLI: запуск оценки датасета (--show флаг)
│   ├── detector.py                # PlateDetector с output_format (auto/yolo/nms)
│   ├── ocr.py                     # PlateOCR с tesseract_cmd и детальным логированием
│   └── evaluate.py                # DatasetEvaluator — метрики IoU, exact match, coverage
└── tests/
    ├── test_normalization.py      # unit-тесты normalize_plate, calculate_iou
    ├── test_ocr.py                # диагностика OCR на crops
    ├── test_ocr_1.py              # глубокая диагностика: thresholds, adaptive, illumination
    └── test_rectification.py      # выпрямление кривых номеров + OCR сравнение
```

### Примечания к структуре

- `models/*.onnx` / `models/*.pt` — бинарные веса; скачиваются отдельно, в репозиторий исходников обычно не коммитятся (см. `.gitignore`).
- `config/allowed_plates.txt` — список разрешённых номеров, хранить с ограниченными правами доступа.
- `config/settings.yaml` — все настройки оборудования и алгоритма, чтобы не менять код при калибровке.
- `data/captures/` и `data/logs/` — локальные данные, не добавлять в Git.
- `tests/` — тесты логики whitelist и нормализации, которые можно запускать без камеры и сервопривода.
- Windows-подпроект в `barrier/windows/` — для разработки и оценки качества на ПК перед деплоем на Pi.

## Pinouts

### SG-90 Servo Motor

| SG-90 wire | Raspberry Pi GPIO | Physical pin |
|---|---|---|
| Signal (orange) | GPIO18 | 12 |
| VCC (red) | External +5V | — |
| GND (brown) | GND | 6 |

The SG-90 signal wire is connected to GPIO18.

Barrier positions:
- 180° — closed
- 90° — open

The barrier automatically closes 10 seconds after opening (configurable via `open_duration`).

The servo is initialized in the closed position.

**Power supply:**
- Use a separate regulated 5V power supply for the servo.
- Connect the external power supply GND to Raspberry Pi GND.
- Do not power the servo from a GPIO pin.

## Конфигурация (config/settings.yaml)

```yaml
camera:
  width: 640
  height: 480

detector:
  model_path: models/license-plate-finetune-v1n.onnx
  input_size: 640
  confidence: 0.45
  nms_threshold: 0.45

ocr:
  language: eng
  min_confidence: 0

access:
  allowed_plates_file: config/allowed_plates.txt
  required_confirmations: 1
  cooldown_seconds: 1

barrier:
  gpio_pin: 18
  closed_angle: 180
  opened_angle: 90
  open_duration: 7

processing:
  interval_seconds: 2
```

### Основные параметры

| Секция | Параметр | Описание |
|--------|----------|----------|
| camera | width/height | Разрешение камеры (RGB888) |
| detector | model_path | Путь к ONNX модели относительно barrier/ |
| detector | confidence | Порог уверенности YOLO |
| detector | nms_threshold | IoU порог для NMS |
| ocr | language | Язык Tesseract (eng) |
| access | required_confirmations | Сколько раз подряд распознать номер |
| access | cooldown_seconds | Пауза после открытия шлагбаума |
| barrier | gpio_pin | GPIO пин сервы (18 = pin 12) |
| barrier | open_duration | Время открытого состояния (сек) |
| processing | interval_seconds | Интервал цикла распознавания |

## Первый запуск

```bash
# 1. Клонировать репозиторий
cd ~/barrier

# 2. Создать виртуальное окружение с доступом к системным пакетам
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 3. Установить Python-зависимости
pip install --upgrade pip
pip install -r requirements.txt

# 4. Скачать модель YOLO (если не скачана)
mkdir -p models
python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
model = hf_hub_download(
    repo_id="morsetechlab/yolov11-license-plate-detection",
    filename="license-plate-finetune-v1n.onnx",
)
Path("models/license-plate-finetune-v1n.onnx").write_bytes(Path(model).read_bytes())
print("Модель сохранена:", Path("models/license-plate-finetune-v1n.onnx").resolve())
PY

# 5. Проверить конфигурацию
cat config/settings.yaml
cat config/allowed_plates.txt

# 6. Запуск
python -m src.main
```

Перед первым запуском убедитесь, что:
- Модель находится по пути `models/license-plate-finetune-v1n.onnx`
- Whitelist заполнен в `config/allowed_plates.txt`
- Камера доступна (`rpicam-hello --list-cameras`)
- Серво проверено без механической нагрузки
- Подключено отдельное питание 5V для SG-90 с общей землей

## Оценка качества на Windows (barrier/windows/)

Для разработки и подбора параметров без Raspberry Pi:

```bash
cd barrier/windows
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Запуск оценки на датасете
python -m src.main

# Диагностика OCR
python -m tests.test_ocr
python -m tests.test_ocr_1

# Выпрямление номеров
python -m tests.test_rectification
```

Результаты сохраняются в `barrier/windows/results/`:
- `results.csv` — таблица с метриками
- `annotated/` — изображения с GT (синий) и детекциями (зелёный)
- `crops/` — вырезанные номера для анализа OCR
- `rectification/comparison.csv` — сравнение OCR до/после выпрямления

## Архитектура

Подробное описание архитектуры, граф зависимостей, поток данных и ключевые решения см. в **Architecture.md**.

### Ключевые моменты

1. **Два потока**: камера (main thread, 30+ FPS) + распознавание (worker, 0.5 Hz)
2. **Barrier state machine**: vehicle_present продлевает открытие, cooldown защищает от дребезга
3. **OCR robustness**: 8 предобработок × 4 PSM = 32 попытки, голосование по частоте и уверенности
4. **Universal YOLO parser**: автоопределение форматов NMS-ready и raw YOLO
5. **Access control**: нормализация + N подтверждений + cooldown