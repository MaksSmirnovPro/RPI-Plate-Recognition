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

Для `Picamera2` создавайте виртуальное окружение с доступом к системным пакетам: так Python из venv увидит установленный через APT модуль `picamera2`.

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
Path("models/license_plate_yolo11n.onnx").write_bytes(Path(model).read_bytes())
print("Модель сохранена:", Path("models/license_plate_yolo11n.onnx").resolve())
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

```text
barrier/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── settings.yaml              # интервал кадров, пороги, GPIO, углы серво
│   └── allowed_plates.txt         # whitelist: один номер на строку
├── models/
│   └── license_plate_yolo11n.onnx # скачать отдельно; не хранить в Git при необходимости
├── src/
│   ├── __init__.py
│   ├── main.py                    # запуск приложения и главный цикл
│   ├── camera.py                  # получение кадров через Picamera2
│   ├── detector.py                # ONNX Runtime: YOLO-детекция рамки номера
│   ├── ocr.py                     # подготовка crop и Tesseract OCR
│   ├── access_control.py          # нормализация, whitelist, подтверждения
│   ├── barrier.py                 # SG-90 через gpiozero.AngularServo
│   └── logger.py                  # журналирование событий
├── data/
│   ├── captures/                  # необязательные диагностические кадры
│   └── logs/
└── tests/
    ├── test_access_control.py
    └── test_plate_normalization.py
```

### Примечания к структуре

- `models/*.onnx` — бинарные веса; скачиваются отдельно, в репозиторий исходников обычно не коммитятся.
- `config/allowed_plates.txt` — список разрешённых номеров, хранить с ограниченными правами доступа.
- `config/settings.yaml` — все настройки оборудования и алгоритма, чтобы не менять код при калибровке.
- `data/captures/` и `data/logs/` — локальные данные, не добавлять в Git.
- `tests/` — тесты логики whitelist и нормализации, которые можно запускать без камеры и сервопривода.



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

The barrier automatically closes 10 seconds after opening.

The servo is initialized in the closed position.

**Power supply:**
- Use a separate regulated 5V power supply for the servo.
- Connect the external power supply GND to Raspberry Pi GND.
- Do not power the servo from a GPIO pin.



## Первый запуск

```bash
source .venv/bin/activate
python -m src.main
```

Перед первым запуском убедитесь, что модель находится по пути `models/license_plate_yolo11n.onnx`, whitelist заполнен, камера доступна, а серво проверен без механической нагрузки.
