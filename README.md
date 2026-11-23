# Raspberry Pi — распознавание автомобильных номеров и управление шлагбаумом

Проект для Raspberry Pi 5, камеры IMX219 / Camera Module 2 и сервопривода SG-90. Камера непрерывно показывает изображение в GUI; фоновые модели обнаруживают транспорт и номерные знаки. OCR читает номер, AccessControl сравнивает его с локальным whitelist, а разрешённый номер открывает шлагбаум.

## Как устроена детекция

Используются две отдельные ONNX-модели:

- **YOLOv8n COCO** — обнаруживает транспортные средства (car, motorcycle, bus, truck); зелёная рамка.
- **YOLO11n License Plate** — обнаруживает область номерного знака; синяя рамка.
- **FastPlateOCR / Tesseract** — распознаёт символы только внутри crop номерного знака (выбор backend задаётся в конфигурации).
- **AccessControl** — нормализует строку и выполняет проверку по `barrier/config/allowed_plates.txt`.

В экспериментальной ветке по умолчанию используется `fast-plate-ocr` 0.3.x (нейросеть через ONNX Runtime, класс `ONNXPlateRecognizer`) вместо Tesseract. Старый Tesseract backend сохранён как резервный вариант и выбирается через `ocr.backend: tesseract`.

## Аппаратная часть

- Raspberry Pi 5 с 64-bit Raspberry Pi OS Desktop.
- Камера IMX219 / Camera Module 2, подключённая к CSI.
- SG-90 для демонстрационного макета.
- Отдельный стабилизированный источник 5 В для сервопривода; соединить GND источника с GND Raspberry Pi.
- Сигнал сервопривода: **BCM GPIO13, физический контакт 33**. Не путать BCM GPIO13 с физическим контактом 13.

SG-90 не предназначен для управления тяжёлой реальной стрелой. Для реального шлагбаума используйте привод и контроллер соответствующего класса, концевые выключатели и независимую защиту от защемления.

## Системные пакеты

Picamera2 и libcamera устанавливаются через APT, а не pip:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y \
  python3 python3-venv python3-pip python3-dev \
  python3-picamera2 python3-libcamera \
  tesseract-ocr tesseract-ocr-eng \
  libgtk-3-0 libgl1 libglib2.0-0
```

Проверьте камеру:

```bash
rpicam-hello --list-cameras
```

Для модуля Picamera2 создавайте venv с `--system-site-packages`, чтобы приложение видело системный пакет:

```bash
git clone https://github.com/MaksSmirnovPro/RPI-Plate-Recognition.git
cd RPI-Plate-Recognition
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Если ранее был установлен headless OpenCV, удалите конфликтующие варианты и поставьте зависимости проекта:

```bash
python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless
python -m pip install -r requirements.txt
```

Используется GUI-сборка `opencv-python`, поскольку приложение открывает live preview через `cv2.imshow`. Не устанавливайте одновременно `opencv-python-headless`.

## Python-зависимости

Файл `requirements.txt` содержит runtime-пакеты:

| Пакет | Назначение |
|---|---|
| `numpy` | массивы и обработка кадров |
| `opencv-python` | компьютерное зрение и GUI |
| `onnxruntime` | инференс обеих YOLO ONNX-моделей на CPU |
| `pytesseract` | резервный OCR backend Tesseract |\n| `fast-plate-ocr` | нейросетевой OCR номерного знака через ONNX Runtime |
| `PyYAML` | чтение настроек |
| `huggingface_hub` | загрузка готовых весов |
| `rpi-hardware-pwm` | аппаратный PWM для сервопривода |

**Важно про YOLO:** в Raspberry Pi устанавливаются ONNX Runtime и OpenCV. Пакет `ultralytics` и PyTorch не нужны для запуска приложения: они требуются только для экспорта модели в ONNX на отдельном компьютере. Для этого есть необязательный файл `requirements-model-tools.txt`.

Установка инструментов экспорта на ПК:

```bash
python -m pip install -r requirements-model-tools.txt
yolo export model=yolov8n.pt format=onnx imgsz=640
```

Полученный `yolov8n.onnx` скопируйте в `barrier/models/yolov8n.onnx`.

## Модели

Обе модели должны находиться в `barrier/models/`. Бинарные веса не хранятся в Git.

### Детектор автомобилей

Экспортируйте YOLOv8n COCO на ПК с Ultralytics:

```bash
python -m pip install -r requirements-model-tools.txt
yolo export model=yolov8n.pt format=onnx imgsz=640
```

Скопируйте файл `yolov8n.onnx` в:

```text
barrier/models/yolov8n.onnx
```

Код выбирает COCO-классы 2, 3, 5 и 7: car, motorcycle, bus, truck.

### Детектор номерных знаков

Используется готовая модель YOLO11n License Plate Detection:

- Модель: <https://huggingface.co/morsetechlab/yolov11-license-plate-detection>
- Файл: `license-plate-finetune-v1n.onnx`
- Путь: `barrier/models/license-plate-finetune-v1n.onnx` или путь, указанный в `barrier/config/settings.yaml`.

Скачайте файл через Hugging Face Hub:

```bash
cd barrier
mkdir -p models
python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

source = hf_hub_download(
    repo_id="morsetechlab/yolov11-license-plate-detection",
    filename="license-plate-finetune-v1n.onnx",
)
target = Path("models/license-plate-finetune-v1n.onnx")
target.write_bytes(Path(source).read_bytes())
print(f"Saved: {target.resolve()}")
PY
```

Проверьте, что путь в `config/settings.yaml` совпадает с фактическим именем файла. Перед использованием сторонних весов проверьте их лицензию и совместимость ONNX-входа/выхода с детектором проекта.

## Настройки

Основной конфигурационный файл: `barrier/config/settings.yaml`.

- `detector.model_path` — модель номерных знаков.
- `detector.vehicle_model_path` — COCO модель транспорта.
- `detector.*_confidence` — пороги детекции.
- `ocr.backend` — `fast_plate_ocr` или `tesseract`.\n- `ocr.model_name` — имя модели FastPlateOCR; по умолчанию `global-plates-mobile-vit-v2-model`.\n- `ocr.device` — `cpu` для Raspberry Pi.\n- `ocr.language` — язык Tesseract, если выбран резервный backend.
- `access.allowed_plates_file` — локальный whitelist.
- `access.required_confirmations` — количество последовательных подтверждений.
- `processing.interval_seconds` — пауза между циклами фоновой обработки (в этой экспериментальной ветке 0.5 секунды).
- `display.enabled` — GUI preview.
- `barrier.pwm_chip` / `pwm_channel` — PWM controller и канал.
- `closed_pulse_width_ms` / `opened_pulse_width_ms` — индивидуальная калибровка положений сервопривода.

Whitelist: один номер на строку в `barrier/config/allowed_plates.txt`. Используйте только те номера, которым действительно разрешён въезд.

## PWM и подключение SG-90

В текущей конфигурации проекта для Raspberry Pi 5 используются BCM GPIO13 (physical pin 33), PWM channel 1 и pwmchip 0. Индекс pwmchip может зависеть от ядра/overlay; проверьте реальное соответствие на устройстве, не меняйте индекс наугад.

Если требуется включить PWM overlay, сначала проверьте актуальную документацию Raspberry Pi OS и содержимое `/boot/firmware/config.txt`, сохраняя существующие настройки. После изменения перезагрузите Pi и проверьте `/sys/class/pwm/` и `pinctrl get 13`.

Сервопривод работает на 50 Hz. Перед калибровкой отсоедините тягу/рог от механизма. У разных SG-90 диапазон отличается; увеличивайте импульс малыми шагами и немедленно остановитесь при гуле, нагреве или упоре. Не пытайтесь силой довести сервопривод до механического предела.

## Запуск

```bash
cd ~/RPI-Plate-Recognition
source .venv/bin/activate
cd barrier
python -m src.main
```

Запускайте команду из графической сессии Raspberry Pi OS Desktop. При включённом GUI должны быть доступны `DISPLAY` или `WAYLAND_DISPLAY`.

Перед запуском проверьте наличие двух ONNX-файлов и whitelist:

```bash
ls -lh models/*.onnx
cat config/allowed_plates.txt
```

## Проверка и диагностика

```bash
python - <<'PY'
import cv2, onnxruntime, pytesseract
print("OpenCV:", cv2.__version__)
print("OpenCV GUI:", [x.strip() for x in cv2.getBuildInformation().splitlines() if x.strip().startswith("GUI:")])
print("ONNX Runtime:", onnxruntime.__version__)
print("Tesseract:", pytesseract.get_tesseract_version())
PY
```

Если GUI не открывается, проверьте, что установлена именно `opencv-python`, а не headless-сборка. Если OCR не работает, проверьте `tesseract --version` и наличие языка `eng`. Если модель не найдена, сверьте путь в YAML и имя файла.

## Структура runtime

```text
barrier/
├── config/
│   ├── settings.yaml
│   └── allowed_plates.txt
├── models/
│   ├── yolov8n.onnx
│   └── license-plate-finetune-v1n.onnx
├── src/
│   ├── main.py
│   ├── camera.py
│   ├── detector.py
│   ├── ocr.py
│   ├── access_control.py
│   ├── barrier.py
│   └── logger.py
└── data/
    └── logs/
```

## Безопасность и ограничения

1. Не открывайте шлагбаум только по наличию рамки: требуется распознанная строка и совпадение с whitelist.
2. Проверьте false positives и OCR-ошибки на реальных кадрах, включая ночь, блики, дождь и разные углы.
3. Настройте подтверждения и cooldown под условия эксплуатации; слишком малое число подтверждений снижает устойчивость к ошибочным распознаваниям.
4. Предусмотрите ручное аварийное управление и безопасное поведение при потере питания/связи.
5. Не используйте SG-90 для реальной тяжёлой стрелы шлагбаума.
6. ONNX-инференс выполняется на CPU; реальная задержка зависит от версии ОС, модели, разрешения и охлаждения Raspberry Pi. Измерьте её на устройстве.
