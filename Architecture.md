# Архитектура проекта RPI-Plate-Recognition

---

## Обзор проекта

Проект реализует систему распознавания автомобильных номеров для управления шлагбаумом на Raspberry Pi 5.

**Две реализации:**
1. **barrier/** — Полная целевая реализация для Raspberry Pi (runtime)
2. **barrier/windows/** — Windows-подпроект для оценки качества детекции и OCR (диагностика)

### Основные компоненты (barrier/)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| Главная система | src/main.py | LicensePlateSystem — оркестратор всех компонентов |
| Камера | src/camera.py | Camera — Picamera2 wrapper (RGB888, 640x480) |
| Детекция | src/detector.py | PlateDetector — YOLO11n ONNX через ONNX Runtime |
| OCR | src/ocr.py | PlateOCR — Tesseract с 8 вариантами предобработки + голосование |
| Контроль доступа | src/access_control.py | AccessControl — whitelist, подтверждения, cooldown |
| Шлагбаум | src/barrier.py | Barrier — SG-90 через gpiozero.AngularServo |
| Логирование | src/logger.py | setup_logger — rotating file + console |
| Конфигурация | config/settings.yaml | YAML настройки всех подсистем |
| Whitelist | config/allowed_plates.txt | Разрешённые номера (по одному на строку) |
| Модели | models/*.onnx | YOLO11n/s/m/l веса (v1n рекомендуется для Pi 5) |

---

## Файловая структура (полная)

```
RPI-Plate-Recognition/
├── README.md
├── Architecture.md
├── requirements.txt
├── .gitignore
├── barrier/
│   ├── config/
│   │   ├── settings.yaml              # все настройки системы
│   │   └── allowed_plates.txt         # whitelist: 6 номеров
│   ├── models/
│   │   ├── license-plate-finetune-v1n.onnx   # 10.5 MB (рекомендуется)
│   │   ├── license-plate-finetune-v1n.pt     # 5.5 MB
│   │   ├── license-plate-finetune-v1s.onnx   # 37.8 MB
│   │   ├── license-plate-finetune-v1m.onnx   # 80.3 MB
│   │   ├── license-plate-finetune-v1m.pt     # 40.5 MB
│   │   └── license-plate-finetune-v1l.onnx   # 101 MB
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py                    # LicensePlateSystem + CLI
│   │   ├── camera.py                  # Picamera2 wrapper
│   │   ├── detector.py                # PlateDetector (ONNX YOLO)
│   │   ├── ocr.py                     # PlateOCR (Tesseract + голосование)
│   │   ├── access_control.py          # AccessControl (whitelist + подтверждения)
│   │   ├── barrier.py                 # Barrier (gpiozero SG-90)
│   │   └── logger.py                  # RotatingFileHandler + console
│   ├── tests/
│   │   ├── test_access_control.py     # whitelist, подтверждения, сброс
│   │   └── test_plate_normalization.py # normalize() тесты
│   └── data/                          # создаётся в runtime
│       ├── captures/
│       └── logs/
└── barrier/windows/                   # Windows-оценка качества
    ├── .venv/
    ├── config/settings.yaml
    ├── dataset/images/ + labels.csv
    ├── results/ (annotated, crops, rectification)
    ├── src/ (main.py, detector.py, ocr.py, evaluate.py)
    └── tests/ (4 диагностических скрипта)
```

---

## Детальное описание каждого файла (barrier/)

---

### 1. barrier/src/main.py

**Назначение**: Главная точка входа — класс LicensePlateSystem оркестрирует все компоненты в runtime режиме.

#### Класс LicensePlateSystem

##### Конструктор __init__()

Инициализирует все подсистемы из config/settings.yaml:

| Подсистема | Конфиг секция | Параметры |
|------------|---------------|-----------|
| Camera | camera | width=640, height=480 |
| Detector | detector | model_path, input_size=640, confidence=0.45, nms_threshold=0.45 |
| OCR | ocr | language=eng, min_confidence=0 |
| AccessControl | access | allowed_plates_file, required_confirmations=1, cooldown_seconds=1 |
| Barrier | barrier | gpio_pin=18, closed_angle=180, opened_angle=90, open_duration=7 |
| Processing | processing | interval_seconds=2 |

**Атрибуты состояния**:
- frame_lock — threading.Lock для синхронизации кадров
- latest_frame — последний кадр от камеры
- result_frame — кадр с аннотацией для отображения
- result_until — timestamp до которого показывать результат
- stop_event — threading.Event для graceful shutdown
- worker — фоновый поток recognition_worker

##### Метод load_config() — staticmethod

Загружает config/settings.yaml относительно PROJECT_DIR.

##### Метод recognition_worker() — фоновый поток

**Алгоритм** (каждые processing_interval секунд, по умолчанию 2с):

1. Берёт копию кадра под frame_lock
2. YOLO detection — detector.detect(frame) → список боксов
3. Сортировка по confidence (desc)
4. Barrier vehicle_present — barrier.update(vehicle_present=len(detections)>0)
5. Для каждого бокса (берёт только первый валидный):
   - Crop области номера
   - ocr.recognize(crop) → plate string
   - Аннотирует кадр (зелёный бокс + текст номера)
   - access.confirm_plate(plate) → если True: barrier.open_for(), лог "ACCESS GRANTED"
6. Если ничего не распознано — access.confirm_plate("") (сброс счётчика)
7. Ожидание stop_event.wait(processing_interval)

##### Метод run() — главный цикл (main thread)

1. camera.start(), barrier.close()
2. Запускает recognition_worker в daemon thread
3. OpenCV window loop:
   - camera.read() → обновляет latest_frame под lock
   - Показывает result_frame если не устарел, иначе live frame
   - cv2.waitKey(1) — выход по 'q'
4. Graceful shutdown в finally:
   - stop_event.set(), worker.join(timeout=5)
   - cv2.destroyAllWindows()
   - camera.stop(), barrier.stop()

##### Функция main()

Создаёт LicensePlateSystem() и вызывает run().

---

### 2. barrier/src/detector.py

**Назначение**: PlateDetector — ONNX Runtime инференс YOLO модели.

#### Конструктор __init__

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| model_path | str | — | Путь к .onnx |
| input_size | int | 640 | Квадратный вход модели |
| confidence | float | 0.45 | Порог фильтрации боксов |
| nms_threshold | float | 0.45 | IoU для NMS |

**Атрибуты**:
- session — ort.InferenceSession(providers=["CPUExecutionProvider"])
- input_name — имя входного тензора

#### Метод detect(frame) → List[(x1,y1,x2,y2,conf)]

**Letterbox preprocessing**:
1. Масштаб scale = min(input_size/width, input_size/height)
2. Resize с сохранением пропорций
3. Padding серым (114) до input_size x input_size
4. cv2.dnn.blobFromImage: scalefactor=1/255, swapRB=True (BGR→RGB)

**Inference**: session.run(None, {input_name: blob})

**Parsing** — _parse_output():

| Формат выхода | Определение | Обработка |
|---------------|-------------|-----------|
| [N, 6] (x1,y1,x2,y2,conf,class) | output.shape[1] == 6 | Прямое чтение боксов |
| YOLOv8/11 [4+classes, N] | иначе | Транспонирование при необходимости, cx,cy,w,h → x1,y1,x2,y2, max(class_scores) |

**NMS**: cv2.dnn.NMSBoxes(boxes, scores, confidence, nms_threshold)

**Coordinate mapping**: Обратное letterbox преобразование с clamp к исходным размерам.

#### Возврат

Список (x1, y1, x2, y2, confidence) в координатах исходного кадра.

---

### 3. barrier/src/ocr.py

**Назначение**: PlateOCR — Tesseract OCR с множественной предобработкой и голосованием.

#### Константы модуля

| Константа | Значение |
|-----------|----------|
| PLATE_LETTERS | "ABEKMHOPCTYX" (12 букв российских номеров) |
| PLATE_PATTERN | ^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}$ |
| CYRILLIC_TO_LATIN | 12 маппингов (А→A, В→B, Е→E, К→K, М→M, Н→H, О→O, Р→P, С→C, Т→T, У→Y, Х→X) |
| DIGIT_MAP | O,Q,D→0; I,L→1; Z→2; S→5; B→8 |
| LETTER_MAP | 0→O; 8→B; 4→A |
| WHITELIST | A-Z0-9 |
| PSM_MODES | (7, 8, 13, 6) — single line, word, raw, block |

#### Методы класса

##### normalize(text) — classmethod

Верхний регистр + кириллица→латиница + только алфавитно-цифровые.

##### extract_plate(text) — classmethod

**Слайдинг окно 6 символов** с позиционными картами:
- Позиция 0 (буква): LETTER_MAP
- Позиции 1-3 (цифры): DIGIT_MAP
- Позиции 4-5 (буквы): LETTER_MAP
- Проверка PLATE_PATTERN.fullmatch()

##### add_border(image) — staticmethod

Белые поля: top/bottom 30px, left/right 40px.

##### preprocess(plate_image) → 8 вариантов

| Вариант | Описание |
|---------|----------|
| enlarged | 4x CUBIC |
| enhanced | enlarged + CLAHE(2.0, 8x8) |
| otsu_original | Otsu на сером |
| otsu | Otsu на enhanced |
| adaptive_21 | Gauss C=7, block=21 |
| adaptive_31 | Gauss C=11, block=31 |
| sharpened | Unsharp: enhanced*1.8 - blur*0.8 |
| otsu_inverted | Инвертированный Otsu |

Все через add_border().

##### recognize_variant(image, psm) → (raw_text, plate, avg_conf)

Tesseract с конфигом: --oem 3 --psm {psm} -c whitelist={WHITELIST}

Возвращает: конкатенированные слова, извлечённый номер, средняя уверенность.

##### recognize_variants(image) → best_plate

**Голосование**:
- Для каждого из 8 вариантов × 4 PSM = 32 попытки
- Собирает candidates[plate] = [confidences...]
- max(len(list), mean(confidences))

##### recognize(plate_image) → str

1. recognize_variants(plate_image)
2. Если пусто — fallback: JPEG 95 quality → снова recognize_variants

---

### 4. barrier/src/camera.py

**Назначение**: Camera — обёртка над Picamera2.

#### Конструктор

| Параметр | Default |
|----------|---------|
| width | 640 |
| height | 480 |

#### Методы

| Метод | Описание |
|-------|----------|
| start() | Picamera2(), create_video_configuration(RGB888, 640x480, buffer_count=2), start(), sleep(1.0) для автоэкспозиции |
| read() | capture_array() → RGB → cv2.cvtColor(RGB2BGR) |
| stop() | stop(), close(), camera = None |

---

### 5. barrier/src/access_control.py

**Назначение**: AccessControl — whitelist + подтверждения + cooldown.

#### Конструктор

| Параметр | Default | Описание |
|----------|---------|----------|
| allowed_plates_file | — | Путь к файлу |
| required_confirmations | 2 | Сколько раз подряд распознать |
| cooldown_seconds | 10.0 | Пауза после открытия |

**Состояние**:
- last_plate — последний распознанный нормализованный номер
- confirmation_count — счётчик повторов
- last_open_time — timestamp последнего открытия

#### Методы

##### normalize(plate) — classmethod

Аналогично PlateOCR.normalize(): upper + кириллица→латиница + alnum.

##### _load_allowed_plates() → set

Читает файл, пропускает пустые и #, нормализует, возвращает set.

##### is_allowed(plate) → bool

Нормализует + проверка вхождения в allowed_plates.

##### confirm_plate(plate) → bool

**Алгоритм**:
1. Нормализация
2. Если не разрешён или пустой → сброс last_plate=None, count=0, return False
3. Если тот же номер → count++, иначе last_plate=plate, count=1
4. Если count < required_confirmations → False
5. Cooldown check: now - last_open_time < cooldown → False
6. last_open_time = now, count=0, last_plate=None, return True

---

### 6. barrier/src/barrier.py

**Назначение**: Barrier — управление SG-90 через gpiozero.AngularServo.

#### Конструктор

| Параметр | Default |
|----------|---------|
| gpio_pin | 18 |
| closed_angle | 180 |
| opened_angle | 90 |
| open_duration | 5.0 сек |

**Инициализация**: AngularServo(pin, min_angle=0, max_angle=180, min_pulse_width=0.5ms, max_pulse_width=2.5ms, initial_angle=closed_angle)

#### Методы

| Метод | Описание |
|-------|----------|
| open() | angle=opened_angle, is_open=True, close_at = now + open_duration |
| close() | angle=closed_angle, is_open=False, close_at=None |
| keep_open(duration) | Продлевает close_at если открыт |
| update(vehicle_present) | Вызывается из worker: если vehicle_present → продлевает таймер; если нет и now >= close_at → close() |
| open_for(duration) | Неблокирующий alias для совместимости |
| stop() | close() + servo.close() |

---

### 7. barrier/src/logger.py

**Назначение**: setup_logger(name) — настройка логирования.

**Конфигурация**:
- Level: INFO
- Format: %(asctime)s | %(levelname)s | %(name)s | %(message)s
- Handlers: StreamHandler (console) + RotatingFileHandler(data/logs/barrier.log, maxBytes=2MB, backupCount=3)

---

### 8. barrier/config/settings.yaml

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

---

### 9. barrier/config/allowed_plates.txt

```
A123BC
B456CD
C789EF
C007CY
M777PO
O001OO
```

---

### 10. barrier/tests/test_access_control.py

| Тест | Проверяет |
|------|-----------|
| test_allowed_plate | is_allowed() — нормализация, case-insensitive, неразрешённый |
| test_confirmation_required | confirm_plate() — нужно N подтверждений |
| test_unknown_plate_resets_confirmation | Чужой номер сбрасывает счётчик |

---

### 11. barrier/tests/test_plate_normalization.py

| Тест | Проверяет |
|------|-----------|
| test_access_normalization | AccessControl.normalize() |
| test_ocr_normalization | PlateOCR.normalize() |

---

### 12. requirements.txt

```
numpy>=1.26,<3
opencv-python-headless>=4.9,<5
onnxruntime>=1.17,<2
pytesseract>=0.3.10,<0.4
gpiozero>=2.0,<3
PyYAML>=6.0,<7
huggingface_hub>=0.23,<2
libcap-dev
picamera2
python3-opencv
```

---

## Граф зависимостей (barrier/)

```
main.py:main()
  |-- LicensePlateSystem.__init__()
        |-- setup_logger() -> logger.py
        |-- load_config() -> settings.yaml
        |-- Camera.start() -> camera.py (Picamera2)
        |-- PlateDetector.__init__() -> detector.py (ONNX Runtime)
        |-- PlateOCR.__init__() -> ocr.py (Tesseract)
        |-- AccessControl.__init__() -> access_control.py
              |-- _load_allowed_plates() -> allowed_plates.txt
        |-- Barrier.__init__() -> barrier.py (gpiozero)
  |-- LicensePlateSystem.run()
        |-- camera.start()
        |-- barrier.close()
        |-- Thread(recognition_worker)
              |-- (loop: wait interval)
              |-- camera.read() -> latest_frame
              |-- detector.detect() -> boxes
              |-- barrier.update(vehicle_present)
              |-- for box in boxes:
                    |-- crop = frame[y1:y2, x1:x2]
                    |-- ocr.recognize(crop) -> plate
                    |-- access.confirm_plate(plate) -> grant?
                          |-- barrier.open_for()
              |-- if no plate: access.confirm_plate("")
        |-- Main loop:
              |-- camera.read() -> display
              |-- cv2.imshow()
              |-- waitKey(q)
        |-- Shutdown: stop_event, join, destroyAllWindows, camera.stop(), barrier.stop()
```

---

## Поток данных (Runtime)

```
settings.yaml + allowed_plates.txt
        |
        v
LicensePlateSystem.__init__()
        |
        v
+------------------+     +------------------+
| Camera thread    |     | Recognition      |
| (main thread)    |     | Worker (daemon)  |
|                  |     |                  |
| camera.read()    |     | detector.detect()|
| -> latest_frame  |---->| -> boxes         |
| cv2.imshow()     |     | barrier.update() |
| waitKey(q)       |     | ocr.recognize()  |
|                  |     | -> plate         |
|                  |     | access.confirm() |
|                  |     | -> barrier.open()|
+------------------+     +------------------+
        |
        v
Graceful shutdown
```

---

## Windows-подпроект (barrier/windows/) — Сводка

Отдельная реализация для **оценки качества** на датасете:

| Файл | Отличие от barrier/ |
|------|---------------------|
| src/detector.py | Добавил output_format (auto/yolo/nms), логирование |
| src/ocr.py | Добавил tesseract_cmd, более детальное логирование, preprocess публичный |
| src/evaluate.py | DatasetEvaluator — полная оценка на датасете (IoU, crops, annotated) |
| src/main.py | CLI для запуска оценки, --show флаг |
| tests/test_rectification.py | Выпрямление номеров (Sobel + квадратичная аппроксимация + remap) |
| tests/test_ocr_1.py | Глубокая диагностика: thresholds, adaptive, illumination, profiles |
| config/settings.yaml | Добавил dataset, evaluation секции |

---

## Ключевые архитектурные решения

### 1. Разделение потоков (Camera vs Recognition)
- **Main thread**: камера + UI (OpenCV window) — 30+ FPS
- **Worker thread**: детекция + OCR + доступ — каждые 2 сек
- **Lock-free frame passing** через threading.Lock + копирование

### 2. Barrier state machine
- vehicle_present продлевает открытие (не закрывает пока машина там)
- Cooldown после открытия защищает от дребезга
- Неблокирующий open_for() — worker не ждёт серво

### 3. OCR Robustness
- 8 предобработок × 4 PSM = 32 попытки
- Голосование: majority vote + средняя уверенность
- Позиционные карты исправлений (буквы/цифры по местам)
- Fallback через JPEG 95

### 4. Universal YOLO Parser
- Поддержка обоих форматов: NMS-ready [N,6] и raw YOLO [4+C, N]
- Автоопределение без конфига

### 5. Access Control Logic
- Нормализация на входе (кириллица→латиница)
- Подтверждения: N раз подряд тот же номер
- Cooldown: защита от повторного открытия
- Сброс счётчика при смене номера или неизвестном

---

## Метрики и качество

### Windows evaluation (barrier/windows/)
- **Detection Coverage**: detected/total
- **Exact OCR Accuracy**: exact_match/total (end-to-end)
- **Detection IoU Success**: iou_match/bbox_records @ 0.5

### Runtime логирование
- ACCESS GRANTED: {plate} — успешное открытие
- Recognized plate: {plate} | YOLO confidence: {conf}
- Access not granted: {plate} — в whitelist, но не прошло подтверждения
- Ошибки с трейсом в data/logs/barrier.log (2MB, 3 бэкапа)

---