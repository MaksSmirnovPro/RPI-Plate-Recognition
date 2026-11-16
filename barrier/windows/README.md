

# Windows 10 — тестирование распознавания автомобильных номеров

## Назначение

Подпроект предназначен для разработки и проверки алгоритма распознавания автомобильных номеров на Windows 10 без Raspberry Pi.

Используются:
- Python и виртуальное окружение `venv`;
- ONNX Runtime для запуска YOLO;
- OpenCV для обработки изображений;
- Tesseract OCR для распознавания символов;
- CSV-файл с эталонными номерами;
- датасет фотографий автомобилей.

Подпроект использует ONNX-модель из основной папки проекта `../models/`.

## 1. Установка программ

Установить:
- Python 3.10 или 3.11;
- Tesseract OCR для Windows.

При установке Tesseract необходимо добавить его в PATH либо указать путь к `tesseract.exe` в `config/settings.yaml`.

Проверка Tesseract:

```powershell
tesseract --version
```

## 2. Создание виртуального окружения

Открыть PowerShell в папке `windows`:

```powershell
cd windows

python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

Если PowerShell запрещает запуск скриптов:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

После этого повторить активацию окружения.

## 3. Установка зависимостей

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

## 4. Подготовка модели

Положить файл модели в основную папку проекта:

```text
barrier/
└── models/
    └── license_plate_yolo11n.onnx
```

Windows-подпроект использует этот же файл. Дублировать модель не требуется.

Путь задаётся в `config/settings.yaml`:

```yaml
model:
  path: ../models/license_plate_yolo11n.onnx
```

## 5. Подготовка датасета

Разместить изображения в:

```text
windows/dataset/images/
```

Пример:

```text
dataset/
├── images/
│   ├── car_001.jpg
│   ├── car_002.jpg
│   └── car_003.jpg
│
└── labels.csv
```

Создать `dataset/labels.csv`:

```csv
image,plate,x1,y1,x2,y2
car_001.jpg,A123BC,120,180,310,230
car_002.jpg,B456CD,85,140,270,190
car_003.jpg,C789EF,,,,
```

Координаты рамки номера необязательны. Если они заполнены, программа дополнительно рассчитает IoU.

## 6. Запуск тестирования

Находясь в папке `windows` с активированным окружением:

```powershell
python -m src.main
```

Запуск с отображением обработанных изображений:

```powershell
python -m src.main --show
```

## 7. Результаты

После выполнения тестирования появится папка:

```text
results/
├── results.csv
└── annotated/
    ├── car_001.jpg
    ├── car_002.jpg
    └── car_003.jpg
```

`results.csv` содержит:
- имя изображения;
- эталонный номер;
- распознанный номер;
- факт обнаружения;
- результат точного сравнения;
- значение IoU, если имеются координаты эталонной рамки;
- итоговый статус.

В `annotated/` сохраняются изображения с рамками детекции и эталонной разметки.

## 8. Метрики

Программа рассчитывает:

**Detection coverage** — доля изображений, на которых детектор обнаружил хотя бы один номер.

**Exact OCR accuracy (end-to-end)** — доля всех изображений, на которых номер обнаружен и полностью совпал с эталонным.

**Detection IoU success** — доля изображений с эталонной рамкой, на которых IoU не ниже заданного порога.

IoU threshold по умолчанию равен 0.5.

Важно: если номер не обнаружен, изображение учитывается как ошибка при расчёте end-to-end accuracy. Это позволяет оценивать весь конвейер, а не только OCR на уже найденных номерах.

## 9. Запуск тестов

```powershell
pytest -v
```

## 10. Настройка модели

В `config/settings.yaml` можно изменять:

```yaml
model:
  input_size: 640
  confidence: 0.45
  nms_threshold: 0.45
  output_format: auto
```

Возможные значения `output_format`:

- `auto` — эвристическое определение формата;
- `yolo` — стандартные сырые выходы YOLO;
- `nms` — выход с координатами, confidence и class_id.

Для достоверных результатов необходимо проверить фактическую структуру выхода используемой ONNX-модели.

## 11. Отличия от Raspberry Pi

| Функция | Raspberry Pi | Windows |
|---|---|---|
| Источник изображения | Camera Module 2 | Датасет |
| Детекция YOLO | ONNX Runtime | ONNX Runtime |
| OCR | Tesseract | Tesseract |
| Управление GPIO | Да | Нет |
| Управление сервоприводом | Да | Нет |
| Проверка номеров | По кадрам | По изображениям |
| Оценка качества | Рабочие логи | CSV и метрики |
| Визуализация | Опционально | OpenCV |

Windows-подпроект не изменяет и не заменяет основную Raspberry Pi-версию.

