import time
import signal
import sys
from pathlib import Path

import yaml

from camera import Camera
from detector import PlateDetector
from ocr import PlateOCR
from access_control import AccessControl
from barrier import Barrier
from logger import setup_logger


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "settings.yaml"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    logger = setup_logger()

    camera = None
    barrier = None

    try:
        logger.info("Запуск системы распознавания номеров")

        camera_config = config["camera"]
        detector_config = config["detector"]
        ocr_config = config["ocr"]
        access_config = config["access"]
        barrier_config = config["barrier"]

        model_path = BASE_DIR / detector_config["model_path"]
        allowed_file = BASE_DIR / access_config["allowed_plates_file"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Не найдена ONNX-модель: {model_path}"
            )

        camera = Camera(
            width=camera_config["width"],
            height=camera_config["height"],
        )

        detector = PlateDetector(
            model_path=str(model_path),
            input_size=detector_config["input_size"],
            confidence=detector_config["confidence"],
            nms_threshold=detector_config["nms_threshold"],
        )

        ocr = PlateOCR(
            language=ocr_config["language"],
            min_confidence=ocr_config["min_confidence"],
        )

        access = AccessControl(
            allowed_plates_file=str(allowed_file),
            required_confirmations=access_config["required_confirmations"],
            cooldown_seconds=access_config["cooldown_seconds"],
        )

        barrier = Barrier(
            gpio_pin=barrier_config["gpio_pin"],
            closed_angle=barrier_config["closed_angle"],
            opened_angle=barrier_config["opened_angle"],
            open_duration=barrier_config["open_duration"],
        )

        camera.start()

        logger.info("Система готова к работе")

        interval = float(config["processing"]["interval_seconds"])

        while True:
            cycle_started = time.monotonic()

            frame = camera.read()
            detections = detector.detect(frame)

            if not detections:
                access.confirm_plate("")
                logger.info("Номерные знаки не обнаружены")
            else:
                # Сначала проверяем наиболее уверенную детекцию.
                detections.sort(
                    key=lambda item: item[4],
                    reverse=True,
                )

                recognized_plate = ""

                for x1, y1, x2, y2, confidence in detections:
                    plate_image = frame[y1:y2, x1:x2]
                    plate = ocr.recognize(plate_image)

                    if plate:
                        recognized_plate = plate
                        logger.info(
                            "Распознан номер: %s (детекция %.2f)",
                            plate,
                            confidence,
                        )
                        break

                if recognized_plate:
                    if access.confirm_plate(recognized_plate):
                        logger.info(
                            "Доступ разрешён: %s",
                            recognized_plate,
                        )

                        barrier.open_for()
                        logger.info("Шлагбаум закрыт")
                else:
                    access.confirm_plate("")
                    logger.info("Не удалось распознать номер")

            # Учитываем время обработки, чтобы выдерживать интервал
            # между началами циклов.
            elapsed = time.monotonic() - cycle_started
            time.sleep(max(0.0, interval - elapsed))

    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")

    except Exception:
        logger.exception("Критическая ошибка системы")
        raise

    finally:
        if camera is not None:
            camera.stop()

        if barrier is not None:
            barrier.stop()

        logger.info("Система остановлена")


if __name__ == "__main__":
    main()