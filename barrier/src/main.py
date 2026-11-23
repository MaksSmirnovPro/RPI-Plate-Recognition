
import threading
import time
from pathlib import Path

import cv2
import yaml

from src.camera import Camera
from src.detector import PlateDetector
from src.ocr import PlateOCR
from src.access_control import AccessControl
from src.barrier import Barrier
from src.logger import setup_logger


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "settings.yaml"


class LicensePlateSystem:

    def __init__(self):
        self.logger = setup_logger()
        self.config = self.load_config()

        # Display is optional: keep False when using opencv-python-headless.
        display_cfg = self.config.get("display", {})
        self.display_enabled = bool(display_cfg.get("enabled", False))

        # Camera
        camera_cfg = self.config["camera"]

        self.camera = Camera(
            width=camera_cfg.get("width", 640),
            height=camera_cfg.get("height", 480),
        )

        # Detector
        detector_cfg = self.config["detector"]

        model_path = PROJECT_DIR / detector_cfg["model_path"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}"
            )

        self.detector = PlateDetector(
            model_path=str(model_path),
            input_size=detector_cfg.get("input_size", 640),
            confidence=detector_cfg.get("confidence", 0.45),
            nms_threshold=detector_cfg.get(
                "nms_threshold", 0.45
            ),
        )

        # OCR
        ocr_cfg = self.config["ocr"]

        self.ocr = PlateOCR(
            language=ocr_cfg.get("language", "eng"),
            min_confidence=ocr_cfg.get("min_confidence", 0),
        )

        # Access control
        access_cfg = self.config["access"]

        allowed_file = (
            PROJECT_DIR / access_cfg["allowed_plates_file"]
        )

        if not allowed_file.exists():
            raise FileNotFoundError(
                f"Allowed plates file not found: {allowed_file}"
            )

        self.access = AccessControl(
            allowed_plates_file=str(allowed_file),
            required_confirmations=access_cfg.get(
                "required_confirmations", 3
            ),
            cooldown_seconds=access_cfg.get(
                "cooldown_seconds", 15
            ),
        )

        # Barrier
        barrier_cfg = self.config["barrier"]

        self.barrier = Barrier(
            pwm_channel=barrier_cfg.get("pwm_channel", 1),
            pwm_chip=barrier_cfg.get("pwm_chip", 0),
            frequency_hz=barrier_cfg.get("frequency_hz", 50),
            min_pulse_width_ms=barrier_cfg.get("min_pulse_width_ms", 1.0),
            max_pulse_width_ms=barrier_cfg.get("max_pulse_width_ms", 2.0),
            closed_angle=barrier_cfg.get("closed_angle", 180),
            opened_angle=barrier_cfg.get("opened_angle", 90),
            open_duration=barrier_cfg.get("open_duration", 10),
        )

        # Processing
        processing_cfg = self.config.get("processing", {})

        self.processing_interval = float(
            processing_cfg.get("interval_seconds", 1.5)
        )

        self.result_display_seconds = 3.0

        # Shared state
        self.frame_lock = threading.Lock()

        self.latest_frame = None
        self.result_frame = None
        self.result_until = 0.0

        self.stop_event = threading.Event()
        self.worker = None

    @staticmethod
    def load_config():
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Configuration not found: {CONFIG_PATH}"
            )

        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def recognition_worker(self):
        """
        Фоновый поток:
        detection -> OCR -> access control -> barrier.

        Управление шлагбаумом не блокирует распознавание.
        """

        self.logger.info("Recognition worker started")

        while not self.stop_event.is_set():

            with self.frame_lock:
                if self.latest_frame is None:
                    frame = None
                else:
                    frame = self.latest_frame.copy()

            if frame is None:
                self.stop_event.wait(0.05)
                continue

            try:
                # YOLO detection
                detections = self.detector.detect(frame)

                # Сортируем по confidence
                detections = sorted(
                    detections,
                    key=lambda item: item[4],
                    reverse=True,
                )

                # Наличие обнаруженной номерной рамки.
                # Пока рамка видна, продлеваем открытое состояние.
                vehicle_present = len(detections) > 0

                self.barrier.update(
                    vehicle_present=vehicle_present
                )

                recognized = False

                height, width = frame.shape[:2]

                for x1, y1, x2, y2, confidence in detections:

                    x1, y1, x2, y2 = map(
                        int, (x1, y1, x2, y2)
                    )

                    x1 = max(0, min(x1, width))
                    x2 = max(0, min(x2, width))
                    y1 = max(0, min(y1, height))
                    y2 = max(0, min(y2, height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    plate = self.ocr.recognize(crop)

                    if not plate:
                        continue

                    recognized = True

                    self.logger.info(
                        "Recognized plate: %s | YOLO confidence: %.3f",
                        plate,
                        confidence,
                    )

                    # Отображение результата
                    display_frame = frame.copy()

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    text_y = max(y1 - 12, 30)

                    cv2.putText(
                        display_frame,
                        plate,
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    with self.frame_lock:
                        self.result_frame = display_frame
                        self.result_until = (
                            time.monotonic()
                            + self.result_display_seconds
                        )

                    # Проверка доступа
                    if self.access.confirm_plate(plate):

                        self.logger.info(
                            "ACCESS GRANTED: %s",
                            plate,
                        )

                        # ВАЖНО:
                        # open_for() теперь неблокирующий.
                        # Поток распознавания продолжит работу.
                        self.barrier.open_for()

                    else:
                        self.logger.info(
                            "Access not granted: %s",
                            plate,
                        )

                    # Обрабатываем один номер за цикл
                    break

                if not recognized:
                    self.access.confirm_plate("")

            except Exception:
                self.logger.exception(
                    "Recognition worker error"
                )

            self.stop_event.wait(
                self.processing_interval
            )

        self.logger.info("Recognition worker stopped")

    def run(self):

        self.logger.info(
            "Starting license plate recognition system"
        )

        try:
            self.camera.start()

            self.barrier.close()

            self.worker = threading.Thread(
                target=self.recognition_worker,
                name="RecognitionWorker",
                daemon=True,
            )

            self.worker.start()

            window_name = "License Plate Recognition"

            if self.display_enabled:
                cv2.namedWindow(
                    window_name,
                    cv2.WINDOW_NORMAL,
                )
                self.logger.info(
                    "Video window started. Press Q to exit."
                )
            else:
                self.logger.info(
                    "Display disabled; running in headless mode. Press Ctrl+C to exit."
                )

            while not self.stop_event.is_set():

                # Камера продолжает работать независимо
                # от процесса распознавания.
                frame = self.camera.read()

                if frame is None:
                    self.stop_event.wait(0.01)
                    continue

                with self.frame_lock:
                    self.latest_frame = frame.copy()

                    if self.display_enabled:
                        if (
                            self.result_frame is not None
                            and time.monotonic() < self.result_until
                        ):
                            display_frame = self.result_frame.copy()
                        else:
                            self.result_frame = None
                            display_frame = frame.copy()

                if self.display_enabled:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        self.logger.info(
                            "Exit requested by user"
                        )
                        break

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt")

        except Exception:
            self.logger.exception(
                "Fatal application error"
            )

        finally:
            self.stop_event.set()

            if self.worker is not None:
                self.worker.join(timeout=5.0)

            if self.display_enabled:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    self.logger.warning(
                        "OpenCV windows could not be closed"
                    )

            try:
                self.camera.stop()
            except Exception:
                self.logger.exception(
                    "Camera shutdown error"
                )

            try:
                self.barrier.stop()
            except Exception:
                self.logger.exception(
                    "Barrier shutdown error"
                )

            self.logger.info(
                "License plate recognition system stopped"
            )


def main():
    system = LicensePlateSystem()
    system.run()


if __name__ == "__main__":
    main()
