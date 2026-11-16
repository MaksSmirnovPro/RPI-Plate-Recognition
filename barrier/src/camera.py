import time
import cv2
from picamera2 import Picamera2


class Camera:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.camera = None

    def start(self) -> None:
        self.camera = Picamera2()

        config = self.camera.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888",
            },
            buffer_count=2,
        )

        self.camera.configure(config)
        self.camera.start()

        # Даём камере время на автоматическую экспозицию.
        time.sleep(1.0)

    def read(self):
        if self.camera is None:
            raise RuntimeError("Камера не запущена")

        rgb_frame = self.camera.capture_array()
        return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    def stop(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None