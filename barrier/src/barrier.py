import time
from gpiozero import AngularServo


class Barrier:
    def __init__(
        self,
        gpio_pin: int = 18,
        closed_angle: int = 0,
        opened_angle: int = 90,
        open_duration: float = 5.0,
    ):
        self.closed_angle = closed_angle
        self.opened_angle = opened_angle
        self.open_duration = open_duration

        self.servo = AngularServo(
            gpio_pin,
            min_angle=0,
            max_angle=180,
            min_pulse_width=0.0005,
            max_pulse_width=0.0025,
            initial_angle=closed_angle,
        )

        self.is_open = False

    def open(self) -> None:
        if self.is_open:
            return

        self.servo.angle = self.opened_angle
        self.is_open = True

    def close(self) -> None:
        self.servo.angle = self.closed_angle
        self.is_open = False

    def open_for(self, duration: float | None = None) -> None:
        try:
            self.open()
            time.sleep(
                self.open_duration if duration is None else duration
            )
        finally:
            self.close()

    def stop(self) -> None:
        self.close()
        self.servo.close()