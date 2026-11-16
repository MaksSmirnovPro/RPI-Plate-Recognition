
import time
from gpiozero import AngularServo


class Barrier:
    def __init__(
        self,
        gpio_pin=18,
        closed_angle=180,
        opened_angle=90,
        open_duration=5.0,
    ):
        self.closed_angle = closed_angle
        self.opened_angle = opened_angle
        self.open_duration = float(open_duration)

        self.is_open = False
        self.close_at = None

        self.servo = AngularServo(
            gpio_pin,
            min_angle=0,
            max_angle=180,
            min_pulse_width=0.0005,
            max_pulse_width=0.0025,
            initial_angle=self.closed_angle,
        )

        self.close()

    def open(self):
        """Открывает шлагбаум и продлевает время его открытого состояния."""
        self.servo.angle = self.opened_angle
        self.is_open = True
        self.close_at = time.monotonic() + self.open_duration

    def close(self):
        """Закрывает шлагбаум."""
        self.servo.angle = self.closed_angle
        self.is_open = False
        self.close_at = None

    def keep_open(self, duration=None):
        """Продлевает время до автоматического закрытия."""
        duration = self.open_duration if duration is None else float(duration)

        if self.is_open:
            self.close_at = time.monotonic() + duration

    def update(self, vehicle_present=False):
        """
        Вызывается регулярно из основного цикла.

        Если автомобиль присутствует, шлагбаум остаётся открытым.
        Если автомобиль уехал, закрытие происходит после таймера.
        """
        if not self.is_open:
            return

        if vehicle_present:
            self.close_at = time.monotonic() + self.open_duration
            return

        if self.close_at is not None:
            if time.monotonic() >= self.close_at:
                self.close()

    def open_for(self, duration=None):
        """
        Совместимость со старым интерфейсом.
        Не блокирует поток.
        """
        duration = self.open_duration if duration is None else float(duration)
        self.open()
        self.close_at = time.monotonic() + duration

    def stop(self):
        try:
            self.close()
        finally:
            self.servo.close()
