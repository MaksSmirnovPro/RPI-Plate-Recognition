import re
import time
from pathlib import Path


class AccessControl:
    def __init__(
        self,
        allowed_plates_file: str,
        required_confirmations: int = 2,
        cooldown_seconds: float = 10.0,
    ):
        self.allowed_file = Path(allowed_plates_file)
        self.required_confirmations = required_confirmations
        self.cooldown_seconds = cooldown_seconds

        self.allowed_plates = self._load_allowed_plates()

        self.last_plate = None
        self.confirmation_count = 0
        self.last_open_time = 0.0

    @staticmethod
    def normalize(plate: str) -> str:
        return re.sub(r"[^A-ZА-ЯЁ0-9]", "", plate.upper())

    def _load_allowed_plates(self) -> set:
        self.allowed_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.allowed_file.exists():
            self.allowed_file.touch()

        plates = set()

        with self.allowed_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                normalized = self.normalize(line)

                if normalized:
                    plates.add(normalized)

        return plates

    def is_allowed(self, plate: str) -> bool:
        normalized = self.normalize(plate)

        if not normalized:
            return False

        return normalized in self.allowed_plates

    def confirm_plate(self, plate: str) -> bool:
        """
        Возвращает True, если номер разрешён и подтверждён
        необходимое количество раз.
        """
        normalized = self.normalize(plate)

        if not normalized or not self.is_allowed(normalized):
            self.last_plate = None
            self.confirmation_count = 0
            return False

        if normalized == self.last_plate:
            self.confirmation_count += 1
        else:
            self.last_plate = normalized
            self.confirmation_count = 1

        if self.confirmation_count < self.required_confirmations:
            return False

        now = time.monotonic()

        if now - self.last_open_time < self.cooldown_seconds:
            return False

        self.last_open_time = now
        self.confirmation_count = 0
        self.last_plate = None

        return True