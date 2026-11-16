
import re
import time
from pathlib import Path


class AccessControl:
    # Кириллические буквы, визуально совпадающие
    # с латинскими буквами на российских номерных знаках.
    CYRILLIC_TO_LATIN = {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }

    def __init__(
        self,
        allowed_plates_file: str,
        required_confirmations: int = 2,
        cooldown_seconds: float = 10.0,
    ):
        self.allowed_file = Path(allowed_plates_file)
        self.required_confirmations = max(1, int(required_confirmations))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))

        self.allowed_plates = self._load_allowed_plates()

        self.last_plate = None
        self.confirmation_count = 0
        self.last_open_time = 0.0

        print("Allowed plates file:", self.allowed_file.resolve())
        print("Allowed plates:", sorted(self.allowed_plates))

    @classmethod
    def normalize(cls, plate: str) -> str:
        """Приводит номер к единому формату."""
        if not plate:
            return ""

        plate = str(plate).upper().strip()

        plate = "".join(
            cls.CYRILLIC_TO_LATIN.get(char, char)
            for char in plate
        )

        return re.sub(r"[^A-Z0-9]", "", plate)

    def _load_allowed_plates(self) -> set:
        """Загружает разрешённые номера из файла."""
        self.allowed_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.allowed_file.exists():
            self.allowed_file.touch()

        plates = set()

        with self.allowed_file.open(
            "r",
            encoding="utf-8-sig",
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
        """Проверяет наличие номера в whitelist."""
        normalized = self.normalize(plate)

        if not normalized:
            return False

        return normalized in self.allowed_plates

    def confirm_plate(self, plate: str) -> bool:
        """
        Возвращает True, если номер разрешён,
        распознан необходимое количество раз
        и cooldown после предыдущего открытия истёк.
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

        print(
            f"Plate confirmation: {normalized} "
            f"({self.confirmation_count}/{self.required_confirmations})"
        )

        if self.confirmation_count < self.required_confirmations:
            return False

        now = time.monotonic()

        if now - self.last_open_time < self.cooldown_seconds:
            return False

        self.last_open_time = now
        self.confirmation_count = 0
        self.last_plate = None

        return True
