import re
import cv2
import pytesseract
from pytesseract import Output


class PlateOCR:
    def __init__(
        self,
        language: str = "eng",
        min_confidence: float = 35.0,
    ):
        self.language = language
        self.min_confidence = min_confidence

    @staticmethod
    def normalize(text: str) -> str:
        """Оставляет только буквы и цифры, переводит в верхний регистр."""
        return re.sub(r"[^A-ZА-ЯЁ0-9]", "", text.upper())

    def recognize(self, plate_image):
        if plate_image is None or plate_image.size == 0:
            return ""

        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        # Увеличиваем маленькое изображение номера для OCR.
        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        _, threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        config = (
            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        result = pytesseract.image_to_data(
            threshold,
            lang=self.language,
            config=config,
            output_type=Output.DICT,
        )

        accepted = []

        for text, confidence in zip(
            result["text"],
            result["conf"],
        ):
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                continue

            cleaned = self.normalize(text)

            if cleaned and confidence >= self.min_confidence:
                accepted.append(cleaned)

        return "".join(accepted)