
import re

import cv2
import numpy as np
import pytesseract


PLATE_LETTERS = "ABEKMHOPCTYX"

PLATE_PATTERN = re.compile(
    rf"[{PLATE_LETTERS}]\d{{3}}[{PLATE_LETTERS}]{{2}}"
)

CYRILLIC_TO_LATIN = str.maketrans({
    "А": "A", "В": "B", "Е": "E",
    "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C",
    "Т": "T", "У": "Y", "Х": "X",
})


def normalize_plate(text):
    text = text.upper().translate(CYRILLIC_TO_LATIN)
    return re.sub(r"[^A-Z0-9]", "", text)


def extract_plate(text):
    """Извлекает номер российского формата из текста OCR."""

    text = normalize_plate(text)

    if len(text) < 6:
        return ""

    digit_map = str.maketrans({
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    })

    letter_map = str.maketrans({
        "0": "O",
        "8": "B",
        "4": "A",
    })

    for start in range(len(text) - 5):
        candidate = list(text[start:start + 6])

        candidate[0] = candidate[0].translate(letter_map)

        for i in range(1, 4):
            candidate[i] = candidate[i].translate(digit_map)

        for i in range(4, 6):
            candidate[i] = candidate[i].translate(letter_map)

        plate = "".join(candidate)

        if PLATE_PATTERN.fullmatch(plate):
            return plate

    return ""


class PlateOCR:

    def __init__(
        self,
        language="eng",
        min_confidence=0,
        tesseract_cmd="",
    ):
        self.language = language
        self.min_confidence = min_confidence

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    @staticmethod
    def add_border(image):
        return cv2.copyMakeBorder(
            image,
            30, 30, 40, 40,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    def preprocess(self, image):
        if image is None or image.size == 0:
            return []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        enlarged = cv2.resize(
            gray,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        enhanced = clahe.apply(enlarged)

        _, otsu = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        _, otsu_original = cv2.threshold(
            enlarged,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        adaptive_21 = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            7,
        )

        adaptive_31 = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        blurred = cv2.GaussianBlur(
            enlarged,
            (0, 0),
            1.0,
        )

        sharpened = cv2.addWeighted(
            enlarged,
            1.8,
            blurred,
            -0.8,
            0,
        )

        variants = [
            ("enlarged", enlarged),
            ("enhanced", enhanced),
            ("otsu_original", otsu_original),
            ("otsu", otsu),
            ("adaptive_21", adaptive_21),
            ("adaptive_31", adaptive_31),
            ("sharpened", sharpened),
            ("otsu_inverted", cv2.bitwise_not(otsu)),
        ]

        return [
            (name, self.add_border(variant))
            for name, variant in variants
        ]

    def recognize_variant(self, image, psm):
        whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        config = (
            f"--oem 3 --psm {psm} "
            f"-c tessedit_char_whitelist={whitelist}"
        )

        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        words = []
        confidences = []

        for word, confidence in zip(
            data["text"],
            data["conf"],
        ):
            word = word.strip()

            if not word:
                continue

            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                continue

            if confidence < 0:
                continue

            words.append(word)
            confidences.append(confidence)

        raw_text = "".join(words)

        if not confidences:
            return raw_text, "", 0.0

        confidence = sum(confidences) / len(confidences)
        plate = extract_plate(raw_text)

        if confidence < self.min_confidence:
            plate = ""

        return raw_text, plate, confidence

    def recognize(self, image):
        if image is None or image.size == 0:
            return ""

        print(
            f"\nOCR input size: "
            f"{image.shape[1]}x{image.shape[0]}"
        )

        def recognize_variants(source_image, source_name):
            variants = self.preprocess(source_image)

            if not variants:
                return ""

            psm_modes = [7, 8, 13, 6]
            candidates = []

            print(f"\n  Source: {source_name}")

            for vi, (variant_name, variant) in enumerate(variants):
                for psm in psm_modes:
                    raw, plate, confidence = self.recognize_variant(
                        variant,
                        psm,
                    )

                    print(
                        f"    Variant {vi}: {variant_name:15} "
                        f"PSM={psm:2} | "
                        f"Raw={raw!r:15} | "
                        f"Plate={plate or '---':6} | "
                        f"Conf={confidence:.1f}"
                    )

                    if plate:
                        candidates.append({
                            "plate": plate,
                            "confidence": confidence,
                            "variant": vi,
                            "psm": psm,
                        })

            if not candidates:
                print("  No valid plate candidates.")
                return ""

            counts = {}

            for candidate in candidates:
                plate = candidate["plate"]

                if plate not in counts:
                    counts[plate] = {
                        "count": 0,
                        "confidence": [],
                    }

                counts[plate]["count"] += 1
                counts[plate]["confidence"].append(
                    candidate["confidence"]
                )

            best_plate = max(
                counts,
                key=lambda plate: (
                    counts[plate]["count"],
                    sum(counts[plate]["confidence"])
                    / len(counts[plate]["confidence"]),
                ),
            )

            print(f"  Selected result: {best_plate}")

            return best_plate

        result = recognize_variants(image, "original")

        if result:
            return result

        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )

        if not success:
            return ""

        jpeg_image = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        if jpeg_image is None:
            return ""

        return recognize_variants(jpeg_image, "jpeg95")
