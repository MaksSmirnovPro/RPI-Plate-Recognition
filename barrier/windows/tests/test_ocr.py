
import sys
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.ocr import PlateOCR


EXPECTED_PLATES = {
    "1": "C007CY",
    "2": "M777PO",
    "3": "C080TT",
    "4": "O908OO",
}

PSM_MODES = [7, 8, 13, 6]


def diagnose_image(ocr, image, title):
    print(f"\n--- {title} ---")

    variants = ocr.preprocess(image)
    results = []

    for variant_index, variant in enumerate(variants):
        for psm in PSM_MODES:
            raw, plate, confidence = ocr.recognize_variant(
                variant,
                psm,
            )

            print(
                f"Variant={variant_index} | "
                f"PSM={psm} | "
                f"Raw={raw!r} | "
                f"Plate={plate or '---'} | "
                f"Confidence={confidence:.1f}"
            )

            if plate:
                results.append({
                    "plate": plate,
                    "confidence": confidence,
                    "variant": variant_index,
                    "psm": psm,
                })

    counts = {}

    for result in results:
        plate = result["plate"]

        if plate not in counts:
            counts[plate] = {
                "count": 0,
                "confidence": [],
            }

        counts[plate]["count"] += 1
        counts[plate]["confidence"].append(
            result["confidence"]
        )

    if not counts:
        return ""

    best_plate = max(
        counts,
        key=lambda plate: (
            counts[plate]["count"],
            sum(counts[plate]["confidence"])
            / len(counts[plate]["confidence"]),
        ),
    )

    print(f"BEST: {best_plate}")
    return best_plate


def main():
    crops_dir = PROJECT_DIR / "results" / "crops"
    ocr = PlateOCR(min_confidence=0)

    for image_path in sorted(crops_dir.glob("*.jpg")):
        print("\n" + "=" * 65)
        print("IMAGE:", image_path.name)
        print("=" * 65)

        image = cv2.imread(str(image_path))

        if image is None:
            print("Cannot read image")
            continue

        image_id = image_path.stem.split("_")[0]
        expected = EXPECTED_PLATES.get(image_id, "UNKNOWN")

        print("Size:", image.shape)
        print("Expected:", expected)

        # 1. Исходный crop.
        original_result = diagnose_image(
            ocr,
            image,
            "ORIGINAL CROP",
        )

        # 2. JPEG-копия качества 95.
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )

        jpeg_result = ""

        if success:
            jpeg_image = cv2.imdecode(
                encoded,
                cv2.IMREAD_COLOR,
            )

            if jpeg_image is not None:
                jpeg_result = diagnose_image(
                    ocr,
                    jpeg_image,
                    "JPEG QUALITY 95",
                )

        print("\nFINAL COMPARISON")
        print("Expected:", expected)
        print("Original:", original_result or "---")
        print("JPEG 95: ", jpeg_result or "---")
        print(
            "Original correct:",
            original_result == expected,
        )
        print(
            "JPEG correct:",
            jpeg_result == expected,
        )


if __name__ == "__main__":
    main()
