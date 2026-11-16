
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

CROPS_DIR = PROJECT_DIR / "results" / "crops"
DATASET_DIR = PROJECT_DIR / "dataset" / "images"

CROP_PATH = CROPS_DIR / "1_0_0.77.jpg"
SOURCE_PATH = DATASET_DIR / "1.jpg"

# Если Tesseract не находится автоматически, укажи путь:
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

WHITELIST = (
    "-c tessedit_char_whitelist="
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def print_image_stats(name, image):
    if image is None:
        print(f"\n{name}: IMAGE NOT FOUND")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"\n--- {name} ---")
    print("Shape:", image.shape)
    print("Min:", int(gray.min()))
    print("Max:", int(gray.max()))
    print("Mean:", round(float(gray.mean()), 2))
    print("Std:", round(float(gray.std()), 2))


def print_components(binary, title):
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    print(f"\n--- {title} ---")
    print("Components:", count - 1)

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        # Пропускаем компонент, занимающий почти всё изображение
        if w > binary.shape[1] * 0.9 and h > binary.shape[0] * 0.9:
            continue

        print(
            f"x={x:3d}, y={y:3d}, "
            f"w={w:3d}, h={h:3d}, area={area:5d}"
        )


def analyze_thresholds(gray):
    print("\n--- GLOBAL THRESHOLDS ---")

    for threshold in [70, 90, 110, 130, 150, 170, 190, 210]:
        _, binary = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY_INV,
        )

        print_components(
            binary,
            f"Threshold {threshold}",
        )


def analyze_adaptive(gray):
    print("\n--- ADAPTIVE THRESHOLD ---")

    scaled = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    for block_size, c in [
        (15, 3),
        (21, 5),
        (31, 7),
        (41, 9),
    ]:
        binary = cv2.adaptiveThreshold(
            scaled,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c,
        )

        print_components(
            binary,
            f"Block={block_size}, C={c}",
        )


def run_tesseract(image, scale, psm, label):
    enlarged = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    prepared = cv2.copyMakeBorder(
        enlarged,
        50,
        50,
        40,
        40,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    config = (
        f"--oem 3 --psm {psm} "
        f"{WHITELIST}"
    )

    text = pytesseract.image_to_string(
        prepared,
        config=config,
        lang="eng",
    )

    print(
        f"{label}, Scale={scale}, PSM={psm}: "
        f"{text.strip()!r}"
    )


def analyze_tesseract(gray):
    print("\n--- TESSERACT ON ADAPTIVE IMAGES ---")

    scaled = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    for block_size, c in [
        (15, 3),
        (21, 5),
        (31, 7),
        (41, 9),
    ]:
        binary = cv2.adaptiveThreshold(
            scaled,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

        prepared = cv2.copyMakeBorder(
            binary,
            40,
            40,
            40,
            40,
            cv2.BORDER_CONSTANT,
            value=255,
        )

        for psm in [7, 8, 13]:
            config = (
                f"--oem 3 --psm {psm} "
                f"{WHITELIST}"
            )

            text = pytesseract.image_to_string(
                prepared,
                config=config,
                lang="eng",
            )

            print(
                f"Block={block_size}, C={c}, "
                f"PSM={psm}: {text.strip()!r}"
            )


def analyze_grayscale_ocr(gray):
    print("\n--- GRAYSCALE OCR ---")

    for scale in [2, 3, 4, 5, 6]:
        for psm in [7, 8, 13]:
            run_tesseract(
                gray,
                scale,
                psm,
                "Grayscale",
            )


def analyze_illumination(gray):
    print("\n--- ILLUMINATION CORRECTION OCR ---")

    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=15,
        sigmaY=15,
    )

    corrected = cv2.divide(
        gray,
        background,
        scale=150,
    )

    for scale in [3, 4, 5]:
        for psm in [7, 8, 13]:
            run_tesseract(
                corrected,
                scale,
                psm,
                "Illumination",
            )


def analyze_profiles(gray):
    print("\n--- IMAGE PROFILE ---")

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    row_mean = blurred.mean(axis=1)
    col_mean = blurred.mean(axis=0)

    print("Image shape:", blurred.shape)

    print("\nRow brightness:")
    for y, value in enumerate(row_mean):
        print(f"y={y:02d}: {value:.1f}")

    print("\nColumn brightness:")
    for x in range(0, len(col_mean), 4):
        values = col_mean[x:x + 4]

        print(
            f"x={x:03d}-{x + len(values) - 1:03d}: "
            f"{values.mean():.1f}"
        )

    print("\nThreshold bounding boxes:")

    for threshold in [70, 90, 110, 130, 150, 170]:
        mask = blurred < threshold
        ys, xs = np.where(mask)

        if len(xs) == 0:
            print(f"Threshold {threshold}: empty")
            continue

        print(
            f"Threshold {threshold}: "
            f"x={xs.min()}..{xs.max()}, "
            f"y={ys.min()}..{ys.max()}, "
            f"width={xs.max() - xs.min() + 1}, "
            f"height={ys.max() - ys.min() + 1}"
        )


def main():
    print("PROJECT DIR:", PROJECT_DIR)
    print("CROP PATH:", CROP_PATH)
    print("SOURCE PATH:", SOURCE_PATH)

    source = cv2.imread(str(SOURCE_PATH))
    crop = cv2.imread(str(CROP_PATH))

    print_image_stats("SOURCE IMAGE", source)
    print_image_stats("CROP IMAGE", crop)

    if crop is None:
        print("\nERROR: Crop image not found.")
        return

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    print("\nCrop dimensions:", crop.shape)

    analyze_thresholds(gray)
    analyze_adaptive(gray)
    analyze_profiles(gray)

    analyze_tesseract(gray)
    analyze_grayscale_ocr(gray)
    analyze_illumination(gray)

    print("\n--- SOURCE IMAGE VS CROP ---")

    if source is not None:
        print("Source dimensions:", source.shape)
        print("Crop dimensions:", crop.shape)

        source_gray = cv2.cvtColor(
            source,
            cv2.COLOR_BGR2GRAY,
        )

        crop_gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        print(
            "Source grayscale mean:",
            round(float(source_gray.mean()), 2),
        )

        print(
            "Crop grayscale mean:",
            round(float(crop_gray.mean()), 2),
        )

    else:
        print("Source image not found.")

    print("\n--- DETECTION BBOX ---")
    print(
        "Coordinates are not available in the crop filename. "
        "We need to obtain them from the detector output."
    )


if __name__ == "__main__":
    main()
