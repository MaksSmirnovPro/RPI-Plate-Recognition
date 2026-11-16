
import argparse
import contextlib
import csv
import io
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CROPS_DIR = PROJECT_DIR / "results" / "crops"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "rectification"

from src.ocr import PlateOCR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def smooth_positions(values, kernel_size=9):
    """Сглаживает найденные координаты вдоль горизонтальной оси."""
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    values = np.asarray(values, dtype=np.float32).reshape(1, -1)
    return cv2.GaussianBlur(
        values,
        (kernel_size, 1),
        0,
    ).reshape(-1)


def fit_quadratic_curve(points, width):
    """
    Аппроксимирует координаты y(x) параболой.
    Координата x нормализуется для устойчивости вычислений.
    """
    x = np.linspace(-1.0, 1.0, width)
    coefficients = np.polyfit(x, points, 2)
    curve = np.polyval(coefficients, x)

    return curve, coefficients


def detect_plate_boundaries(image):
    """
    Ищет верхнюю и нижнюю горизонтальные границы
    по вертикальному градиенту яркости.

    Важно: это диагностический поиск границ, а не гарантированное
    распознавание именно кромки рамки.
    """
    height, width = image.shape[:2]

    if height < 12 or width < 20:
        raise ValueError(
            f"Crop слишком маленький: {width}x{height}"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gradient = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )
    gradient = np.abs(gradient)

    # Ищем верхнюю границу в верхней части crop.
    top_start = max(1, int(height * 0.05))
    top_end = max(top_start + 1, int(height * 0.45))

    # Нижнюю границу ищем в нижней части crop.
    bottom_start = min(height - 2, int(height * 0.55))
    bottom_end = height - 1

    top_region = gradient[top_start:top_end, :]
    bottom_region = gradient[bottom_start:bottom_end, :]

    top_points = (
        np.argmax(top_region, axis=0).astype(np.float32)
        + top_start
    )

    bottom_points = (
        np.argmax(bottom_region, axis=0).astype(np.float32)
        + bottom_start
    )

    # Сглаживаем отдельные скачки, вызванные символами и шумом.
    kernel_size = max(3, (width // 8) | 1)

    top_points = smooth_positions(top_points, kernel_size)
    bottom_points = smooth_positions(bottom_points, kernel_size)

    top_curve, top_coefficients = fit_quadratic_curve(
        top_points,
        width,
    )
    bottom_curve, bottom_coefficients = fit_quadratic_curve(
        bottom_points,
        width,
    )

    # Ограничиваем кривые размерами исходного crop.
    top_curve = np.clip(top_curve, 0, height - 1)
    bottom_curve = np.clip(bottom_curve, 0, height - 1)

    # Проверяем, что между кривыми остаётся достаточная высота.
    gap = bottom_curve - top_curve

    if np.min(gap) < height * 0.30:
        raise ValueError(
            "Найденные границы пересекаются или расположены "
            "слишком близко. Автоматическая коррекция отменена."
        )

    return {
        "top_points": top_points,
        "bottom_points": bottom_points,
        "top_curve": top_curve,
        "bottom_curve": bottom_curve,
        "top_coefficients": top_coefficients,
        "bottom_coefficients": bottom_coefficients,
    }


def rectify_image(image, boundaries):
    """
    Выпрямляет изображение с помощью cv2.remap().

    Верхняя и нижняя найденные границы переводятся
    в горизонтальные линии.
    """
    height, width = image.shape[:2]

    top = boundaries["top_curve"]
    bottom = boundaries["bottom_curve"]

    map_x = np.tile(
        np.arange(width, dtype=np.float32),
        (height, 1),
    )

    vertical_position = np.linspace(
        0.0,
        1.0,
        height,
        dtype=np.float32,
    ).reshape(-1, 1)

    map_y = (
        top.reshape(1, -1)
        + vertical_position
        * (bottom - top).reshape(1, -1)
    )

    rectified = cv2.remap(
        image,
        map_x,
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rectified


def calculate_sagitta(curve):
    """Оценивает изгиб параболы относительно линии между краями."""
    width = len(curve)

    middle = float(curve[width // 2])
    edge_average = float((curve[0] + curve[-1]) / 2.0)

    return middle - edge_average


def create_diagnostic_image(image, boundaries, scale=4):
    """Рисует найденные точки и аппроксимированные границы."""
    result = image.copy()
    height, width = image.shape[:2]

    for x in range(width):
        top_y = int(round(boundaries["top_curve"][x]))
        bottom_y = int(round(boundaries["bottom_curve"][x]))

        cv2.circle(result, (x, top_y), 1, (0, 255, 0), -1)
        cv2.circle(result, (x, bottom_y), 1, (255, 0, 0), -1)

    # Исходные найденные точки показываем отдельно.
    for x in range(0, width, max(1, width // 20)):
        top_y = int(round(boundaries["top_points"][x]))
        bottom_y = int(round(boundaries["bottom_points"][x]))

        cv2.circle(result, (x, top_y), 1, (0, 255, 255), -1)
        cv2.circle(result, (x, bottom_y), 1, (255, 0, 255), -1)

    return cv2.resize(
        result,
        (width * scale, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )


def run_ocr(ocr, image):
    """
    Запускает существующий OCR без изменения его алгоритма.
    Перехватывает диагностический вывод, чтобы не засорять консоль.
    """
    output = io.StringIO()

    try:
        with contextlib.redirect_stdout(output):
            result = ocr.recognize(image)

        return repr(result)

    except Exception as error:
        return f"OCR_ERROR: {type(error).__name__}: {error}"


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Не удалось сохранить изображение: {path}")


def process_image(image_path, output_dir, ocr):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[SKIP] Не удалось прочитать: {image_path.name}")
        return None

    print(f"\n{'=' * 60}")
    print(f"IMAGE: {image_path.name}")
    print(f"SIZE: {image.shape[1]}x{image.shape[0]}")

    original_ocr = run_ocr(ocr, image)

    image_output_dir = output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    save_image(
        image_output_dir / "original.png",
        cv2.resize(
            image,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        ),
    )

    try:
        boundaries = detect_plate_boundaries(image)
        rectified = rectify_image(image, boundaries)

    except Exception as error:
        print(f"[RECTIFICATION FAILED] {error}")

        return {
            "image": image_path.name,
            "width": image.shape[1],
            "height": image.shape[0],
            "top_sagitta": "",
            "bottom_sagitta": "",
            "original_ocr": original_ocr,
            "rectified_ocr": "",
            "status": f"FAILED: {error}",
        }

    rectified_ocr = run_ocr(ocr, rectified)

    diagnostic = create_diagnostic_image(
        image,
        boundaries,
    )

    save_image(
        image_output_dir / "boundaries.png",
        diagnostic,
    )

    save_image(
        image_output_dir / "rectified.png",
        cv2.resize(
            rectified,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        ),
    )

    top_sagitta = calculate_sagitta(
        boundaries["top_curve"]
    )
    bottom_sagitta = calculate_sagitta(
        boundaries["bottom_curve"]
    )

    print(
        "Top coefficients:",
        np.round(boundaries["top_coefficients"], 4),
    )
    print(
        "Bottom coefficients:",
        np.round(boundaries["bottom_coefficients"], 4),
    )
    print(f"Top sagitta:    {top_sagitta:.2f} px")
    print(f"Bottom sagitta: {bottom_sagitta:.2f} px")
    print(f"Original OCR:   {original_ocr}")
    print(f"Rectified OCR:  {rectified_ocr}")

    return {
        "image": image_path.name,
        "width": image.shape[1],
        "height": image.shape[0],
        "top_sagitta": round(top_sagitta, 3),
        "bottom_sagitta": round(bottom_sagitta, 3),
        "original_ocr": original_ocr,
        "rectified_ocr": rectified_ocr,
        "status": "OK",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=DEFAULT_CROPS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    crops_dir = args.crops_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not crops_dir.exists():
        raise FileNotFoundError(
            f"Каталог crop не найден: {crops_dir}"
        )

    image_paths = sorted(
        path
        for path in crops_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"В каталоге нет изображений: {crops_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    ocr = PlateOCR(min_confidence=0)

    results = []

    print(f"Crop directory: {crops_dir}")
    print(f"Images found: {len(image_paths)}")
    print(f"Output directory: {output_dir}")

    for image_path in image_paths:
        result = process_image(
            image_path,
            output_dir,
            ocr,
        )

        if result is not None:
            results.append(result)

    csv_path = output_dir / "comparison.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "width",
                "height",
                "top_sagitta",
                "bottom_sagitta",
                "original_ocr",
                "rectified_ocr",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 60}")
    print(f"Processed: {len(results)}")
    print(f"CSV: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
