
import argparse
from pathlib import Path

import yaml

from src.detector import PlateDetector
from src.ocr import PlateOCR
from src.evaluate import DatasetEvaluator


BASE_DIR = Path(__file__).resolve().parent.parent


def load_config():
    config_path = BASE_DIR / "config" / "settings.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description="License plate dataset evaluation"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated images during processing",
    )

    args = parser.parse_args()
    config = load_config()

    model_path = (BASE_DIR / config["model"]["path"]).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    images_dir = (
        BASE_DIR / config["dataset"]["images_dir"]
    ).resolve()

    labels_file = (
        BASE_DIR / config["dataset"]["labels_file"]
    ).resolve()

    results_dir = BASE_DIR / "results"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset images folder not found: {images_dir}"
        )

    if not labels_file.is_file():
        raise FileNotFoundError(
            f"Dataset labels not found: {labels_file}"
        )

    detector = PlateDetector(
        model_path=model_path,
        input_size=config["model"]["input_size"],
        confidence=config["model"]["confidence"],
        nms_threshold=config["model"]["nms_threshold"],
        output_format=config["model"]["output_format"],
    )

    ocr = PlateOCR(
        language=config["ocr"]["language"],
        min_confidence=config["ocr"]["min_confidence"],
        tesseract_cmd=config["ocr"]["tesseract_cmd"],
    )

    evaluator = DatasetEvaluator(
        detector=detector,
        ocr=ocr,
        images_dir=images_dir,
        labels_file=labels_file,
        results_dir=results_dir,
        save_images=config["evaluation"]["save_images"],
        iou_threshold=config["evaluation"]["iou_threshold"],
    )

    results = evaluator.evaluate()

    if args.show:
        import cv2

        annotated_dir = results_dir / "annotated"

        for result in results:
            image_path = annotated_dir / result["image"]

            if not image_path.exists():
                continue

            image = cv2.imread(str(image_path))

            if image is not None:
                cv2.imshow(result["image"], image)
                cv2.waitKey(0)

        cv2.destroyAllWindows()

    print(f"\nProcessed images: {len(results)}")


if __name__ == "__main__":
    main()
