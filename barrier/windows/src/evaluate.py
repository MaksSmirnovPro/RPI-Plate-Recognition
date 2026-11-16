
import csv
from pathlib import Path

import cv2

from src.ocr import normalize_plate


def calculate_iou(box_a, box_b):
    """Intersection over Union для двух рамок x1,y1,x2,y2."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


class DatasetEvaluator:
    def __init__(
        self,
        detector,
        ocr,
        images_dir,
        labels_file,
        results_dir,
        save_images=True,
        iou_threshold=0.5,
    ):
        self.detector = detector
        self.ocr = ocr

        self.images_dir = Path(images_dir)
        self.labels_file = Path(labels_file)
        self.results_dir = Path(results_dir)

        self.save_images = save_images
        self.iou_threshold = iou_threshold

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.save_images:
            self.annotated_dir = self.results_dir / "annotated"
            self.annotated_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

    def load_labels(self):
        records = []

        with self.labels_file.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            required = {"image", "plate"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    "labels.csv должен содержать столбцы image и plate"
                )

            for row in reader:
                image_name = row["image"].strip()
                plate = normalize_plate(row["plate"])

                if not image_name or not plate:
                    continue

                bbox = None

                coordinate_names = ["x1", "y1", "x2", "y2"]
                values = [row.get(name, "").strip()
                          for name in coordinate_names]

                if all(values):
                    bbox = tuple(int(float(v)) for v in values)

                records.append({
                    "image": image_name,
                    "plate": plate,
                    "bbox": bbox,
                })

        return records

    def evaluate(self):
        records = self.load_labels()
        results = []

        for position, record in enumerate(records, start=1):
            image_path = self.images_dir / record["image"]

            print(
                f"[{position}/{len(records)}] "
                f"Processing: {record['image']}"
            )

            if not image_path.is_file():
                print("  ERROR: image not found")
                results.append({
                    "image": record["image"],
                    "expected": record["plate"],
                    "predicted": "",
                    "detected": False,
                    "exact_match": False,
                    "iou": "",
                    "detection_match": False,
                    "status": "IMAGE_NOT_FOUND",
                })
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print("  ERROR: cannot read image")
                results.append({
                    "image": record["image"],
                    "expected": record["plate"],
                    "predicted": "",
                    "detected": False,
                    "exact_match": False,
                    "iou": "",
                    "detection_match": False,
                    "status": "IMAGE_READ_ERROR",
                })
                continue

            detections = self.detector.detect(image)
            
            # Сохраняем все найденные области номерных знаков.
            if detections:
                crops_dir = self.results_dir / "crops"
                crops_dir.mkdir(parents=True, exist_ok=True)

                for index, (x1, y1, x2, y2, confidence) in enumerate(detections):
                    crop = image[y1:y2, x1:x2]

                    if crop.size > 0:
                        crop_path = (
                            crops_dir /
                            f"{image_path.stem}_{index}_{confidence:.2f}.jpg"
                        )
                        cv2.imwrite(str(crop_path), crop)

            best_plate = ""
            best_confidence = 0.0
            best_box = None

            for x1, y1, x2, y2, confidence in detections:
                crop = image[y1:y2, x1:x2]
                recognized = self.ocr.recognize(crop)

                if recognized and confidence > best_confidence:
                    best_plate = recognized
                    best_confidence = confidence
                    best_box = (x1, y1, x2, y2)

            expected = record["plate"]
            exact_match = best_plate == expected

            iou = ""
            detection_match = False

            if record["bbox"] is not None and best_box is not None:
                iou = calculate_iou(record["bbox"], best_box)
                detection_match = iou >= self.iou_threshold

            status = (
                "OK" if exact_match
                else "OCR_MISMATCH" if best_plate
                else "OCR_FAILED" if detections
                else "NOT_DETECTED"
            )

            print(
                f"  Expected: {expected} | "
                f"Predicted: {best_plate or '---'} | "
                f"Status: {status}"
            )

            if iou != "":
                print(
                    f"  IoU: {iou:.3f} | "
                    f"Detection match: {detection_match}"
                )

            if self.save_images:
                annotated = image.copy()

                if record["bbox"] is not None:
                    gx1, gy1, gx2, gy2 = record["bbox"]
                    cv2.rectangle(
                        annotated,
                        (gx1, gy1),
                        (gx2, gy2),
                        (255, 0, 0),
                        2,
                    )

                for x1, y1, x2, y2, confidence in detections:
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        annotated,
                        f"{confidence:.2f}",
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                label = (
                    f"GT: {expected} | "
                    f"OCR: {best_plate or '---'}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                output_path = self.annotated_dir / image_path.name
                cv2.imwrite(str(output_path), annotated)

            results.append({
                "image": record["image"],
                "expected": expected,
                "predicted": best_plate,
                "detected": bool(detections),
                "exact_match": exact_match,
                "iou": iou,
                "detection_match": detection_match,
                "status": status,
            })

        self.save_results(results)
        self.print_summary(results)

        return results

    def save_results(self, results):
        output_file = self.results_dir / "results.csv"

        fields = [
            "image",
            "expected",
            "predicted",
            "detected",
            "exact_match",
            "iou",
            "detection_match",
            "status",
        ]

        with output_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved: {output_file}")

    @staticmethod
    def print_summary(results):
        total = len(results)

        if total == 0:
            print("Dataset is empty.")
            return

        detected = sum(r["detected"] for r in results)
        exact = sum(r["exact_match"] for r in results)

        bbox_records = [
            r for r in results
            if r["iou"] != ""
        ]

        iou_matches = sum(
            r["detection_match"] for r in bbox_records
        )

        print("\n========== EVALUATION ==========")
        print(f"Images: {total}")
        print(
            f"Detection coverage: "
            f"{detected}/{total} ({detected / total:.1%})"
        )
        print(
            f"Exact OCR accuracy (end-to-end): "
            f"{exact}/{total} ({exact / total:.1%})"
        )

        if bbox_records:
            print(
                f"Detection IoU success: "
                f"{iou_matches}/{len(bbox_records)} "
                f"({iou_matches / len(bbox_records):.1%})"
            )
        else:
            print("Detection IoU: no ground-truth boxes provided")

        print("================================")
