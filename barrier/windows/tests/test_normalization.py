
from src.ocr import normalize_plate
from src.evaluate import calculate_iou


def test_normalize_plate():
    assert normalize_plate("a-123-bc") == "A123BC"
    assert normalize_plate("A 123 BC") == "A123BC"
    assert normalize_plate("a.123.bc") == "A123BC"


def test_iou_identical_boxes():
    box = (10, 10, 100, 100)
    assert calculate_iou(box, box) == 1.0


def test_iou_non_overlapping_boxes():
    box_a = (0, 0, 10, 10)
    box_b = (20, 20, 30, 30)

    assert calculate_iou(box_a, box_b) == 0.0
