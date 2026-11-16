from src.access_control import AccessControl
from src.ocr import PlateOCR


def test_access_normalization():
    assert AccessControl.normalize("a-123-bc") == "A123BC"
    assert AccessControl.normalize(" A 123 BC ") == "A123BC"


def test_ocr_normalization():
    assert PlateOCR.normalize("a-123-bc") == "A123BC"
    assert PlateOCR.normalize("A 123 BC") == "A123BC"