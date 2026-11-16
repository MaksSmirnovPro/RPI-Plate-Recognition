from src.access_control import AccessControl


def test_allowed_plate(tmp_path):
    allowed_file = tmp_path / "allowed_plates.txt"
    allowed_file.write_text("A123BC\n", encoding="utf-8")

    access = AccessControl(
        allowed_plates_file=str(allowed_file),
        required_confirmations=1,
        cooldown_seconds=0,
    )

    assert access.is_allowed("A123BC")
    assert access.is_allowed("a123bc")
    assert not access.is_allowed("X999XX")


def test_confirmation_required(tmp_path):
    allowed_file = tmp_path / "allowed_plates.txt"
    allowed_file.write_text("A123BC\n", encoding="utf-8")

    access = AccessControl(
        allowed_plates_file=str(allowed_file),
        required_confirmations=2,
        cooldown_seconds=0,
    )

    assert not access.confirm_plate("A123BC")
    assert access.confirm_plate("A123BC")


def test_unknown_plate_resets_confirmation(tmp_path):
    allowed_file = tmp_path / "allowed_plates.txt"
    allowed_file.write_text("A123BC\n", encoding="utf-8")

    access = AccessControl(
        allowed_plates_file=str(allowed_file),
        required_confirmations=2,
        cooldown_seconds=0,
    )

    assert not access.confirm_plate("A123BC")
    assert not access.confirm_plate("UNKNOWN")
    assert not access.confirm_plate("A123BC")