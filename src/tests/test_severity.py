from src.utils.severity import derive_serious_from_max_severity


def test_severity_mapping_basic():
    assert derive_serious_from_max_severity(0) == 0
    assert derive_serious_from_max_severity(1) == 1
    assert derive_serious_from_max_severity(2) == 1
    assert derive_serious_from_max_severity(3) == 0
    assert derive_serious_from_max_severity(4) == 0
    assert derive_serious_from_max_severity(8) == 0
    assert derive_serious_from_max_severity(9) == 0


def test_severity_mapping_string_inputs():
    assert derive_serious_from_max_severity("1") == 1
    assert derive_serious_from_max_severity(" 2 ") == 1
    assert derive_serious_from_max_severity("0") == 0
    assert derive_serious_from_max_severity("foo") == 0

