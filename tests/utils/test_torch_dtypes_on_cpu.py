from verl.utils.torch_dtypes import PrecisionType


def test_precision_type_helpers_use_declared_values():
    assert PrecisionType.supported_types() == ["16", "32", "64", "bf16", "mixed"]
    assert PrecisionType.supported_type(16)
    assert PrecisionType.supported_type("bf16")
    assert not PrecisionType.supported_type("float16")
    assert not PrecisionType.supported_type("unknown")
