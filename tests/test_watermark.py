from __future__ import annotations

import numpy as np
import pytest

from src.qim_watermark.metrics import bit_error_rate
from src.qim_watermark.watermark import (
    generate_watermark,
    insert_watermark,
    extract_watermark,
)


def test_generate_watermark_is_deterministic_for_same_seed():
    first = generate_watermark(size=32, seed=1234)
    second = generate_watermark(size=32, seed=1234)

    np.testing.assert_array_equal(first, second)


def test_insert_then_extract_watermark_without_attack():
    rows = np.linspace(0, 255, 128, dtype=np.float32)
    image = np.tile(rows, (128, 1))
    watermark = generate_watermark(size=64, seed=42)

    watermarked, positions = insert_watermark(
        image=image,
        watermark_bits=watermark,
        delta=18.0,
        seed=42,
    )
    extracted = extract_watermark(
        image=watermarked,
        positions=positions,
        delta=18.0,
    )

    assert watermarked.shape == image.shape
    assert watermarked.dtype == np.float32
    assert bit_error_rate(watermark, extracted) == 0.0


def test_insert_watermark_rejects_too_many_bits():
    image = np.zeros((8, 8), dtype=np.float32)
    watermark = np.zeros(1000, dtype=np.uint8)

    with pytest.raises(ValueError):
        insert_watermark(image=image, watermark_bits=watermark, delta=18.0, seed=1)
