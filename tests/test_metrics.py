from __future__ import annotations

import numpy as np
import pytest

from src.qim_watermark.metrics import bit_error_rate, compute_psnr


def test_bit_error_rate_counts_different_bits():
    reference = np.array([0, 1, 1, 0], dtype=np.uint8)
    extracted = np.array([0, 0, 1, 1], dtype=np.uint8)

    assert bit_error_rate(reference, extracted) == 0.5


def test_bit_error_rate_rejects_different_shapes():
    with pytest.raises(ValueError):
        bit_error_rate(np.array([0, 1]), np.array([0, 1, 1]))


def test_compute_psnr_is_infinite_for_identical_images():
    image = np.full((8, 8), 128, dtype=np.float32)

    with pytest.warns(RuntimeWarning):
        assert compute_psnr(image, image) == float("inf")
