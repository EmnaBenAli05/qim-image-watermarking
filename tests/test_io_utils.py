from __future__ import annotations

import numpy as np
import pytest

from src.qim_watermark.io_utils import load_grayscale_image, save_image


def test_save_and_load_grayscale_image(tmp_path):
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    output_path = tmp_path / "image.png"

    save_image(output_path, image)
    loaded = load_grayscale_image(output_path)

    assert loaded.dtype == np.float32
    assert loaded.shape == image.shape
    np.testing.assert_array_equal(loaded, image)


def test_load_grayscale_image_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_grayscale_image(tmp_path / "missing.png")


def test_load_grayscale_image_rejects_invalid_file(tmp_path):
    invalid_path = tmp_path / "not-an-image.txt"
    invalid_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError):
        load_grayscale_image(invalid_path)
