from __future__ import annotations

import numpy as np 
import cv2
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
def test_load_grayscale_image_rejects_empty_file(tmp_path):
    empty_path = tmp_path / "empty.png"
    empty_path.write_bytes(b"")

    with pytest.raises(ValueError):
        load_grayscale_image(empty_path)


def test_load_grayscale_image_accepts_unicode_path(tmp_path):
    folder = tmp_path / "dossier_é"
    folder.mkdir()
    image_path = folder / "image_é.png"
    ok, buffer = cv2.imencode(".png", np.zeros((16, 16), dtype=np.uint8))
    assert ok
    image_path.write_bytes(buffer.tobytes())

    loaded = load_grayscale_image(image_path)

    assert loaded.shape == (16, 16)