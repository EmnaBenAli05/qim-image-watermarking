from __future__ import annotations
 
import numpy as np
import pytest
 
from src.qim_watermark.attacks import add_gaussian_noise, jpeg_compress
 
 
def _image() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.uniform(40, 215, size=(64, 64)).astype(np.float32)
 
 
def test_gaussian_noise_is_reproducible():
    image = _image()
 
    first = add_gaussian_noise(image, sigma=8.0, seed=3)
    second = add_gaussian_noise(image, sigma=8.0, seed=3)
 
    np.testing.assert_array_equal(first, second)
 
 
def test_gaussian_noise_changes_with_the_seed():
    image = _image()
 
    first = add_gaussian_noise(image, sigma=8.0, seed=3)
    second = add_gaussian_noise(image, sigma=8.0, seed=4)
 
    assert not np.array_equal(first, second)
 
 
def test_gaussian_noise_has_the_requested_strength():
    image = np.full((128, 128), 128.0, dtype=np.float32)
 
    attacked = add_gaussian_noise(image, sigma=8.0)
 
    assert attacked.dtype == np.float32
    assert abs(float(np.std(attacked - image)) - 8.0) < 0.5
    assert attacked.min() >= 0 and attacked.max() <= 255
 
 
def test_jpeg_compress_keeps_shape_and_type():
    image = _image()
 
    attacked = jpeg_compress(image, quality=50)
 
    assert attacked.shape == image.shape
    assert attacked.dtype == np.float32
    assert attacked.min() >= 0 and attacked.max() <= 255
 
 
def test_jpeg_lower_quality_changes_the_image_more():
    image = _image()
 
    error_high = np.mean(np.abs(jpeg_compress(image, quality=90) - image))
    error_low = np.mean(np.abs(jpeg_compress(image, quality=20) - image))
 
    assert error_low > error_high
 
 
@pytest.mark.parametrize("quality", [0, 101])
def test_jpeg_rejects_invalid_quality(quality):
    with pytest.raises(ValueError):
        jpeg_compress(_image(), quality=quality)