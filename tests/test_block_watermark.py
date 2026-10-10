from __future__ import annotations
 
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
 
from src.qim_watermark.attacks import add_gaussian_noise, jpeg_compress
from src.qim_watermark.block_watermark import (
    extract_block_watermark,
    insert_block_watermark,
    jpeg_quant_table,
)
from src.qim_watermark.metrics import bit_error_rate
from src.qim_watermark.watermark import generate_watermark
 
SEED = 42
 
 
def _natural_image(shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Reproducible image with smooth areas and fine texture, like a photo."""
    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.normal(size=shape), sigma=6)
    base = (base - base.min()) / (base.max() - base.min())
    texture = rng.normal(scale=6, size=shape)
    return np.clip(70 + 110 * base + texture, 0, 255).astype(np.float32)
 
 
def test_jpeg_quant_table_matches_the_standard():
    table = jpeg_quant_table(50)
    np.testing.assert_array_equal(table[0], [16, 11, 10, 16, 24, 40, 51, 61])
    assert np.all(jpeg_quant_table(100) == 1)
    assert np.all(jpeg_quant_table(30) >= jpeg_quant_table(50))
    assert np.all(jpeg_quant_table(50) >= jpeg_quant_table(90))
 
 
@pytest.mark.parametrize("quality", [0, 101])
def test_jpeg_quant_table_rejects_invalid_quality(quality):
    with pytest.raises(ValueError):
        jpeg_quant_table(quality)
 
 
def test_insert_then_extract_without_attack():
    image = _natural_image()
    watermark = generate_watermark(size=64, seed=SEED)
 
    marked = insert_block_watermark(image, watermark, SEED)
    extracted = extract_block_watermark(marked, 64, SEED)
 
    assert marked.shape == image.shape
    assert marked.dtype == np.float32
    assert marked.min() >= 0 and marked.max() <= 255
    assert bit_error_rate(watermark, extracted) == 0.0
 
 
@pytest.mark.parametrize("quality", [90, 70, 40])
def test_survives_jpeg_at_the_design_quality_and_above(quality):
    image = _natural_image()
    watermark = generate_watermark(size=64, seed=SEED)
 
    marked = insert_block_watermark(image, watermark, SEED, quality=40)
    attacked = jpeg_compress(marked, quality=quality)
    extracted = extract_block_watermark(attacked, 64, SEED, quality=40)
 
    assert bit_error_rate(watermark, extracted) == 0.0
 
 
def test_survives_gaussian_noise():
    image = _natural_image((256, 256))
    watermark = generate_watermark(size=256, seed=SEED)
 
    marked = insert_block_watermark(image, watermark, SEED)
    attacked = add_gaussian_noise(marked, sigma=8.0)
    extracted = extract_block_watermark(attacked, 256, SEED)
 
    assert bit_error_rate(watermark, extracted) < 0.15
 
 
def test_wrong_key_gives_random_ber():
    image = _natural_image((256, 256))
    watermark = generate_watermark(size=256, seed=SEED)
 
    marked = insert_block_watermark(image, watermark, SEED)
    extracted = extract_block_watermark(marked, 256, SEED + 1)
 
    assert 0.35 < bit_error_rate(watermark, extracted) < 0.65
 
 
def test_insertion_is_deterministic():
    image = _natural_image()
    watermark = generate_watermark(size=64, seed=SEED)
 
    first = insert_block_watermark(image, watermark, SEED)
    second = insert_block_watermark(image, watermark, SEED)
 
    np.testing.assert_array_equal(first, second)
 
 
def test_image_size_not_multiple_of_block_size():
    image = _natural_image((131, 150))
    watermark = generate_watermark(size=64, seed=SEED)
 
    marked = insert_block_watermark(image, watermark, SEED)
    extracted = extract_block_watermark(marked, 64, SEED)
 
    assert marked.shape == image.shape
    # Pixels outside the last full 8x8 block are left untouched.
    np.testing.assert_array_equal(marked[:, 144:], image[:, 144:])
    np.testing.assert_array_equal(marked[128:, :], image[128:, :])
    assert bit_error_rate(watermark, extracted) == 0.0
 
 
def test_rejects_too_many_bits():
    image = np.zeros((8, 8), dtype=np.float32)
    watermark = np.zeros(1000, dtype=np.uint8)
 
    with pytest.raises(ValueError):
        insert_block_watermark(image, watermark, SEED)
 
 
def test_rejects_empty_watermark_and_bad_redundancy():
    image = _natural_image()
 
    with pytest.raises(ValueError):
        insert_block_watermark(image, np.zeros(0, dtype=np.uint8), SEED)
    with pytest.raises(ValueError):
        insert_block_watermark(
            image, np.zeros(8, dtype=np.uint8), SEED, redundancy=0
        )