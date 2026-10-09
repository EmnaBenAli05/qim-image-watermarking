from __future__ import annotations
 
import numpy as np
import pytest
 
from src.qim_watermark.metrics import bit_error_rate
from src.qim_watermark.watermark import (
    _select_positions,
    derive_seed,
    extract_watermark,
    generate_watermark,
    insert_watermark,
)
 
 
def _texture_image(size: int = 128) -> np.ndarray:
    """Reproducible noisy image: coefficients are spread, like a real photo."""
    rng = np.random.default_rng(0)
    return rng.uniform(0, 255, size=(size, size)).astype(np.float32)
 
 
def test_generate_watermark_is_deterministic_for_same_seed():
    first = generate_watermark(size=32, seed=1234)
    second = generate_watermark(size=32, seed=1234)
 
    np.testing.assert_array_equal(first, second)
 
 
def test_insert_then_extract_watermark_without_attack():
    rows = np.linspace(0, 255, 128, dtype=np.float32)
    image = np.tile(rows, (128, 1))
    watermark = generate_watermark(size=64, seed=42)
 
    watermarked, _ = insert_watermark(
        image=image,
        watermark_bits=watermark,
        delta=18.0,
        seed=42,
    )
    extracted = extract_watermark(
        image=watermarked,
        watermark_size=64,
        delta=18.0,
        seed=42,
    )
 
    assert watermarked.shape == image.shape
    assert watermarked.dtype == np.float32
    assert bit_error_rate(watermark, extracted) == 0.0
 
 
def test_insert_watermark_rejects_too_many_bits():
    image = np.zeros((8, 8), dtype=np.float32)
    watermark = np.zeros(1000, dtype=np.uint8)
 
    with pytest.raises(ValueError):
        insert_watermark(image=image, watermark_bits=watermark, delta=18.0, seed=1)
 
 
def test_blind_extraction_with_right_key():
    image = _texture_image()
    watermark = generate_watermark(size=64, seed=42)
 
    watermarked, _ = insert_watermark(
        image=image, watermark_bits=watermark, delta=18.0, seed=42
    )
    extracted = extract_watermark(
        image=watermarked, watermark_size=64, delta=18.0, seed=42
    )
 
    assert bit_error_rate(watermark, extracted) == 0.0
 
 
def test_wrong_key_gives_random_ber():
    image = _texture_image()
    watermark = generate_watermark(size=256, seed=42)
 
    watermarked, _ = insert_watermark(
        image=image, watermark_bits=watermark, delta=18.0, seed=42
    )
    extracted = extract_watermark(
        image=watermarked, watermark_size=256, delta=18.0, seed=43
    )
 
    assert 0.35 < bit_error_rate(watermark, extracted) < 0.65
 
 
def test_extraction_recomputes_the_inserted_positions():
    image = _texture_image()
    watermark = generate_watermark(size=64, seed=42)
 
    _, positions = insert_watermark(
        image=image, watermark_bits=watermark, delta=18.0, seed=42
    )
 
    assert list(positions) == _select_positions(image.shape, 64, 42)
 
 
def test_message_and_positions_use_different_subkeys():
    assert derive_seed(42, "message") != derive_seed(42, "positions")
    assert derive_seed(42, "message") == derive_seed(42, "message")
    assert derive_seed(42, "message") != derive_seed(43, "message")