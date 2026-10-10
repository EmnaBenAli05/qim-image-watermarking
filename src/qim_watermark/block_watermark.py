"""JPEG-aware QIM watermarking on 8x8 block DCT coefficients.
 
Why this exists: JPEG quantizes the DCT of every 8x8 block. A watermark that
lives in the *same* block DCT, on multiples of the JPEG quantization step,
is left untouched when the image is compressed at that quality or better.
 
Embedding (per slot = one coefficient of one block):
    delta = JPEG quantization step of that coefficient (times ``strength``)
    the coefficient is moved to the nearest multiple of delta whose parity
    matches the bit (even multiple = 0, odd multiple = 1).
 
Redundancy: every bit is carried by several slots chosen with the secret key,
and extraction combines them with a soft decision (average of cos(pi*c/delta)),
which resists noise much better than a hard majority vote.
"""
 
from __future__ import annotations
 
import numpy as np
from scipy.fft import dctn, idctn
 
from .watermark import derive_seed
 
BLOCK = 8
 
# Standard JPEG luminance quantization table (quality 50).
_JPEG_LUMA_Q50 = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)
 
# Mid-frequency coefficients of each block: the DC and the lowest frequencies
# would be visible, the highest ones are the first to be removed by JPEG.
POSITIONS = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1)]
_UV = np.array(POSITIONS)
 
 
def jpeg_quant_table(quality: int) -> np.ndarray:
    """Luminance quantization table used by libjpeg for a quality in 1..100."""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    table = np.floor((_JPEG_LUMA_Q50 * scale + 50) / 100)
    return np.clip(table, 1, 255)
 
 
def _to_blocks(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    hb, wb = h // BLOCK, w // BLOCK
    cropped = image[: hb * BLOCK, : wb * BLOCK]
    return cropped.reshape(hb, BLOCK, wb, BLOCK).transpose(0, 2, 1, 3)
 
 
def _from_blocks(blocks: np.ndarray) -> np.ndarray:
    hb, wb = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(hb * BLOCK, wb * BLOCK)
 
 
def _slots(blocks_shape: tuple[int, int], size: int, seed: int, redundancy: int):
    """Key-dependent slots (block row, block column, position, bit index)."""
    if size <= 0:
        raise ValueError("The watermark must contain at least one bit")
    if redundancy <= 0:
        raise ValueError("Redundancy must be a positive integer")
 
    hb, wb = blocks_shape
    n_available = hb * wb * len(POSITIONS)
    if size > n_available:
        raise ValueError(
            f"Watermark too large: {size} bits, but only {n_available} "
            "coefficients are available in this image."
        )
 
    copies = min(redundancy, n_available // size)
    rng = np.random.default_rng(derive_seed(seed, "block-slots"))
    chosen = rng.permutation(n_available)[: size * copies]
    pos_idx = chosen % len(POSITIONS)
    block_idx = chosen // len(POSITIONS)
    rows, cols = block_idx // wb, block_idx % wb
    bit_index = np.arange(size * copies) % size
    return rows, cols, pos_idx, bit_index
 
 
def _steps(pos_idx: np.ndarray, quality: int, strength: float) -> np.ndarray:
    table = jpeg_quant_table(quality)
    return table[_UV[pos_idx, 0], _UV[pos_idx, 1]] * strength
 
 
def insert_block_watermark(
    image: np.ndarray,
    watermark_bits: np.ndarray,
    seed: int,
    quality: int = 40,
    strength: float = 1.0,
    redundancy: int = 4,
) -> np.ndarray:
    """Embed the bits. ``quality`` is the lowest JPEG quality that must survive."""
    blocks = _to_blocks(image.astype(np.float64))
    coeffs = dctn(blocks, axes=(2, 3), norm="ortho")
    rows, cols, pos_idx, bit_index = _slots(
        blocks.shape[:2], len(watermark_bits), seed, redundancy
    )
    u, v = _UV[pos_idx, 0], _UV[pos_idx, 1]
    delta = _steps(pos_idx, quality, strength)
 
    values = coeffs[rows, cols, u, v]
    bits = np.asarray(watermark_bits)[bit_index].astype(np.int64)
    q = np.round(values / delta)
    wrong_parity = (q.astype(np.int64) % 2) != bits
    q = np.where(wrong_parity, q + np.where(values >= q * delta, 1, -1), q)
    coeffs[rows, cols, u, v] = q * delta
 
    marked = _from_blocks(idctn(coeffs, axes=(2, 3), norm="ortho"))
    out = image.astype(np.float64).copy()
    out[: marked.shape[0], : marked.shape[1]] = marked
    return np.clip(out, 0, 255).astype(np.float32)
 
 
def extract_block_watermark(
    image: np.ndarray,
    watermark_size: int,
    seed: int,
    quality: int = 40,
    strength: float = 1.0,
    redundancy: int = 4,
) -> np.ndarray:
    """Blind extraction: only the image, the size, the parameters and the key."""
    blocks = _to_blocks(image.astype(np.float64))
    coeffs = dctn(blocks, axes=(2, 3), norm="ortho")
    rows, cols, pos_idx, bit_index = _slots(
        blocks.shape[:2], watermark_size, seed, redundancy
    )
    values = coeffs[rows, cols, _UV[pos_idx, 0], _UV[pos_idx, 1]]
    delta = _steps(pos_idx, quality, strength)
 
    # Soft decision: +1 on even multiples of delta (bit 0), -1 on odd ones (bit 1).
    votes = np.cos(np.pi * values / delta)
    total = np.zeros(watermark_size)
    np.add.at(total, bit_index, votes)
    return (total < 0).astype(np.uint8)