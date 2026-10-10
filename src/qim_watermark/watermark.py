from __future__ import annotations
 
import hashlib
from typing import List, Sequence, Tuple
 
import numpy as np
 
from .dct_utils import dct2, idct2
 
Position = Tuple[int, int]
 
 
def derive_seed(seed: int, label: str) -> int:
    """Derive an independent sub-key from the master seed, one per purpose."""
    data = f"{seed}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")
 
 
def generate_watermark(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(derive_seed(seed, "message"))
    return rng.integers(0, 2, size=size, dtype=np.uint8)
 
 
def _mid_frequency_mask(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    rr, cc = np.indices((rows, cols))
 
    # On évite les très basses fréquences et les très hautes fréquences.
    low = (rr + cc) > int(0.10 * (rows + cols))
    high = (rr + cc) < int(0.65 * (rows + cols))
 
    mask = low & high
    mask[0, 0] = False  # éviter le coefficient DC
    return mask
 
 
def _select_positions(
    shape: tuple[int, int], watermark_size: int, seed: int
) -> List[Position]:
    mask = _mid_frequency_mask(shape)
    candidates = np.argwhere(mask)
 
    if watermark_size > len(candidates):
        raise ValueError(
            f"Watermark trop grand : {watermark_size} bits, "
            f"mais seulement {len(candidates)} coefficients disponibles."
        )
 
    rng = np.random.default_rng(derive_seed(seed, "positions"))
    chosen_indices = rng.choice(len(candidates), size=watermark_size, replace=False)
    return [tuple(map(int, candidates[idx])) for idx in chosen_indices]
 
 
def _qim_embed(value: float, bit: int, delta: float) -> float:
    if bit not in (0, 1):
        raise ValueError("Un bit doit être 0 ou 1")
 
    q = np.round(value / delta)
    if int(q) % 2 != bit:
        q += 1 if value >= q * delta else -1
    return float(q * delta)
 
 
def _qim_extract(value: float, delta: float) -> int:
    q = int(np.round(value / delta))
    return q % 2
 
 
def insert_watermark(
    image: np.ndarray,
    watermark_bits: np.ndarray,
    delta: float,
    seed: int,
) -> tuple[np.ndarray, Sequence[Position]]:
    coeffs = dct2(image)
    positions = _select_positions(coeffs.shape, len(watermark_bits), seed)
 
    modified = coeffs.copy()
    for bit, (r, c) in zip(watermark_bits, positions):
        modified[r, c] = _qim_embed(float(modified[r, c]), int(bit), delta)
 
    reconstructed = idct2(modified)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.float32)
    return reconstructed, positions
 
 
def extract_watermark(
    image: np.ndarray,
    watermark_size: int,
    delta: float,
    seed: int,
) -> np.ndarray:
    """Blind extraction: only the image, its size, delta and the key are needed.
 
    The embedding positions are recomputed from the key, never passed in.
    """
    coeffs = dct2(image)
    positions = _select_positions(coeffs.shape, watermark_size, seed)
    extracted = [_qim_extract(float(coeffs[r, c]), delta) for r, c in positions]
    return np.array(extracted, dtype=np.uint8)

