from __future__ import annotations
 
import cv2
import numpy as np
 
 
def add_gaussian_noise(
    image: np.ndarray, sigma: float = 8.0, seed: int = 0
) -> np.ndarray:
    """Add Gaussian noise. The noise is reproducible: same seed, same noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    attacked = image + noise
    return np.clip(attacked, 0, 255).astype(np.float32)
 
 
def jpeg_compress(image: np.ndarray, quality: int = 50) -> np.ndarray:
    if not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    # Round to the nearest integer: astype(np.uint8) alone would truncate.
    image_u8 = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", image_u8, encode_params)
    if not ok:
        raise IOError("Échec de la compression JPEG")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise IOError("Échec de la décompression JPEG")
    return decoded.astype(np.float32)