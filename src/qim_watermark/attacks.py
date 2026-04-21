from __future__ import annotations

import cv2
import numpy as np


def add_gaussian_noise(image: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    attacked = image + noise
    return np.clip(attacked, 0, 255).astype(np.float32)


def jpeg_compress(image: np.ndarray, quality: int = 50) -> np.ndarray:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", image_u8, encode_params)
    if not ok:
        raise IOError("Échec de la compression JPEG")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise IOError("Échec de la décompression JPEG")
    return decoded.astype(np.float32)
