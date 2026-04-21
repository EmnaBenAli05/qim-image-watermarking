from __future__ import annotations

import numpy as np
from scipy.fftpack import dct, idct


def dct2(image: np.ndarray) -> np.ndarray:
    return dct(dct(image.T, norm="ortho").T, norm="ortho")


def idct2(coeffs: np.ndarray) -> np.ndarray:
    return idct(idct(coeffs.T, norm="ortho").T, norm="ortho")
