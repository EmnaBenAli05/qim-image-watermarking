from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(original, modified, data_range=255))


def bit_error_rate(reference_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    if reference_bits.shape != extracted_bits.shape:
        raise ValueError("Les deux watermarks doivent avoir la même taille")
    return float(np.mean(reference_bits != extracted_bits))
