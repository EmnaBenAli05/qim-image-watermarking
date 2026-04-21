from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_grayscale_image(path: str | Path) -> np.ndarray:
    image_path = Path(path).expanduser()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"The file is not a valid readable image: {image_path}")

    return image.astype(np.float32)


def save_image(path: str | Path, image: np.ndarray) -> None:
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(path), clipped)
    if not ok:
        raise IOError(f"Impossible d'enregistrer l'image : {path}")
