from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.qim_watermark.attacks import add_gaussian_noise, jpeg_compress
from src.qim_watermark.io_utils import load_grayscale_image, save_image
from src.qim_watermark.metrics import bit_error_rate, compute_psnr
from src.qim_watermark.watermark import (
    extract_watermark,
    generate_watermark,
    insert_watermark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tatouage numérique invisible par DCT + QIM"
    )
    parser.add_argument("--input", required=True, help="Chemin de l'image hôte")
    parser.add_argument(
        "--watermark-size",
        type=int,
        default=1024,
        help="Nombre de bits du watermark",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=18.0,
        help="Pas de quantification QIM",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Clé secrète / seed pseudo-aléatoire",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=50,
        help="Qualité JPEG pour l'attaque de compression",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=8.0,
        help="Écart-type du bruit gaussien",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Dossier de sortie",
    )
    return parser


def save_metrics(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def create_comparison_figure(
    host: np.ndarray,
    watermarked: np.ndarray,
    attacked_noise: np.ndarray,
    attacked_jpeg: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(10, 8))

    images = [
        (host, "Image hôte"),
        (watermarked, "Image tatouée"),
        (attacked_noise, "Après bruit gaussien"),
        (attacked_jpeg, "Après compression JPEG"),
    ]

    for i, (img, title) in enumerate(images, start=1):
        ax = fig.add_subplot(2, 2, i)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    host = load_grayscale_image(args.input)
    watermark = generate_watermark(args.watermark_size, args.seed)

    watermarked, selected_positions = insert_watermark(
        image=host,
        watermark_bits=watermark,
        delta=args.delta,
        seed=args.seed,
    )

    attacked_noise = add_gaussian_noise(watermarked, sigma=args.noise_sigma)
    attacked_jpeg = jpeg_compress(watermarked, quality=args.jpeg_quality)

    extracted_clean = extract_watermark(
        image=watermarked,
        positions=selected_positions,
        delta=args.delta,
    )
    extracted_noise = extract_watermark(
        image=attacked_noise,
        positions=selected_positions,
        delta=args.delta,
    )
    extracted_jpeg = extract_watermark(
        image=attacked_jpeg,
        positions=selected_positions,
        delta=args.delta,
    )

    psnr_value = compute_psnr(host, watermarked)
    ber_clean = bit_error_rate(watermark, extracted_clean)
    ber_noise = bit_error_rate(watermark, extracted_noise)
    ber_jpeg = bit_error_rate(watermark, extracted_jpeg)

    save_image(output_dir / "watermarked.png", watermarked)
    save_image(output_dir / "attacked_noise.png", attacked_noise)
    save_image(output_dir / "attacked_jpeg.png", attacked_jpeg)

    create_comparison_figure(
        host,
        watermarked,
        attacked_noise,
        attacked_jpeg,
        output_dir / "comparison.png",
    )

    metrics_lines = [
        "=== Résultats du projet QIM/DCT ===",
        f"Image d'entrée : {os.path.abspath(args.input)}",
        f"Taille watermark : {args.watermark_size} bits",
        f"Delta QIM : {args.delta}",
        f"Seed secrète : {args.seed}",
        f"PSNR (host vs watermarked) : {psnr_value:.4f} dB",
        f"BER sans attaque : {ber_clean:.4f}",
        f"BER après bruit gaussien : {ber_noise:.4f}",
        f"BER après compression JPEG : {ber_jpeg:.4f}",
    ]
    save_metrics(output_dir / "metrics.txt", metrics_lines)

    print("\n".join(metrics_lines))
    print(f"\nFichiers générés dans : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
