from __future__ import annotations
 
import argparse
import hashlib
import os
import sys
from pathlib import Path
 
import numpy as np
from matplotlib.figure import Figure
 
from src.qim_watermark.attacks import add_gaussian_noise, jpeg_compress
from src.qim_watermark.block_watermark import (
    extract_block_watermark,
    insert_block_watermark,
)
from src.qim_watermark.io_utils import load_grayscale_image, save_image
from src.qim_watermark.metrics import bit_error_rate, compute_psnr
from src.qim_watermark.watermark import (
    extract_watermark,
    generate_watermark,
    insert_watermark,
)
 
 
def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number
 
 
def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not number > 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {value}")
    return number
 
 
def jpeg_quality(value: str) -> int:
    number = positive_int(value)
    if number > 100:
        raise argparse.ArgumentTypeError("JPEG quality must be between 1 and 100")
    return number
 
 
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tatouage numérique invisible par DCT + QIM"
    )
    parser.add_argument(
        "--input",
        help="Path to the host image. If omitted, the program asks for it.",
    )
    parser.add_argument(
        "--method",
        choices=["block", "global"],
        default="block",
        help="block: JPEG-aware 8x8 block DCT (default). global: whole-image DCT.",
    )
    parser.add_argument(
        "--watermark-size",
        type=positive_int,
        default=1024,
        help="Nombre de bits du watermark",
    )
    parser.add_argument(
        "--delta",
        type=positive_float,
        default=18.0,
        help="QIM step (global method only)",
    )
    parser.add_argument(
        "--design-quality",
        type=jpeg_quality,
        default=40,
        help="Lowest JPEG quality the watermark must survive (block method only)",
    )
    parser.add_argument(
        "--redundancy",
        type=positive_int,
        default=4,
        help="Number of coefficients carrying each bit (block method only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Public demo seed, used only if QIM_KEY is not set.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=jpeg_quality,
        default=50,
        help="Qualité JPEG pour l'attaque de compression",
    )
    parser.add_argument(
        "--noise-sigma",
        type=positive_float,
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
 
 
def resolve_input_path(input_path: str | None) -> Path:
    if input_path is None:
        if not sys.stdin.isatty():
            raise ValueError(
                "No --input given and no interactive terminal available."
            )
        input_path = input("Enter the path to the host image: ").strip()
    if not input_path:
        raise ValueError("No input image path was provided.")
    return Path(input_path).expanduser()
 
 
def resolve_seed(cli_seed: int) -> tuple[int, str]:
    """Return (seed, source). The secret key is read from QIM_KEY if set."""
    secret = os.environ.get("QIM_KEY")
    if secret:
        digest = hashlib.sha256(secret.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big"), "environment variable QIM_KEY"
    print(
        "WARNING: QIM_KEY is not set; using the public demo seed (--seed).",
        file=sys.stderr,
    )
    return cli_seed, "demo seed (NOT secret)"
 
 
def build_codec(args: argparse.Namespace, seed: int):
    """Return (embed, extract) for the chosen method.
 
    extract(image, key) is blind: it only needs the image, the parameters and
    the key. A different key is used to measure the wrong-key BER.
    """
    if args.method == "block":
 
        def embed(image: np.ndarray, bits: np.ndarray) -> np.ndarray:
            return insert_block_watermark(
                image,
                bits,
                seed,
                quality=args.design_quality,
                redundancy=args.redundancy,
            )
 
        def extract(image: np.ndarray, key: int = seed) -> np.ndarray:
            return extract_block_watermark(
                image,
                args.watermark_size,
                key,
                quality=args.design_quality,
                redundancy=args.redundancy,
            )
 
    else:
 
        def embed(image: np.ndarray, bits: np.ndarray) -> np.ndarray:
            watermarked, _ = insert_watermark(
                image=image, watermark_bits=bits, delta=args.delta, seed=seed
            )
            return watermarked
 
        def extract(image: np.ndarray, key: int = seed) -> np.ndarray:
            return extract_watermark(
                image=image,
                watermark_size=args.watermark_size,
                delta=args.delta,
                seed=key,
            )
 
    return embed, extract
 
 
def create_comparison_figure(
    host: np.ndarray,
    watermarked: np.ndarray,
    attacked_noise: np.ndarray,
    attacked_jpeg: np.ndarray,
    output_path: Path,
) -> None:
    fig = Figure(figsize=(10, 8))
 
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
 
 
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
 
    try:
        input_path = resolve_input_path(args.input)
    except ValueError as exc:
        parser.error(str(exc))
 
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    try:
        host = load_grayscale_image(input_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
 
    seed, key_source = resolve_seed(args.seed)
    watermark = generate_watermark(args.watermark_size, seed)
    embed, extract = build_codec(args, seed)
 
    try:
        watermarked = embed(host, watermark)
    except ValueError as exc:
        parser.error(str(exc))
 
    attacked_noise = add_gaussian_noise(watermarked, sigma=args.noise_sigma)
    attacked_jpeg = jpeg_compress(watermarked, quality=args.jpeg_quality)
 
    extracted_clean = extract(watermarked)
    extracted_noise = extract(attacked_noise)
    extracted_jpeg = extract(attacked_jpeg)
    extracted_wrong_key = extract(watermarked, seed + 1)
 
    psnr_value = compute_psnr(host, watermarked)
    ber_clean = bit_error_rate(watermark, extracted_clean)
    ber_noise = bit_error_rate(watermark, extracted_noise)
    ber_jpeg = bit_error_rate(watermark, extracted_jpeg)
    ber_wrong_key = bit_error_rate(watermark, extracted_wrong_key)
 
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
 
    if args.method == "block":
        method_line = (
            f"Qualité JPEG visée : {args.design_quality} "
            f"(redondance {args.redundancy})"
        )
    else:
        method_line = f"Delta QIM : {args.delta}"
 
    metrics_lines = [
        "=== Résultats du projet QIM/DCT ===",
        f"Image d'entrée : {input_path.name}",
        f"Méthode : {args.method}",
        f"Taille watermark : {args.watermark_size} bits",
        method_line,
        f"Key source : {key_source}",
        f"PSNR (host vs watermarked) : {psnr_value:.4f} dB",
        f"BER sans attaque : {ber_clean:.4f}",
        f"BER après bruit gaussien : {ber_noise:.4f}",
        f"BER après compression JPEG : {ber_jpeg:.4f}",
        f"BER avec une mauvaise clé : {ber_wrong_key:.4f}",
    ]
    save_metrics(output_dir / "metrics.txt", metrics_lines)
 
    print("\n".join(metrics_lines))
    print(f"\nFichiers générés dans : {output_dir.resolve()}")
 
 
if __name__ == "__main__":
    main()