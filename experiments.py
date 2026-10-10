"""Compare watermarking methods: invisibility (PSNR) and robustness (BER)."""
 
from __future__ import annotations
 
import argparse
import csv
from pathlib import Path
 
import numpy as np
from matplotlib.figure import Figure
 
from main import resolve_seed
from src.qim_watermark.attacks import add_gaussian_noise, jpeg_compress
from src.qim_watermark.block_watermark import (
    extract_block_watermark,
    insert_block_watermark,
)
from src.qim_watermark.io_utils import load_grayscale_image
from src.qim_watermark.metrics import bit_error_rate, compute_psnr
from src.qim_watermark.watermark import (
    extract_watermark,
    generate_watermark,
    insert_watermark,
)
 
JPEG_QUALITIES = [90, 70, 50, 30]
NOISE_SIGMA = 8.0
 
 
def _global_codec(delta: float, size: int, seed: int):
    def embed(image, bits):
        marked, _ = insert_watermark(image=image, watermark_bits=bits, delta=delta, seed=seed)
        return marked
 
    def extract(image, key=seed):
        return extract_watermark(image=image, watermark_size=size, delta=delta, seed=key)
 
    return embed, extract
 
 
def _block_codec(quality: int, redundancy: int, size: int, seed: int):
    def embed(image, bits):
        return insert_block_watermark(image, bits, seed, quality=quality, redundancy=redundancy)
 
    def extract(image, key=seed):
        return extract_block_watermark(image, size, key, quality=quality, redundancy=redundancy)
 
    return embed, extract
 
 
def run_comparison(host: np.ndarray, size: int, seed: int) -> list[dict]:
    watermark = generate_watermark(size, seed)
    methods = {
        "global, delta 18": _global_codec(18.0, size, seed),
        "global, delta 72": _global_codec(72.0, size, seed),
        "block, quality 40, x4": _block_codec(40, 4, size, seed),
        "block, quality 40, x6": _block_codec(40, 6, size, seed),
    }
 
    rows: list[dict] = []
    for name, (embed, extract) in methods.items():
        marked = embed(host, watermark)
 
        def ber(image: np.ndarray, key: int = seed) -> float:
            return bit_error_rate(watermark, extract(image, key))
 
        row = {
            "method": name,
            "psnr": compute_psnr(host, marked),
            "ber_clean": ber(marked),
            "ber_noise": ber(add_gaussian_noise(marked, sigma=NOISE_SIGMA)),
            "ber_wrong_key": ber(marked, seed + 1),
        }
        for quality in JPEG_QUALITIES:
            row[f"ber_jpeg_{quality}"] = ber(jpeg_compress(marked, quality=quality))
        rows.append(row)
    return rows
 
 
def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
 
 
def save_plot(rows: list[dict], path: Path) -> None:
    fig = Figure(figsize=(8, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    for row in rows:
        values = [row[f"ber_jpeg_{q}"] for q in JPEG_QUALITIES]
        ax.plot(JPEG_QUALITIES, values, marker="o", label=f"{row['method']} ({row['psnr']:.1f} dB)")
    ax.axhline(0.5, linestyle="--", color="gray", label="chance level (0.5)")
    ax.set_xlabel("JPEG quality of the attack")
    ax.set_ylabel("BER")
    ax.set_ylim(-0.02, 0.55)
    ax.set_title("Robustness to JPEG compression")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare watermarking methods")
    parser.add_argument("--input", default="test_assets/host.png")
    parser.add_argument("--watermark-size", type=int, default=1024)
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Public demo seed, used only if QIM_KEY is not set.",
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
 
    try:
        host = load_grayscale_image(args.input)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
 
    seed, _ = resolve_seed(args.seed)
    try:
        rows = run_comparison(host, args.watermark_size, seed)
    except ValueError as exc:
        parser.error(str(exc))
 
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, output_dir / "comparison.csv")
    save_plot(rows, output_dir / "comparison_jpeg.png")
 
    columns = [key for key in rows[0] if key != "method"]
    print(f"{'method':<24}" + "".join(f"{name:>14}" for name in columns))
    for row in rows:
        print(f"{row['method']:<24}" + "".join(f"{row[name]:>14.4f}" for name in columns))
    print(f"\nFiles written to: {output_dir.resolve()}")
 
 
if __name__ == "__main__":
    main()