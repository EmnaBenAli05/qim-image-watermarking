# QIM Image Watermarking

Invisible, key-based watermarking of grayscale images with the Discrete Cosine Transform (DCT) and Quantization Index Modulation (QIM), in Python. The project embeds a binary watermark, simulates attacks (Gaussian noise, JPEG compression), extracts the watermark blindly and measures invisibility (PSNR) and robustness (BER).

## How it works

QIM hides a bit in the parity of `round(c / delta)`, where `c` is a DCT coefficient: the coefficient is moved to the nearest multiple of `delta` with the right parity (even = 0, odd = 1). The bit survives as long as the coefficient moves by less than `delta / 2`.

Two methods are implemented:

- **block** (default, JPEG-aware): 8x8 block DCT like JPEG, with `delta` equal to the JPEG quantization step of each coefficient. Each bit is spread over several key-selected coefficients and decoded with a soft decision. JPEG leaves coefficients that already sit on its quantization grid unchanged.
- **global** (baseline): the same QIM on the DCT of the whole image. It is more invisible for a small `delta` but is destroyed by JPEG.

Extraction is blind: only the image, the parameters and the secret key are needed.

## Security

- The key is read from the `QIM_KEY` environment variable (hashed with SHA-256) and never written to the outputs.
- Message and embedding positions use independent sub-keys. With a wrong key the BER is close to 0.5 (chance level).
- Without `QIM_KEY`, a public demo seed is used and a warning is printed: it is **not secret**.
- The Docker image runs as a non-root user.

## Usage

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

```bash
# Linux, macOS, Git Bash
export QIM_KEY="my secret phrase"
python main.py --input path/to/image.png
```

```powershell
# Windows PowerShell
$env:QIM_KEY="my secret phrase"
python main.py --input path/to/image.png
```

| Option | Default | Description |
|---|---|---|
| `--input` | none | Host image (asked interactively if omitted and a terminal is available) |
| `--method` | block | `block` or `global` |
| `--watermark-size` | 1024 | Number of watermark bits |
| `--design-quality` | 40 | Lowest JPEG quality to survive (block) |
| `--redundancy` | 4 | Coefficients carrying each bit (block) |
| `--delta` | 18.0 | QIM step (global) |
| `--jpeg-quality` | 50 | JPEG quality of the attack |
| `--noise-sigma` | 8.0 | Gaussian noise of the attack |
| `--output-dir` | outputs | Output directory |

Outputs: `watermarked.png`, `attacked_noise.png`, `attacked_jpeg.png`, `comparison.png` and `metrics.txt`.

## Results

1024 bits, Gaussian noise sigma 8, on the `camera` sample image of scikit-image (512x512). The attacks are seeded, so the numbers are reproducible. The BER without attack is 0 everywhere; 0.5 is chance level.

| Method | PSNR (dB) | BER noise | BER JPEG 70 | BER JPEG 50 | BER JPEG 30 | BER wrong key |
|---|---|---|---|---|---|---|
| global, delta 18 | 51.70 | 0.272 | 0.271 | 0.354 | 0.402 | 0.516 |
| global, delta 72 | 39.09 | 0.000 | 0.259 | 0.323 | 0.389 | 0.482 |
| **block, default** | 42.31 | 0.024 | 0.000 | 0.000 | 0.001 | 0.487 |

- A larger `delta` does not save the global method from JPEG: the problem is structural, not a question of amplitude.
- The block method survives JPEG down to quality 30 at a higher PSNR than the global method with `delta` 72, but it is less invisible than the global method with a small `delta`: robustness costs invisibility.
- On the synthetic image `test_assets/host.png` (flat areas), the block method reaches 35.6 dB with the same robustness.

Reproduce (the photograph is not stored in the repository):

```bash
python -c "import cv2; from skimage import data; cv2.imwrite('camera_sample.png', data.camera())"
QIM_KEY="my secret phrase" python experiments.py --input camera_sample.png --output-dir outputs_photo
```

This writes `comparison.csv` (all methods, JPEG 90 to 30) and `comparison_jpeg.png`.

## Docker

```bash
docker build -t qim-watermark-project .
docker run --name qim-run -e QIM_KEY="my secret phrase" qim-watermark-project
docker cp qim-run:/app/outputs ./outputs && docker rm qim-run
```

## Tests and CI

```bash
pip install -r requirements-dev.txt
python -m pytest
```

GitHub Actions runs the tests on Python 3.10 and 3.12, builds the Docker image, runs it without a terminal and checks that the key does not appear in `metrics.txt`.

## Limitations

- The block method is designed for a JPEG quality (`--design-quality`); lower qualities are not guaranteed to be survived.
- Geometric attacks (rotation, cropping, rescaling) are not covered, and only two attacks are implemented.
- Results come from one standard photograph and one synthetic image; other images will differ.
- The key is derived on 32 bits (brute-forceable) and the message is generated from the key: fine for an educational project, not for production.

See also [docs/azure-deployment.md](docs/azure-deployment.md).
## Azure Deployment

See [docs/azure-deployment.md](docs/azure-deployment.md).