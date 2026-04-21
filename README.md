# QIM Image Watermarking

## Description
This project presents the design and implementation of a robust digital image watermarking system based on the Quantization Index Modulation (QIM) method using Python.

The system embeds an invisible watermark into an image in the frequency domain using the Discrete Cosine Transform (DCT), ensuring a balance between imperceptibility and robustness.

---

## Features
- Embed binary watermark into images
- Extract watermark from watermarked images
- Apply attacks:
  - Gaussian noise
  - JPEG compression
- Evaluate performance using:
  - PSNR (Peak Signal-to-Noise Ratio)
  - BER (Bit Error Rate)

---

## Technologies Used
- Python 3
- OpenCV
- NumPy
- SciPy
- Matplotlib
- scikit-image

---

## Project Structure
src/ # Core implementation
outputs/ # Generated results
test_assets/ # Test images
main.py # Main execution script
requirements.txt # Dependencies

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

Run the watermarking workflow with your own input image:

```bash
python main.py --input path/to/image.png
```

If you do not pass `--input`, the program asks you to enter the image path:

```bash
python main.py
Enter the path to the host image: path/to/image.png
```

The input path must point to an existing, readable image file. If the file does
not exist or cannot be decoded as an image, the program stops with a clear error
message.

Optional parameters can still be used with your custom image:

```bash
python main.py --input path/to/image.png --watermark-size 1024 --delta 18 --output-dir outputs
```

Generated files are saved in the output directory:

- `watermarked.png`
- `attacked_noise.png`
- `attacked_jpeg.png`
- `comparison.png`
- `metrics.txt`
