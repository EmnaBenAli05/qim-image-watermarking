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