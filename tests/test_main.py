from __future__ import annotations

from pathlib import Path

import main


def test_main_writes_expected_outputs(tmp_path, monkeypatch):
    input_path = Path("test_assets/host.png")
    output_dir = tmp_path / "outputs"

    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--input",
            str(input_path),
            "--watermark-size",
            "64",
            "--output-dir",
            str(output_dir),
        ],
    )

    main.main()

    assert (output_dir / "watermarked.png").is_file()
    assert (output_dir / "attacked_noise.png").is_file()
    assert (output_dir / "attacked_jpeg.png").is_file()
    assert (output_dir / "comparison.png").is_file()

    metrics = (output_dir / "metrics.txt").read_text(encoding="utf-8")
    assert "PSNR" in metrics
    assert "BER sans attaque" in metrics
