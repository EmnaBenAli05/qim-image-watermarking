from __future__ import annotations

from pathlib import Path
import main
from main import resolve_input_path
import sys
import pytest 
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
def test_resolve_input_path_uses_given_value():
    assert resolve_input_path("a/b.png").name == "b.png"


def test_resolve_input_path_rejects_empty_prompt(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "   ")
    with pytest.raises(ValueError):
        resolve_input_path(None)


def test_resolve_input_path_rejects_non_interactive(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ValueError):
        resolve_input_path(None)