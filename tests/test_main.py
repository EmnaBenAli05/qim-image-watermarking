from __future__ import annotations

import sys
from pathlib import Path

import pytest

import main
from main import resolve_input_path

HOST_IMAGE = Path(__file__).resolve().parent.parent / "test_assets" / "host.png"


def run_main(monkeypatch, output_dir: Path) -> str:
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--input",
            str(HOST_IMAGE),
            "--watermark-size",
            "64",
            "--output-dir",
            str(output_dir),
        ],
    )
    main.main()
    return (output_dir / "metrics.txt").read_text(encoding="utf-8")


def test_main_writes_expected_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("QIM_KEY", "my secret phrase")
    output_dir = tmp_path / "outputs"

    metrics = run_main(monkeypatch, output_dir)

    assert (output_dir / "watermarked.png").is_file()
    assert (output_dir / "attacked_noise.png").is_file()
    assert (output_dir / "attacked_jpeg.png").is_file()
    assert (output_dir / "comparison.png").is_file()
    assert "PSNR" in metrics
    assert "BER sans attaque" in metrics
    assert "Key source" in metrics


def test_main_does_not_leak_secret_key(tmp_path, monkeypatch):
    secret = "my secret phrase"
    monkeypatch.setenv("QIM_KEY", secret)
    derived_seed, _ = main.resolve_seed(0)

    metrics = run_main(monkeypatch, tmp_path / "outputs")

    assert secret not in metrics
    assert str(derived_seed) not in metrics
    assert "Seed" not in metrics
    assert "environment variable QIM_KEY" in metrics


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


def test_resolve_seed_uses_env_key(monkeypatch):
    monkeypatch.setenv("QIM_KEY", "my secret phrase")
    seed, source = main.resolve_seed(42)
    assert seed != 42
    assert "QIM_KEY" in source


def test_resolve_seed_depends_only_on_env_key(monkeypatch):
    monkeypatch.setenv("QIM_KEY", "abc")
    assert main.resolve_seed(1)[0] == main.resolve_seed(2)[0]


def test_resolve_seed_falls_back_to_demo(monkeypatch, capsys):
    monkeypatch.delenv("QIM_KEY", raising=False)
    seed, source = main.resolve_seed(42)
    assert seed == 42
    assert "NOT secret" in source
    assert "WARNING" in capsys.readouterr().err