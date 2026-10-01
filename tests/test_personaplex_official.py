"""CPU-only tests for the fail-closed official PersonaPlex runner."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runners.personaplex.run_official import (
    build_official_command,
    checkpoint_provenance,
    crop_input,
)


def test_crop_requires_mono_24khz_and_starts_at_context(tmp_path: Path):
    source = tmp_path / "mix.wav"
    output = tmp_path / "crop.wav"
    sf.write(source, np.zeros(24_000 * 4, dtype=np.float32), 24_000)
    info = crop_input(source, {
        "context_start": 1.0,
        "aligned_evaluation_window_end": 3.25,
    }, output)
    audio, sr = sf.read(output, always_2d=True)
    assert sr == 24_000
    assert audio.shape == (24_000 * 2 + 6_000, 1)
    assert info["audio_start_time"] == 1.0
    assert info["crop_end_time"] == 3.25


def test_crop_rejects_wrong_input_format(tmp_path: Path):
    source = tmp_path / "mix.wav"
    output = tmp_path / "crop.wav"
    sf.write(source, np.zeros((1000, 2), dtype=np.float32), 16_000)
    with pytest.raises(ValueError, match="mono 24 kHz"):
        crop_input(source, {"context_start": 0, "aligned_evaluation_window_end": 1}, output)


def test_checkpoint_provenance_rejects_non_snapshot_and_fake_config(tmp_path: Path):
    checkpoint = tmp_path / "revision"
    checkpoint.mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model", "voices.tgz"):
        (checkpoint / name).write_bytes(b"x")
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "personaplex", "version": "7b-v1"}))
    with pytest.raises(ValueError, match="immutable HF snapshot"):
        checkpoint_provenance(checkpoint, "different")


def test_official_command_has_no_local_proxy_helper():
    cmd = build_official_command(
        moshi_python="python",
        official_source=Path("/opt/personaplex"),
        checkpoint_dir=Path("/snapshot/fdaf4090a61cb315c138a1faee287ffd6c716309"),
        voices_dir=Path("/snapshot/voices"),
        input_wav=Path("input.wav"),
        output_wav=Path("output.wav"),
        output_text=Path("output.json"),
        text_prompt="prompt",
        seed=17,
        device="cuda",
        greedy=False,
    )
    assert "moshi.offline" in cmd
    assert "personaplex_offline_local.py" not in " ".join(cmd)
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "17"
