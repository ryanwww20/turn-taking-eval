#!/usr/bin/env python
"""Run the immutable official PersonaPlex checkpoint on schema-v2 triplets.

This command deliberately invokes NVIDIA's ``moshi.offline`` module unchanged.
The checkpoint directory must be an immutable Hugging Face snapshot, and the
runner refuses the local Moshiko proxy layout used by the historical probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Iterable

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_REPO = "nvidia/personaplex-7b-v1"
OFFICIAL_SAMPLE_RATE = 24_000
MODEL_FILENAMES = (
    "config.json",
    "model.safetensors",
    "tokenizer-e351c8d8-checkpoint125.safetensors",
    "tokenizer_spm_32k_3.model",
    "voices.tgz",
)
DEFAULT_PROMPT = (
    "You are a wise and friendly AI Agent in a group conversation. "
    "Answer clearly and briefly when AI Agent is asked to respond. "
    "When the humans are talking among themselves, remain silent."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_provenance(checkpoint_dir: Path, revision: str) -> dict:
    """Validate and hash all official assets used by the runtime."""
    checkpoint_dir = checkpoint_dir.resolve()
    if checkpoint_dir.name != revision:
        raise ValueError(
            "checkpoint-dir must be an immutable HF snapshot whose final path "
            f"component is revision {revision!r}: {checkpoint_dir}"
        )
    files = {}
    for filename in MODEL_FILENAMES:
        path = checkpoint_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing official PersonaPlex asset: {path}")
        files[filename] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    config = json.loads((checkpoint_dir / "config.json").read_text())
    if config.get("model_type") != "personaplex" or config.get("version") != "7b-v1":
        raise ValueError(f"unexpected PersonaPlex config: {config}")
    if files["model.safetensors"]["size_bytes"] < 10_000_000_000:
        raise ValueError("model.safetensors is implausibly small for official PersonaPlex")
    return {
        "hf_repo": OFFICIAL_REPO,
        "immutable_revision": revision,
        "snapshot_dir": str(checkpoint_dir),
        "files": files,
        "config": config,
    }


def runtime_provenance(moshi_python: str, official_source: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(official_source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(official_source), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"official PersonaPlex source tree is dirty: {official_source}")
    probe = r'''
import importlib.metadata as md
import json
import sys
packages = {}
for name in ("torch", "numpy", "safetensors", "huggingface-hub", "sentencepiece", "sphn"):
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        packages[name] = None
result = {"packages": packages, "python": sys.version}
try:
    import torch
    result["torch_version"] = torch.__version__
    result["cuda_version"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    result["gpu_count"] = torch.cuda.device_count()
    result["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception as exc:
    result["torch_probe_error"] = repr(exc)
print(json.dumps(result))
'''
    probe_result = json.loads(
        subprocess.check_output([moshi_python, "-c", probe], text=True)
    )
    return {
        "python": str(Path(moshi_python).resolve()),
        "python_version": subprocess.check_output(
            [moshi_python, "--version"], text=True
        ).strip(),
        "official_source": str(official_source.resolve()),
        "official_source_commit": commit,
        "official_source_dirty": False,
        "package_versions": probe_result.pop("packages"),
        "hardware_probe": probe_result,
    }


def _read_one_jsonl(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one event in {path}, got {len(rows)}")
    return rows[0]


def complete_groups(scripts_root: Path, max_groups: int) -> list[list[Path]]:
    groups: list[list[Path]] = []
    for group_file in sorted(scripts_root.glob("tt_*_group.json")):
        group = json.loads(group_file.read_text())
        if not group.get("triplet_complete"):
            continue
        members = group.get("members") or {}
        if set(members) != {"AI_AGENT", "EVERYONE", "SPECIFIC_HUMAN"}:
            continue
        paths = [scripts_root / members[role] for role in (
            "AI_AGENT", "EVERYONE", "SPECIFIC_HUMAN"
        )]
        if not all(path.is_dir() for path in paths):
            continue
        groups.append(paths)
        if max_groups and len(groups) >= max_groups:
            break
    return groups


def crop_input(mix_path: Path, event: dict, output: Path) -> dict:
    """Write exactly [context_start, aligned_window_end) on the input timeline."""
    audio, sr = sf.read(mix_path, dtype="float32", always_2d=True)
    if sr != OFFICIAL_SAMPLE_RATE or audio.shape[1] != 1:
        raise ValueError(
            f"official PersonaPlex input must be mono 24 kHz: {mix_path} "
            f"has sr={sr}, channels={audio.shape[1]}"
        )
    start = float(event["context_start"])
    end = float(event["aligned_evaluation_window_end"])
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError(f"invalid context/window bounds: {start}, {end}")
    first = max(0, math.floor(start * sr + 1e-7))
    last = min(len(audio), math.ceil(end * sr - 1e-7))
    if last <= first or last / sr < end - 1 / sr:
        raise ValueError(f"{mix_path} does not cover evaluation window ending at {end}s")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio[first:last, 0], sr, subtype="PCM_16")
    return {
        "source_sample_rate": sr,
        "source_channels": audio.shape[1],
        "first_sample": first,
        "last_sample_exclusive": last,
        "input_samples": last - first,
        "input_duration_seconds": (last - first) / sr,
        "audio_start_time": first / sr,
        "crop_end_time": last / sr,
    }


def ensure_official_voices(checkpoint_dir: Path, out_root: Path) -> Path:
    """Extract only the official voices/ tree from the pinned archive."""
    existing = checkpoint_dir / "voices"
    if (existing / "NATF2.pt").is_file():
        return existing
    target = out_root / "_official_voices"
    if not (target / "NATF2.pt").is_file():
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(checkpoint_dir / "voices.tgz", "r:gz") as archive:
            members = archive.getmembers()
            if not members or any(
                not (member.name == "voices" or member.name.startswith("voices/"))
                for member in members
            ):
                raise ValueError("official voices.tgz contains an unexpected path")
            archive.extractall(path=out_root, filter="data")
        extracted = out_root / "voices"
        if extracted != target:
            if target.exists():
                shutil.rmtree(target)
            extracted.rename(target)
    if not (target / "NATF2.pt").is_file():
        raise FileNotFoundError(f"official NATF2.pt not found after extraction: {target}")
    return target


def build_official_command(
    *,
    moshi_python: str,
    official_source: Path,
    checkpoint_dir: Path,
    voices_dir: Path,
    input_wav: Path,
    output_wav: Path,
    output_text: Path,
    text_prompt: str,
    seed: int,
    device: str,
    greedy: bool,
) -> list[str]:
    """Build the unchanged official CLI invocation with pinned local assets."""
    cmd = [
        moshi_python,
        "-m",
        "moshi.offline",
        "--hf-repo",
        OFFICIAL_REPO,
        "--voice-prompt",
        "NATF2.pt",
        "--voice-prompt-dir",
        str(voices_dir),
        "--input-wav",
        str(input_wav),
        "--seed",
        str(seed),
        "--output-wav",
        str(output_wav),
        "--output-text",
        str(output_text),
        "--text-prompt",
        text_prompt,
        "--tokenizer",
        str(checkpoint_dir / "tokenizer_spm_32k_3.model"),
        "--moshi-weight",
        str(checkpoint_dir / "model.safetensors"),
        "--mimi-weight",
        str(checkpoint_dir / "tokenizer-e351c8d8-checkpoint125.safetensors"),
        "--device",
        device,
    ]
    if greedy:
        cmd.append("--greedy")
    return cmd


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _validate_output(path: Path, expected_samples: int) -> dict:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != OFFICIAL_SAMPLE_RATE or audio.shape[1] != 1:
        raise ValueError(f"agent output must be mono 24 kHz: {path}")
    if len(audio) < expected_samples:
        raise ValueError(
            f"agent output does not cover observation horizon: {path} "
            f"has {len(audio)} samples, needs {expected_samples}"
        )
    return {
        "sample_rate": sr,
        "channels": audio.shape[1],
        "samples": len(audio),
        "duration_seconds": len(audio) / sr,
        "covers_input": len(audio) >= expected_samples,
    }


def run_dataset(args: argparse.Namespace) -> None:
    scripts_root = args.scripts_root.resolve()
    audio_root = args.audio_root.resolve()
    out_root = args.out_root.resolve()
    if out_root.exists() and any(out_root.iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    groups = complete_groups(scripts_root, args.max_groups)
    if not groups:
        raise SystemExit(f"no complete triplets found under {scripts_root}")
    checkpoint_dir = args.checkpoint_dir.resolve()
    provenance = checkpoint_provenance(checkpoint_dir, args.revision)
    official_source = args.official_source.resolve()
    runtime = runtime_provenance(args.moshi_python, official_source)
    voices_dir = ensure_official_voices(checkpoint_dir, out_root)
    samples = [sample for group in groups for sample in group]
    run_config = {
        "runner": "official NVIDIA PersonaPlex via unchanged moshi.offline CLI",
        "scripts_root": str(scripts_root),
        "audio_root": str(audio_root),
        "out_root": str(out_root),
        "checkpoint_provenance": provenance,
        "runtime_provenance": runtime,
        "sampling": {
            "seed": args.seed,
            "shared_seed_across_triplet": True,
            "greedy": args.greedy,
            "temp_audio": 0.8,
            "temp_text": 0.7,
            "topk_audio": 250,
            "topk_text": 25,
        },
        "text_prompt": args.text_prompt,
        "voice_prompt": "NATF2.pt",
        "voice_prompt_dir": str(voices_dir),
        "official_cli_unchanged": True,
        "future_audio_exposed": False,
        "labels_exposed": False,
        "metadata_exposed": False,
        "inference_seeds": [args.seed],
        "planned_groups": len(groups),
        "planned_samples": len(samples),
        "results": [],
    }
    _atomic_json(out_root / "run_config.json", run_config)

    env = os.environ.copy()
    source_python = official_source / "moshi"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_python), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    manifest_rows = []
    for index, script_dir in enumerate(samples, 1):
        sample = script_dir.name
        audio_dir = audio_root / sample
        event = _read_one_jsonl(audio_dir / "aligned_turn_taking_events.jsonl")
        mix_path = audio_dir / args.input_wav_name
        meta_path = script_dir / "turn_taking_meta.json"
        dest = out_root / sample
        input_wav = dest / "input_through_eval_window.wav"
        output_wav = dest / "agent.wav"
        output_text = dest / "agent.json"
        crop = crop_input(mix_path, event, input_wav)
        if crop["audio_start_time"] != float(event["context_start"]):
            raise ValueError(
                f"sample-aligned crop does not start at context_start for {sample}: "
                f"{crop['audio_start_time']} != {event['context_start']}"
            )
        cmd = build_official_command(
            moshi_python=args.moshi_python,
            official_source=official_source,
            checkpoint_dir=checkpoint_dir,
            voices_dir=voices_dir,
            input_wav=input_wav,
            output_wav=output_wav,
            output_text=output_text,
            text_prompt=args.text_prompt,
            seed=args.seed,
            device=args.device,
            greedy=args.greedy,
        )
        print(f"[{index}/{len(samples)}] official PersonaPlex {sample}", flush=True)
        subprocess.check_call(cmd, cwd=official_source, env=env)
        output_info = _validate_output(output_wav, crop["input_samples"])
        text_tokens = json.loads(output_text.read_text())
        result = {
            "sample": sample,
            "minimal_group_id": event.get("minimal_group_id"),
            "condition": event.get("addressee_type"),
            "seed": args.seed,
            "status": "ok",
            "input_crop": crop,
            "output": output_info,
            "decoded_text_token_count": len(text_tokens),
        }
        run_config["results"].append(result)
        manifest_rows.append({
            "events": str((audio_dir / "aligned_turn_taking_events.jsonl").resolve()),
            "meta": str(meta_path.resolve()),
            "output_audio": str(output_wav.resolve()),
            "audio_start_time": crop["audio_start_time"],
        })
        run_config["completed_samples"] = index
        _atomic_json(out_root / "run_config.json", run_config)

    manifest = out_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    run_config["manifest"] = str(manifest)
    run_config["status"] = "complete"
    _atomic_json(out_root / "run_config.json", run_config)
    _atomic_json(out_root / "run_progress.json", {
        "planned_groups": len(groups),
        "planned_samples": len(samples),
        "completed_samples": len(manifest_rows),
        "status": "complete",
    })
    print(f"Wrote evaluator manifest: {manifest}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scripts-root", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--official-source", type=Path, required=True,
                    help="clean checkout of github.com/NVIDIA/personaplex")
    ap.add_argument("--moshi-python", default=sys.executable)
    ap.add_argument("--input-wav-name", default="audio_qwen3-tts-1.7b-clone.wav")
    ap.add_argument("--max-groups", type=int, default=1)
    ap.add_argument("--text-prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main() -> None:
    run_dataset(parse_args())


if __name__ == "__main__":
    main()
