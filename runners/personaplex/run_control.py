#!/usr/bin/env python
"""Run at most five official PersonaPlex positive-control trials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from runners.personaplex.run_official import (
        OFFICIAL_REPO,
        checkpoint_provenance,
        ensure_official_voices,
        build_official_command,
        runtime_provenance,
        _atomic_json,
    )
except ModuleNotFoundError:  # direct execution: ``python runners/personaplex/...py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from runners.personaplex.run_official import (
        OFFICIAL_REPO,
        checkpoint_provenance,
        ensure_official_voices,
        build_official_command,
        runtime_provenance,
        _atomic_json,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--official-source", type=Path, required=True,
                    help="clean checkout of github.com/NVIDIA/personaplex")
    ap.add_argument("--moshi-python", default=sys.executable)
    ap.add_argument("--input-wav", type=Path, default=None,
                    help="default: <official-source>/assets/test/input_assistant.wav")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--text-prompt", default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()
    if args.input_wav is None:
        args.input_wav = args.official_source / "assets" / "test" / "input_assistant.wav"
    if not 1 <= len(args.seeds) <= 5:
        raise SystemExit("positive control requires between 1 and 5 trials")
    out_root = args.out_root.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir.resolve()
    source = args.official_source.resolve()
    provenance = checkpoint_provenance(checkpoint_dir, args.revision)
    runtime = runtime_provenance(args.moshi_python, source)
    voices_dir = ensure_official_voices(checkpoint_dir, out_root)
    config = {
        "runner": "official NVIDIA PersonaPlex positive control via unchanged moshi.offline CLI",
        "hf_repo": OFFICIAL_REPO,
        "checkpoint_provenance": provenance,
        "runtime_provenance": runtime,
        "input_wav": str(args.input_wav.resolve()),
        "text_prompt": args.text_prompt,
        "voice_prompt": "NATF2.pt",
        "sampling": {
            "seeds": args.seeds,
            "greedy": args.greedy,
            "temp_audio": 0.8,
            "temp_text": 0.7,
            "topk_audio": 250,
            "topk_text": 25,
        },
        "official_cli_unchanged": True,
        "trials": [],
    }
    _atomic_json(out_root / "run_config.json", config)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source / "moshi"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    for trial_index, seed in enumerate(args.seeds, 1):
        trial = out_root / f"trial_{trial_index:02d}_seed_{seed}"
        trial.mkdir(parents=True, exist_ok=False)
        output_wav = trial / "output.wav"
        output_text = trial / "output.json"
        cmd = build_official_command(
            moshi_python=args.moshi_python,
            official_source=source,
            checkpoint_dir=checkpoint_dir,
            voices_dir=voices_dir,
            input_wav=args.input_wav.resolve(),
            output_wav=output_wav,
            output_text=output_text,
            text_prompt=args.text_prompt,
            seed=seed,
            device=args.device,
            greedy=args.greedy,
        )
        print(f"[{trial_index}/{len(args.seeds)}] official positive control seed={seed}", flush=True)
        subprocess.check_call(cmd, cwd=source, env=env)
        tokens = json.loads(output_text.read_text())
        trial_info = {
            "seed": seed,
            "status": "ok",
            "output_wav": str(output_wav),
            "output_text": str(output_text),
            "decoded_text_token_count": len(tokens),
            "decoded_text_preview": "".join(tokens)[:500],
        }
        config["trials"].append(trial_info)
        _atomic_json(out_root / "run_config.json", config)
    config["status"] = "complete"
    _atomic_json(out_root / "run_config.json", config)


if __name__ == "__main__":
    main()
