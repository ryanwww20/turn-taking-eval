#!/usr/bin/env python
"""Quantify target-time and pre-target waveform differences within triplets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--input-wav-name", default="audio_qwen3-tts-1.7b-clone.wav")
    args = ap.parse_args()
    rows = []
    by_group = {}
    for event_path in sorted(args.audio_root.glob("tt_*_*/aligned_turn_taking_events.jsonl")):
        event = json.loads(event_path.read_text().splitlines()[0])
        wav = event_path.parent / args.input_wav_name
        audio, sr = sf.read(wav, dtype="float64")
        if sr != 24_000 or audio.ndim != 1:
            raise ValueError(f"expected mono 24 kHz mix: {wav}")
        row = {
            "group": event["minimal_group_id"],
            "condition": event["addressee_type"],
            "sample": event_path.parent.name,
            "target_time": event["aligned_evaluation_window_start"],
            "duration_seconds": len(audio) / sr,
        }
        by_group.setdefault(row["group"], []).append((row, audio, sr))
    for group, members in sorted(by_group.items()):
        target_times = [float(row["target_time"]) for row, _, _ in members]
        common_prefix_seconds = min(target_times)
        prefix_samples = int(np.floor(common_prefix_seconds * 24_000))
        base = members[0][1][:prefix_samples]
        for row, audio, sr in members:
            prefix = audio[:prefix_samples]
            row["common_prefix_seconds"] = common_prefix_seconds
            row["target_time_spread_seconds"] = max(target_times) - min(target_times)
            row["prefix_rms"] = float(np.sqrt(np.mean(prefix * prefix))) if len(prefix) else 0.0
            row["prefix_equal_to_first"] = bool(np.array_equal(prefix, base))
            row["prefix_rmse_to_first"] = float(
                np.sqrt(np.mean((prefix - base) ** 2))
            ) if len(prefix) else 0.0
            rows.append(row)
    exact_nonreference_matches = 0
    all_group_prefixes_identical = 0
    for group in by_group:
        group_rows = [row for row in rows if row["group"] == group]
        exact_nonreference_matches += sum(
            row["prefix_equal_to_first"] for row in group_rows[1:]
        )
        all_group_prefixes_identical += int(
            all(row["prefix_equal_to_first"] for row in group_rows)
        )
    payload = {
        "audio_root": str(args.audio_root.resolve()),
        "sample_rate": 24_000,
        "n_groups": len(by_group),
        "n_members": len(rows),
        "groups": rows,
        "summary": {
            "exact_nonreference_prefix_matches": exact_nonreference_matches,
            "groups_with_all_prefixes_identical": all_group_prefixes_identical,
            "mean_prefix_rmse": float(np.mean([row["prefix_rmse_to_first"] for row in rows])) if rows else None,
            "max_target_time_spread_seconds": max(
                (row["target_time_spread_seconds"] for row in rows), default=None
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote prefix audit: {args.out}")


if __name__ == "__main__":
    main()
