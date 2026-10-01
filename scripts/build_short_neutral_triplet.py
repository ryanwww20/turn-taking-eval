#!/usr/bin/env python
"""Build a diagnostic short-context triplet from one audited TTS triplet.

The neutral preamble is copied from a non-question human utterance. Only the
target utterance varies by addressee condition; the resulting clip ends at the
evaluation deadline and contains no post-window continuation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROLES = ("AI_AGENT", "EVERYONE", "SPECIFIC_HUMAN")
SAMPLE_RATE = 24_000
WAV_NAME = "audio_qwen3-tts-1.7b-clone.wav"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stem_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE or audio.shape[1] != 1:
        raise ValueError(f"expected mono 24 kHz stem: {path}")
    return audio[:, 0], sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scripts-root", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--out-scripts-root", type=Path, required=True)
    ap.add_argument("--out-audio-root", type=Path, required=True)
    ap.add_argument("--source-group", default="tt_h3_g0001")
    args = ap.parse_args()
    group_file = args.scripts_root / f"{args.source_group}_group.json"
    group = json.loads(group_file.read_text())
    members = group["members"]
    source_ai = args.scripts_root / members["AI_AGENT"]
    source_ai_audio = args.audio_root / members["AI_AGENT"]
    ai_aligned = read_jsonl(source_ai_audio / "aligned_script.jsonl")
    ai_meta = json.loads((source_ai / "turn_taking_meta.json").read_text())
    target_index = int(ai_meta["llm_output_summary"]["target_utterance_index"])
    neutral = next(
        utterance for utterance in reversed(ai_aligned[:target_index])
        if "?" not in utterance["text"] and utterance["end"] - utterance["start"] >= 1.0
    )
    neutral_speaker = neutral["speaker"]
    neutral_audio, _ = stem_audio(source_ai_audio / f"{WAV_NAME[:-4]}_{neutral_speaker}.wav")
    neutral_start = int(round(float(neutral["start"]) * SAMPLE_RATE))
    neutral_end = int(round(float(neutral["end"]) * SAMPLE_RATE))
    neutral_clip = neutral_audio[neutral_start:neutral_end]
    if len(neutral_clip) == 0:
        raise ValueError("selected neutral preamble is empty")

    out_scripts = args.out_scripts_root.resolve()
    out_audio = args.out_audio_root.resolve()
    out_scripts.mkdir(parents=True, exist_ok=True)
    out_audio.mkdir(parents=True, exist_ok=True)
    group_id = "tt_h3_short_neutral_g0001"
    out_members = {}
    descriptions = []
    target_start = 2.5
    for role in ROLES:
        source_script = args.scripts_root / members[role]
        source_audio = args.audio_root / members[role]
        meta = json.loads((source_script / "turn_taking_meta.json").read_text())
        aligned = read_jsonl(source_audio / "aligned_script.jsonl")
        target = aligned[target_index]
        target_speaker = target["speaker"]
        target_wave, _ = stem_audio(source_audio / f"{WAV_NAME[:-4]}_{target_speaker}.wav")
        target_first = int(round(float(target["start"]) * SAMPLE_RATE))
        target_last = int(round(float(target["end"]) * SAMPLE_RATE))
        target_clip = target_wave[target_first:target_last]
        target_end = target_start + len(target_clip) / SAMPLE_RATE
        total_samples = int(round((target_end + 10.0) * SAMPLE_RATE))
        rendered = np.zeros(total_samples, dtype=np.float32)
        neutral_first = int(round(0.5 * SAMPLE_RATE))
        target_first_out = int(round(target_start * SAMPLE_RATE))
        rendered[neutral_first:neutral_first + len(neutral_clip)] = neutral_clip
        rendered[target_first_out:target_first_out + len(target_clip)] = target_clip

        sample = f"tt_h3_short_neutral_{role.lower()}_0001"
        out_members[role] = sample
        script_dir = out_scripts / sample
        audio_dir = out_audio / sample
        script_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        dialogue_id = sample
        meta.update({
            "dialogue_id": dialogue_id,
            "scenario_id": dialogue_id,
            "minimal_group_id": group_id,
            "triplet_role": role,
            "diagnostic_kind": "short_neutral_context",
        })
        (script_dir / "turn_taking_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
        )
        event = read_jsonl(source_audio / "aligned_turn_taking_events.jsonl")[0]
        event.update({
            "event_id": f"{dialogue_id}_event_00",
            "dialogue_id": dialogue_id,
            "minimal_group_id": group_id,
            "context_start": 0.0,
            "context_end": target_end,
            "aligned_target_time": target_end,
            "aligned_evaluation_window_start": target_end,
            "aligned_evaluation_window_end": target_end + 10.0,
            "aligned_next_human_start": target_end + 10.0,
            "planned_target_time": target_end,
            "planned_evaluation_window_start": target_end,
            "planned_evaluation_window_end": target_end + 10.0,
            "evaluation_window_duration": 10.0,
            "diagnostic_kind": "short_neutral_context",
        })
        (audio_dir / "aligned_turn_taking_events.jsonl").write_text(
            json.dumps(event, ensure_ascii=False) + "\n"
        )
        sf.write(audio_dir / WAV_NAME, rendered, SAMPLE_RATE, subtype="PCM_16")
        (script_dir / "transcript.txt").write_text(
            f"Neutral human utterance ({neutral_speaker}): {neutral['text']}\n"
            f"Target ({role}): {target['text']}\n"
        )
        descriptions.append({
            "sample": sample,
            "role": role,
            "neutral_text": neutral["text"],
            "target_text": target["text"],
            "target_start": target_start,
            "target_end": target_end,
            "window_end": target_end + 10.0,
        })
    out_group = {
        "minimal_group_id": group_id,
        "num_humans": 3,
        "members": out_members,
        "triplet_complete": True,
        "diagnostic_kind": "short_neutral_context",
        "source_group": args.source_group,
        "descriptions": descriptions,
    }
    (out_scripts / f"{group_id}_group.json").write_text(
        json.dumps(out_group, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Built short-neutral diagnostic triplet under {out_scripts} and {out_audio}")


if __name__ == "__main__":
    main()
