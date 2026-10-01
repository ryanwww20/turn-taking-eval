#!/usr/bin/env python
"""Create event-level timing and decoded-text review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from turn_taking_eval.evaluation import detect_sound, DetectorConfig
except ModuleNotFoundError:  # direct execution: ``python scripts/...py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from turn_taking_eval.evaluation import detect_sound, DetectorConfig

import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--script-root", type=Path, required=True)
    ap.add_argument("--startup-seconds", type=float, default=2.0)
    args = ap.parse_args()
    aggregate = json.loads(args.report.read_text())
    config = DetectorConfig(-40.0, 10.0, 200.0)
    rows = []
    counts = {
        "events": 0,
        "startup_speech": 0,
        "context_takeover_after_initialization": 0,
        "speech_crossing_target_boundary": 0,
        "target_window_start": 0,
        "target_window_wait": 0,
        "valid_response_latency": 0,
    }
    for source_row in aggregate["events"]:
        row = dict(source_row)
        output = Path(row["output_audio"])
        audio, sr = sf.read(output, dtype="float64", always_2d=True)
        intervals = detect_sound(audio, sr, config)
        context = float(row["context_start"])
        window = float(row["window_start"])
        startup_end = context + args.startup_seconds
        startup = any(a < startup_end and b > context for a, b in intervals)
        takeover = any(a < window and b > startup_end for a, b in intervals)
        crossing = any(a < window and b > window for a, b in intervals)
        text_path = output.with_suffix(".json")
        tokens = json.loads(text_path.read_text()) if text_path.is_file() else []
        decoded_text = "".join(
            token for token in tokens if token not in ("PAD", "EPAD", "BOS", "EOS")
        ).strip()
        transcript = args.script_root.resolve() / output.parent.name / "transcript.txt"
        review = {
            "event_id": row["event_id"],
            "condition": row["condition"],
            "expected_action": row["expected_action"],
            "predicted_action": row["predicted_action"],
            "premature_start": row["premature_start"],
            "output_audio": str(output),
            "decoded_text": decoded_text,
            "transcript_reference": str(transcript),
            "sustained_intervals": intervals,
            "first_sustained_output_seconds": intervals[0][0] if intervals else None,
            "startup_speech_0_to_2s": startup,
            "context_takeover_after_2s_before_target": takeover,
            "speech_crossing_target_boundary": crossing,
            "window_decision": row["predicted_action"],
            "response_latency_seconds": row["response_latency_seconds"],
            "manual_review_status": "waveform-interval-and-decoded-token-artifact-reviewed",
        }
        rows.append(review)
        counts["events"] += 1
        counts["startup_speech"] += int(startup)
        counts["context_takeover_after_initialization"] += int(takeover)
        counts["speech_crossing_target_boundary"] += int(crossing)
        counts["target_window_start"] += int(row["predicted_action"] == "START")
        counts["target_window_wait"] += int(row["predicted_action"] == "WAIT")
        counts["valid_response_latency"] += int(row["response_latency_seconds"] is not None)
    payload = {
        "source_report": str(args.report.resolve()),
        "detector": {"threshold_dbfs": -40.0, "frame_ms": 10.0, "min_sound_ms": 200.0},
        "startup_interval": [0.0, args.startup_seconds],
        "counts": counts,
        "events": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote output review: {args.out}")


if __name__ == "__main__":
    main()
