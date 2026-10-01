"""Audio-based, event-level evaluation for turn-taking dataset schema v2.

See docs/EVALUATION.md for the manifest contract and metric definitions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


CONDITIONS = ("AI_AGENT", "EVERYONE", "SPECIFIC_HUMAN")


@dataclass(frozen=True)
class DetectorConfig:
    threshold_dbfs: float = -40.0
    frame_ms: float = 10.0
    min_sound_ms: float = 200.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError("Detector parameters must be finite")
        if self.threshold_dbfs > 0 or self.frame_ms <= 0 or self.min_sound_ms <= 0:
            raise ValueError("Require threshold <= 0 dBFS and positive durations")


def detect_sound(audio, sample_rate, config=DetectorConfig()):
    """Return sustained RMS-active intervals in seconds; no semantic VAD.

    Take maximum channel RMS so stereo phase cancellation cannot hide sound.
    Do not bridge quiet frames. Retain original onset after duration filtering.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or not audio.shape[1] or sample_rate <= 0:
        raise ValueError("Expected audio [samples, channels] and positive sample rate")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    frame = max(1, round(sample_rate * config.frame_ms / 1000))
    threshold = 10 ** (config.threshold_dbfs / 20)
    intervals = []
    onset = None
    for start in range(0, len(audio), frame):
        chunk = audio[start:start + frame]
        active = float(np.sqrt(np.mean(chunk * chunk, axis=0)).max()) >= threshold
        if active and onset is None:
            onset = start
        if not active and onset is not None:
            if (start - onset) * 1000 / sample_rate + 1e-9 >= config.min_sound_ms:
                intervals.append((onset / sample_rate, start / sample_rate))
            onset = None
    if onset is not None and (len(audio) - onset) * 1000 / sample_rate + 1e-9 >= config.min_sound_ms:
        intervals.append((onset / sample_rate, len(audio) / sample_rate))
    return intervals


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def score_event(event, meta, intervals):
    """Intervals are sustained sound on the aligned input-audio timeline."""
    if event.get("dataset_schema_version") != "2.0" or event.get("legacy_format"):
        raise ValueError("Only explicit non-legacy schema 2.0 events are supported")
    condition = event["addressee_type"]
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    expected = "WAIT" if condition == "SPECIFIC_HUMAN" else "START"
    if event["expected_action"] != expected or event.get("agent_state") != "SILENT":
        raise ValueError("Event does not match the v2 silent-agent START/WAIT contract")
    start = _finite(event["aligned_evaluation_window_start"], "window start")
    end = _finite(event["aligned_evaluation_window_end"], "window end")
    target = _finite(event["aligned_target_time"], "target end")
    context = _finite(event["context_start"], "context start")
    if not 0 <= context <= start < end or abs(target - start) > 1e-4:
        raise ValueError("Invalid aligned context/target/window boundaries")
    if meta["dialogue_id"] != event["dialogue_id"]:
        raise ValueError("Metadata and event dialogue_id differ")
    if meta.get("dataset_schema_version") != "2.0":
        raise ValueError("Metadata must use schema 2.0")
    group = event.get("minimal_group_id")
    if group != meta.get("minimal_group_id"):
        raise ValueError("Metadata and event minimal_group_id differ")
    if meta.get("triplet_role", condition) != condition:
        raise ValueError("Metadata triplet_role disagrees with event")
    humans = meta["num_humans"]
    if type(humans) is not int or humans < 1:
        raise ValueError("num_humans must be a positive integer")
    complete = meta.get("is_complete_triplet_member", False)
    if type(complete) is not bool or (complete and not group):
        raise ValueError("Complete triplet membership needs a boolean flag and group ID")
    for a, b in intervals:
        if not math.isfinite(a) or not math.isfinite(b) or a >= b:
            raise ValueError("Invalid sound interval")
    premature = any(a < start and b > context for a, b in intervals)
    window_sound = any(a < end and b > start for a, b in intervals)
    onsets = [a for a, b in intervals if start <= a < end]
    latency = min(onsets) - start if expected == "START" and onsets and not premature else None
    predicted = "START" if window_sound else "WAIT"
    correct = predicted == expected
    return {
        "event_id": event["event_id"], "dialogue_id": event["dialogue_id"],
        "minimal_group_id": group, "is_complete_triplet_member": complete,
        "condition": condition, "num_humans": humans,
        "expected_action": expected, "predicted_action": predicted,
        "window_correct": correct, "premature_start": premature,
        "event_success": correct and not premature,
        "response_latency_seconds": latency,
        "context_start": context, "window_start": start, "window_end": end,
        "sound_intervals": [[a, b] for a, b in intervals],
    }


def _ratio(n, d):
    return n / d if d else None


def summarize(rows):
    positives = [r for r in rows if r["expected_action"] == "START"]
    negatives = [r for r in rows if r["expected_action"] == "WAIT"]
    tp = sum(r["predicted_action"] == "START" for r in positives)
    fp = sum(r["predicted_action"] == "START" for r in negatives)
    fn, tn = len(positives) - tp, len(negatives) - fp
    latencies = [r["response_latency_seconds"] for r in positives
                 if r["response_latency_seconds"] is not None]
    f1_start = _ratio(2 * tp, 2 * tp + fp + fn) if positives else None
    f1_wait = _ratio(2 * tn, 2 * tn + fp + fn) if negatives else None
    return {
        "n": len(rows), "n_start": len(positives), "n_wait": len(negatives),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "macro_f1": (f1_start + f1_wait) / 2 if positives and negatives else None,
        "f1_start": f1_start, "f1_wait": f1_wait,
        "response_rate": _ratio(tp, len(positives)),
        "false_takeover_rate": _ratio(fp, len(negatives)),
        "premature_start_rate": _ratio(sum(r["premature_start"] for r in rows), len(rows)),
        "event_success_rate": _ratio(sum(r["event_success"] for r in rows), len(rows)),
        "latency": {
            "n": len(latencies),
            "median_seconds": float(np.median(latencies)) if latencies else None,
            "p90_seconds": float(np.percentile(latencies, 90)) if latencies else None,
        },
        "timely_response_rate_at_seconds": {
            str(t): {
                "n_eligible": sum(r["window_end"] - r["window_start"] >= t for r in positives),
                "rate": _ratio(sum(r["response_latency_seconds"] is not None
                                   and r["response_latency_seconds"] <= t
                                   for r in positives if r["window_end"] - r["window_start"] >= t),
                               sum(r["window_end"] - r["window_start"] >= t for r in positives)),
            } for t in (1, 2, 5, 10)
        },
    }


def contrastive(rows):
    groups = {}
    for row in rows:
        if row["minimal_group_id"] and row["is_complete_triplet_member"]:
            group = groups.setdefault(row["minimal_group_id"], {})
            if row["condition"] in group:
                raise ValueError(f"Duplicate condition in group {row['minimal_group_id']}")
            group[row["condition"]] = row
    pairs, triplets = [], []
    for group in groups.values():
        if len({r["num_humans"] for r in group.values()}) > 1:
            raise ValueError("Group contains inconsistent num_humans")
        if {"AI_AGENT", "SPECIFIC_HUMAN"} <= group.keys():
            pairs.append(all(group[k]["event_success"] for k in ("AI_AGENT", "SPECIFIC_HUMAN")))
        if set(CONDITIONS) <= group.keys():
            triplets.append(all(group[k]["event_success"] for k in CONDITIONS))
    return {
        "core_pair": {"n": len(pairs), "n_correct": sum(pairs), "accuracy": _ratio(sum(pairs), len(pairs))},
        "strict_triplet": {"n": len(triplets), "n_correct": sum(triplets), "accuracy": _ratio(sum(triplets), len(triplets))},
        "n_groups_seen": len(groups), "n_incomplete_triplets": len(groups) - len(triplets),
        "n_excluded_standalone_events": sum(not r["is_complete_triplet_member"] for r in rows),
    }


def report(rows):
    everyone = [r for r in rows if r["condition"] == "EVERYONE"]
    es = summarize(everyone)
    return {
        "core": summarize([r for r in rows if r["condition"] != "EVERYONE"]),
        "everyone": {
            "n": len(everyone), "participation_rate": es["response_rate"],
            "latency": es["latency"], "premature_start_rate": es["premature_start_rate"],
        },
        "strict_original": summarize(rows),
        "contrastive": contrastive(rows),
        "per_condition": {c: summarize([r for r in rows if r["condition"] == c]) for c in CONDITIONS},
    }


def evaluate_manifest(manifest_path, config=DetectorConfig()):
    manifest_path = Path(manifest_path).resolve()
    entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError("Manifest is empty")
    rows, seen = [], set()
    def resolve(path):
        return (manifest_path.parent / path).resolve()
    for i, entry in enumerate(entries, 1):
        try:
            events = [json.loads(line) for line in resolve(entry["events"]).read_text().splitlines() if line.strip()]
            if "event_id" in entry:
                events = [e for e in events if e["event_id"] == entry["event_id"]]
            if len(events) != 1:
                raise ValueError("Select exactly one event per manifest row using event_id")
            event = events[0]
            if event["event_id"] in seen:
                raise ValueError(f"Duplicate event_id: {event['event_id']}")
            seen.add(event["event_id"])
            meta = json.loads(resolve(entry["meta"]).read_text())
            audio_path = resolve(entry["output_audio"])
            audio, sr = sf.read(audio_path, dtype="float64", always_2d=True)
            offset = _finite(entry["audio_start_time"], "audio_start_time")
            row = score_event(event, meta, [])
            # Missing recordings must never be silently interpreted as WAIT.
            if offset > row["context_start"] + 1e-6 or offset + len(audio) / sr < row["window_end"] - 1e-6:
                raise ValueError("Output audio must cover context_start through evaluation_window_end, including silence")
            # Fixed observation horizon: post-deadline audio cannot qualify
            # a short pre-deadline sound, regardless of recording length.
            first = max(0, math.ceil((row["context_start"] - offset) * sr - 1e-7))
            last = min(len(audio), math.ceil((row["window_end"] - offset) * sr - 1e-7))
            observed_offset = offset + first / sr
            intervals = [(a + observed_offset, b + observed_offset)
                         for a, b in detect_sound(audio[first:last], sr, config)]
            row = score_event(event, meta, intervals)
            row["output_audio"] = str(audio_path)
            row["audio_start_time"] = offset
            rows.append(row)
        except (KeyError, ValueError, TypeError, OSError, RuntimeError) as exc:
            raise ValueError(f"Manifest row {i}: {exc}") from exc
    result = report(rows)
    result["per_num_humans"] = {
        str(n): report([r for r in rows if r["num_humans"] == n])
        for n in sorted({r["num_humans"] for r in rows})
    }
    result["protocol"] = {
        "version": "1.0", "manifest": str(manifest_path), "detector": asdict(config),
        "prediction": "sustained sound overlapping [window_start, window_end)",
        "premature_start": "sustained sound overlapping [context_start, window_start)",
        "event_success": "correct window decision and no premature start",
        "latency": "START only; onset inside window and no premature start",
        "everyone_policy": "descriptive participation; START in strict_original",
    }
    result["events"] = rows
    return result


def markdown_report(result):
    def fmt(x):
        return "N/A" if x is None else f"{x:.4f}"
    lines = ["# Turn-taking evaluation", "", "Rates are fractions (0–1); latency is seconds.", "",
             "| Scope | N | Macro-F1 | Response / participation | False takeover | Premature start | Latency median / P90 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    scopes = [("Core", result["core"]), ("Strict original", result["strict_original"])]
    scopes += list(result["per_condition"].items())
    for name, s in scopes:
        lines.append(f"| {name} | {s['n']} | {fmt(s['macro_f1'])} | {fmt(s['response_rate'])} | "
                     f"{fmt(s['false_takeover_rate'])} | {fmt(s['premature_start_rate'])} | "
                     f"{fmt(s['latency']['median_seconds'])} / {fmt(s['latency']['p90_seconds'])} |")
    lines += ["", "EVERYONE participation has no preferred direction unless a participation policy was fixed in advance.", ""]
    for key in ("core_pair", "strict_triplet"):
        s = result["contrastive"][key]
        lines.append(f"- {key}: {fmt(s['accuracy'])} ({s['n_correct']}/{s['n']})")
    lines += ["", "| Humans | Core N | Core Macro-F1 | EVERYONE participation | Core pair accuracy | Strict triplet accuracy |",
              "|---|---:|---:|---:|---:|---:|"]
    for n, sub in result.get("per_num_humans", {}).items():
        lines.append(f"| {n} | {sub['core']['n']} | {fmt(sub['core']['macro_f1'])} | "
                     f"{fmt(sub['everyone']['participation_rate'])} | "
                     f"{fmt(sub['contrastive']['core_pair']['accuracy'])} | "
                     f"{fmt(sub['contrastive']['strict_triplet']['accuracy'])} |")
    lines += ["", "Pair/triplet success also requires no premature start. See JSON for event diagnostics, latency counts, time thresholds and human-count breakdowns.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold-dbfs", type=float, default=-40)
    parser.add_argument("--frame-ms", type=float, default=10)
    parser.add_argument("--min-sound-ms", type=float, default=200)
    args = parser.parse_args()
    try:
        config = DetectorConfig(args.threshold_dbfs, args.frame_ms, args.min_sound_ms)
        result = evaluate_manifest(args.manifest, config)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.out_dir / "report.md").write_text(markdown_report(result))
    print(f"Evaluated {len(result['events'])} events: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
