"""Deterministic audio and metric regression tests; no model/TTS required."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turn_taking_eval.evaluation import (
    DetectorConfig, contrastive, detect_sound, evaluate_manifest, report, score_event, summarize,
)


def fixture(condition="AI_AGENT"):
    event = {
        "dataset_schema_version": "2.0", "legacy_format": False,
        "event_id": condition, "dialogue_id": condition,
        "addressee_type": condition, "expected_action": "WAIT" if condition == "SPECIFIC_HUMAN" else "START",
        "agent_state": "SILENT", "aligned_target_time": 2.0,
        "aligned_evaluation_window_start": 2.0, "aligned_evaluation_window_end": 12.0,
        "context_start": 0.0, "minimal_group_id": "g1",
    }
    meta = {
        "dataset_schema_version": "2.0", "dialogue_id": condition,
        "minimal_group_id": "g1", "num_humans": 3,
        "triplet_role": condition, "is_complete_triplet_member": True,
    }
    return event, meta


def row(condition, intervals):
    return score_event(*fixture(condition), intervals)


class EvaluationTests(unittest.TestCase):
    def test_energy_duration_and_phase(self):
        x = np.zeros(1000)
        x[100:290] = .1  # 190ms excluded
        x[400:600] = .1  # 200ms included
        stereo = np.stack([x, -x], axis=1)
        self.assertEqual(detect_sound(stereo, 1000), [(0.4, 0.6)])
        self.assertEqual(detect_sound(np.zeros(1000), 1000), [])
        self.assertEqual(detect_sound(np.full(1000, .001), 1000), [])

    def test_exact_duration_partial_frame(self):
        x = np.full(205, .1)
        self.assertEqual(detect_sound(x, 1000), [(0, .205)])
        self.assertEqual(detect_sound(x[:195], 1000), [])

    def test_invalid_detector(self):
        for kwargs in ({"threshold_dbfs": float("nan")}, {"min_sound_ms": 0}, {"frame_ms": -1}):
            with self.assertRaises(ValueError):
                DetectorConfig(**kwargs)
        with self.assertRaises(ValueError):
            detect_sound([float("nan")], 1000)

    def test_everyone_optional_core_and_strict(self):
        rows = [row("AI_AGENT", [(2.5, 3)]), row("EVERYONE", []), row("SPECIFIC_HUMAN", [])]
        result = report(rows)
        self.assertEqual(result["core"]["macro_f1"], 1)
        self.assertEqual(result["everyone"]["participation_rate"], 0)
        self.assertAlmostEqual(result["strict_original"]["macro_f1"], 2 / 3)
        self.assertEqual(result["contrastive"]["core_pair"]["accuracy"], 1)
        self.assertEqual(result["contrastive"]["strict_triplet"]["accuracy"], 0)

    def test_all_start_all_wait(self):
        for intervals, response, takeover in (([], 0, 0), ([(2.1, 3)], 1, 1)):
            result = summarize([row(c, intervals) for c in ("AI_AGENT", "SPECIFIC_HUMAN")])
            self.assertEqual(result["response_rate"], response)
            self.assertEqual(result["false_takeover_rate"], takeover)
            self.assertAlmostEqual(result["macro_f1"], 1 / 3)

    def test_early_speech_crossing_target(self):
        r = row("AI_AGENT", [(1.8, 2.5), (3, 4)])
        self.assertEqual(r["predicted_action"], "START")
        self.assertTrue(r["premature_start"])
        self.assertFalse(r["event_success"])
        self.assertIsNone(r["response_latency_seconds"])
        self.assertEqual(contrastive([r, row("SPECIFIC_HUMAN", [])])["core_pair"]["accuracy"], 0)

    def test_early_only_wait_failure(self):
        r = row("SPECIFIC_HUMAN", [(1, 1.5)])
        self.assertTrue(r["window_correct"])
        self.assertFalse(r["event_success"])
        self.assertEqual(summarize([r])["premature_start_rate"], 1)

    def test_half_open_boundaries(self):
        self.assertEqual(row("AI_AGENT", [(2, 2.5)])["response_latency_seconds"], 0)
        self.assertEqual(row("AI_AGENT", [(12, 12.5)])["predicted_action"], "WAIT")
        self.assertEqual(row("AI_AGENT", [(1.5, 2)])["predicted_action"], "WAIT")

    def test_latency_missing_responses_and_short_windows(self):
        fast = row("AI_AGENT", [(2.5, 3)])
        slow = row("AI_AGENT", [(5, 6)])
        missed = row("AI_AGENT", [])
        s = summarize([fast, slow, missed])
        self.assertEqual(s["latency"]["n"], 2)
        self.assertEqual(s["latency"]["median_seconds"], 1.75)
        self.assertAlmostEqual(s["latency"]["p90_seconds"], 2.75)
        self.assertEqual(s["timely_response_rate_at_seconds"]["1"]["rate"], 1 / 3)
        fast["window_end"] = 4
        self.assertIsNone(summarize([fast])["timely_response_rate_at_seconds"]["5"]["rate"])

    def test_empty_and_single_class(self):
        self.assertIsNone(summarize([])["macro_f1"])
        self.assertIsNone(summarize([row("AI_AGENT", [])])["macro_f1"])
        self.assertIsNone(report([])["contrastive"]["strict_triplet"]["accuracy"])

    def test_group_integrity(self):
        a, h = row("AI_AGENT", []), row("SPECIFIC_HUMAN", [])
        self.assertEqual(contrastive([a, h])["n_incomplete_triplets"], 1)
        with self.assertRaises(ValueError):
            contrastive([a, a])
        h["num_humans"] = 4
        with self.assertRaises(ValueError):
            contrastive([a, h])
        a["is_complete_triplet_member"] = False
        self.assertEqual(contrastive([a])["core_pair"]["n"], 0)

    def test_reject_legacy_and_wrong_alignment(self):
        event, meta = fixture()
        for change in ({"legacy_format": True}, {"dataset_schema_version": "1.0"},
                       {"aligned_target_time": 8}, {"expected_action": "WAIT"}):
            with self.assertRaises(ValueError):
                score_event(dict(event, **change), meta, [])
        del event["aligned_evaluation_window_start"]
        with self.assertRaises(KeyError):
            score_event(event, meta, [])

    def test_manifest_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            entries = []
            for c in ("AI_AGENT", "EVERYONE", "SPECIFIC_HUMAN"):
                event, meta = fixture(c)
                (root / f"{c}.jsonl").write_text(json.dumps(event) + "\n")
                (root / f"{c}.json").write_text(json.dumps(meta))
                x = np.zeros(12000)
                if c == "AI_AGENT":
                    x[2500:3000] = .1
                sf.write(root / f"{c}.wav", x, 1000)
                entries.append({"events": f"{c}.jsonl", "meta": f"{c}.json",
                                "output_audio": f"{c}.wav", "audio_start_time": 0})
            manifest = root / "manifest.jsonl"
            def write(items):
                manifest.write_text("\n".join(json.dumps(x) for x in items) + "\n")
            write(entries)
            result = evaluate_manifest(manifest)
            self.assertEqual(result["core"]["macro_f1"], 1)
            self.assertIn("3", result["per_num_humans"])
            proc = subprocess.run([sys.executable, "-m", "turn_taking_eval.evaluation", str(manifest),
                                   "--out-dir", str(root / "out")],
                                  cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((root / "out/report.md").exists())
            self.assertEqual(json.loads((root / "out/report.json").read_text())["core"]["macro_f1"], 1)
            # Audio beyond the deadline must not make a 100ms run eligible.
            x = np.zeros(13000)
            x[11900:12500] = .1
            sf.write(root / "AI_AGENT.wav", x, 1000)
            self.assertEqual(evaluate_manifest(manifest)["core"]["response_rate"], 0)
            # Explicit negative recording offset is supported and preserved.
            x = np.zeros(13000)
            x[3500:4000] = .1
            sf.write(root / "AI_AGENT.wav", x, 1000)
            offset_entries = copy.deepcopy(entries)
            offset_entries[0]["audio_start_time"] = -1
            write(offset_entries)
            self.assertEqual(evaluate_manifest(manifest)["events"][0]["response_latency_seconds"], .5)
            write(entries + [entries[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate event_id"):
                evaluate_manifest(manifest)
            shifted = copy.deepcopy(entries)
            shifted[0]["audio_start_time"] = 1
            write(shifted)
            with self.assertRaisesRegex(ValueError, "cover context_start"):
                evaluate_manifest(manifest)
            write(entries)
            sf.write(root / "AI_AGENT.wav", np.zeros(1000), 1000)
            with self.assertRaisesRegex(ValueError, "cover context_start"):
                evaluate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
