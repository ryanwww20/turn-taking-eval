#!/usr/bin/env python
"""Compare matched current and official-prompt turn-taking runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

try:
    from turn_taking_eval.evaluation import report
except ModuleNotFoundError:  # direct execution: ``python scripts/...py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from turn_taking_eval.evaluation import report


def seed_of(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot infer seed from {path}")
    return int(match.group(1))


def namespaced_rows(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        result = json.loads((run_dir / "report" / "report.json").read_text())
        ns = run_dir.name
        for original in result["events"]:
            row = dict(original)
            row["event_id"] = f"{ns}::{row['event_id']}"
            row["dialogue_id"] = f"{ns}::{row['dialogue_id']}"
            row["minimal_group_id"] = f"{ns}::{row['minimal_group_id']}"
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current", type=Path, nargs="+", required=True)
    ap.add_argument("--official", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group-id", default="tt_h3_g0001",
                    help="minimal group in --current runs matched by the --official runs")
    args = ap.parse_args()
    current = sorted((p.resolve() for p in args.current), key=seed_of)
    official = sorted((p.resolve() for p in args.official), key=seed_of)
    if [seed_of(p) for p in current] != [seed_of(p) for p in official]:
        raise SystemExit("current and official prompt runs must use the same seeds")
    pairs = []
    paired_current_rows = []
    paired_official_rows = []
    for current_dir, official_dir in zip(current, official):
        current_rows = [
            row for row in json.loads((current_dir / "report" / "report.json").read_text())["events"]
            if row["minimal_group_id"] == args.group_id
        ]
        official_rows = json.loads((official_dir / "report" / "report.json").read_text())["events"]
        current_by_condition = {row["condition"]: row for row in current_rows}
        official_by_condition = {row["condition"]: row for row in official_rows}
        for condition in ("AI_AGENT", "EVERYONE", "SPECIFIC_HUMAN"):
            c = current_by_condition[condition]
            o = official_by_condition[condition]
            c = dict(c)
            o = dict(o)
            c["minimal_group_id"] = f"{current_dir.name}::{c['minimal_group_id']}"
            o["minimal_group_id"] = f"{official_dir.name}::{o['minimal_group_id']}"
            c["event_id"] = f"{current_dir.name}::{c['event_id']}"
            o["event_id"] = f"{official_dir.name}::{o['event_id']}"
            paired_current_rows.append(c)
            paired_official_rows.append(o)
            pairs.append({
                "seed": seed_of(current_dir),
                "condition": condition,
                "current_predicted_action": c["predicted_action"],
                "official_prompt_predicted_action": o["predicted_action"],
                "current_premature_start": c["premature_start"],
                "official_prompt_premature_start": o["premature_start"],
                "current_sound_intervals": c["sound_intervals"],
                "official_prompt_sound_intervals": o["sound_intervals"],
            })
    current_rows = namespaced_rows(current)
    official_rows = namespaced_rows(official)
    payload = {
        "current_prompt": json.loads((current[0] / "run_config.json").read_text())["text_prompt"],
        "official_prompt": json.loads((official[0] / "run_config.json").read_text())["text_prompt"],
        "seeds": [seed_of(p) for p in current],
        "paired_events": len(pairs),
        "current_metrics": report(paired_current_rows),
        "official_prompt_metrics": report(paired_official_rows),
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote matched prompt comparison: {args.out}")


if __name__ == "__main__":
    main()
