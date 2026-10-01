#!/usr/bin/env python
"""Aggregate evaluator reports from independent, namespaced stochastic runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from turn_taking_eval.evaluation import report, markdown_report
except ModuleNotFoundError:  # direct execution: ``python scripts/...py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from turn_taking_eval.evaluation import report, markdown_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if len(args.run_dirs) < 2:
        raise SystemExit("aggregate at least two independent run directories")
    loaded = []
    rows = []
    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        result = json.loads((run_dir / "report" / "report.json").read_text())
        namespace = run_dir.name
        per_run_rows = []
        for original in result["events"]:
            row = dict(original)
            row["event_id"] = f"{namespace}::{row['event_id']}"
            row["dialogue_id"] = f"{namespace}::{row['dialogue_id']}"
            row["minimal_group_id"] = f"{namespace}::{row['minimal_group_id']}"
            rows.append(row)
            per_run_rows.append(row)
        loaded.append({
            "run_dir": str(run_dir),
            "n_events": len(per_run_rows),
            "core": result["core"],
            "everyone": result["everyone"],
            "strict_original": result["strict_original"],
            "contrastive": result["contrastive"],
        })
    aggregate = report(rows)
    aggregate["per_num_humans"] = {
        str(n): report([row for row in rows if row["num_humans"] == n])
        for n in sorted({row["num_humans"] for row in rows})
    }
    aggregate["protocol"] = {
        "version": "1.0",
        "aggregation": "event rows from independent reports with run-namespaced group IDs",
        "detector": json.loads(
            ((args.run_dirs[0].resolve() / "report" / "report.json").read_text())
        )["protocol"]["detector"],
    }
    aggregate["source_runs"] = loaded
    aggregate["events"] = rows
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (out_dir / "report.md").write_text(markdown_report(aggregate) + "\n")
    print(f"Aggregated {len(rows)} events from {len(loaded)} runs: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
