# turn-taking-eval

Audio-based evaluation of whether a **silent AI agent in a multi-party
conversation** correctly decides to START speaking or WAIT, depending on who
the last speaker addressed.

| Addressee of the target utterance | Expected |
|---|---|
| `AI_AGENT` ("AI Agent, what time…?") | START |
| `EVERYONE` ("What does everyone think…?") | START (reported separately as participation) |
| `SPECIFIC_HUMAN` ("Brian, what time…?") | WAIT |

The model hears the conversation up to the end of a 10 s evaluation window; we
record its agent-only output audio and score it with an energy detector
(−40 dBFS, 10 ms frames, ≥200 ms sustained).

Protocol and metric definitions: [`docs/EVALUATION.md`](docs/EVALUATION.md).
Dataset layout and status: [`docs/DATASET.md`](docs/DATASET.md).

## Install

```bash
pip install -e '.[test]'
python -m pytest
```

Model runners have their own requirements (PersonaPlex: `torch` plus a clean
checkout of [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) and
the gated `nvidia/personaplex-7b-v1` snapshot).

## Workflow

1. **Run a model** → agent-only WAVs + `manifest.jsonl`.
   ```bash
   python runners/personaplex/run_official.py \
     --scripts-root data/scripts --audio-root data/audio \
     --out-root runs/personaplex_seed101 \
     --checkpoint-dir <hf-cache>/snapshots/<revision> --revision <revision> \
     --official-source /path/to/personaplex --seed 101 --max-groups 10
   ```
2. **Score**
   ```bash
   python -m turn_taking_eval.evaluation runs/personaplex_seed101/manifest.jsonl \
     --out-dir runs/personaplex_seed101/report
   ```
3. **Aggregate seeds / analyse** (optional).
   ```bash
   python scripts/aggregate_reports.py runs/personaplex_seed{101,202,303} --out-dir runs/personaplex_3seeds
   ```

Any other model can be evaluated by producing the same manifest format
(see `docs/EVALUATION.md`).

## Layout

| Path | Purpose |
|---|---|
| `turn_taking_eval/evaluation.py` | Scorer: manifest → `report.json` / `report.md` |
| `runners/personaplex/run_official.py` | Official PersonaPlex via unmodified `moshi.offline`; records checkpoint hashes |
| `runners/personaplex/run_control.py` | Positive control (≤5 trials) that the runtime works at all |
| `scripts/aggregate_reports.py` | Merge per-seed reports with namespaced group ids |
| `scripts/compare_prompt_runs.py` | Paired comparison of two prompt variants on matched events |
| `scripts/audit_outputs.py` | Per-event timing + decoded-text review |
| `scripts/audit_prefixes.py` | Pre-target waveform differences within triplets (confound check) |
| `scripts/build_short_neutral_triplet.py` | Diagnostic short-neutral-context triplet |
| `docs/results/` | Recorded baselines |

## Results so far

| Model | Events | Core Macro-F1 | Premature start | Core pair | Report |
|---|---:|---:|---:|---:|---|
| PersonaPlex-7B v1 (official, NATF2) | 90 | 0.394 | 90/90 | 0/30 | [2026-09-14](docs/results/personaplex_7b_v1_20260914.md) |

## Scope

START/WAIT for a silent agent only. YIELD/CONTINUE and response-content
quality are not evaluated.

Origin: extracted from `Audio_synthesize` (`feature/qwen-local-backend`).
