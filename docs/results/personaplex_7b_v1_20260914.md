> Imported from `Audio_synthesize` (branch `feature/qwen-local-backend`,
> `run_eval_turn_taking_personaplex_recovery_20260914/REPORT.md`). Artifact
> paths below refer to that working tree; script names map to this repo as:
> `scripts/run_turn_taking_personaplex_official.py` → `runners/personaplex/run_official.py`,
> `scripts/run_personaplex_official_control.py` → `runners/personaplex/run_control.py`,
> `scripts/*_turn_taking_*.py` → `scripts/` (see README).

# PersonaPlex turn-taking evaluation recovery

Date: 2026-09-14 (Asia/Taipei)

## Executive verdict

The previous run_eval_turn_taking_personaplex_stock_20260913 result is invalid
as a stock PersonaPlex baseline. It used a Moshiko proxy checkpoint, a
.6-to-.7 weight-copy shim, fake config.json bookkeeping, and local Roger voice
audio rather than an official packaged voice.

The recovered run uses NVIDIA's official moshi.offline CLI and the exact gated
nvidia/personaplex-7b-v1 snapshot. The official positive control passes 5/5,
but the model fails the schema-v2 addressee-aware behavior gate on 90 events:
it speaks before the target in every event, including the matched short-neutral
context pilot. This is normal full-duplex participation without reliable group
addressee control, not a checkpoint-load failure.

## Final decision

Dataset production: GO

Model behavioral readiness: NO-GO

Dataset/TTS/semantic gates remain separate from PersonaPlex behavior. No
dataset labels, evaluator thresholds, or production pipeline were changed.

## Repository and environment

- Branch: feature/qwen-local-backend
- HEAD: 3cb17f117d4a775db070e0cdd352ad2c09ca81d4 (newer than the prompt minimum)
- Official source: /work/b12901015/personaplex
- Official source commit: 3428dfd95309a7f3c84fd93259ded0f810d1ff91
- Official source tree was clean.
- Runtime: Python 3.12.14; torch 2.4.1+cu124; CUDA 12.4; NumPy 2.1.3;
  safetensors 0.4.5; huggingface-hub 0.24.7; sentencepiece 0.2.0;
  sphn 0.1.12; benchmark GPU NVIDIA H200.
- Hugging Face authentication as Ryantaku succeeded and the gated assets
  downloaded successfully.
- Final test suite: 99 passed.

## Official checkpoint provenance

Pinned revision:
fdaf4090a61cb315c138a1faee287ffd6c716309

| Asset | Size | SHA-256 |
|---|---:|---|
| config.json | 56 B | 2d1cfef43c2cec307f2a3c7f7835b69a87c685261e785549e2c6558f811e1b08 |
| model.safetensors | 16,742,874,000 B | db1290db583cdaa6cb4de444ed279e0b586ca2a372b41434b07a7461c8c0e2f4 |
| tokenizer-e351c8d8-checkpoint125.safetensors | 384,644,900 B | 09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50 |
| tokenizer_spm_32k_3.model | 552,778 B | 78d4336533ddc26f9acf7250d7fb83492152196c6ea4212c841df76933f18d2d |
| voices.tgz | 6,095,521 B | 8564e9ca7a06ca723b07c3a77c623f0faa5937d04b2647b3a727b06c5ca0b7bb |

The official voice was NATF2.pt. Every run config retains the same hashes
under checkpoint_provenance. No proxy weight-copy shim was used.

## Root-cause table

| Class | Finding | Evidence |
|---|---|---|
| Proven | Previous LM was not official PersonaPlex. | Its config points to moshi_proxy_no_human_20260819; the model has 355 keys and only depformer_emb.0 through .6. |
| Proven | Previous loader fabricated compatibility state. | It copied .6 into missing .7 and returned a tokenizer path for the unused config lookup. |
| Proven | Previous voice was not packaged PersonaPlex. | It used local_voices/01_Roger.mp3 rather than NATF2.pt. |
| Proven | Official runtime is healthy. | Official CLI produced five complete 40-second controls. |
| Proven | Strict early speech is pervasive. | Recovered benchmark premature start is 90/90; short-neutral pilot is 9/9. |
| Proven | Group discrimination is poor. | SPECIFIC_HUMAN false takeover is 28/30; core pair accuracy is 0/30. |
| Hypothesis | Mixed audio has no explicit speaker/addressee identity. | A two-party full-duplex model receiving one mixed user channel is consistent with this behavior. |
| Hypothesis | Prompt alone cannot enforce silence. | Official casual prompt and silent-group prompt have identical matched strict metrics. |
| Ruled out as primary | Official checkpoint/runtime incompatibility. | Positive control is 5/5 with plausible advice and exact duration. |
| Ruled out as primary | Future audio or labels leaked. | The official runner passes only the cropped WAV; run configs mark future audio, labels, and metadata exposure false. |
| Ruled out as primary | Timestamp-only detector artifact. | Inputs/outputs are mono 24 kHz, crops start at context_start and end at the deadline, and decoded tokens show active responses. |
| Not sufficient | Long natural preamble alone. | Short-neutral construction still has 9/9 premature starts. |

## Positive control

Artifacts:

- run_eval_personaplex_official_control_20260913/positive_control_result.json
- run_eval_personaplex_official_control_20260913/run_config.json
- trial WAV/JSON files under run_eval_personaplex_official_control_20260913/trial_*

Seeds 101, 202, 303, 404, and 505 all produced mono 24 kHz, exactly 40.0 s
outputs and non-empty decoded speech. Manual token review found coherent
rice/cooking advice in every trial. Required gate: at least 4/5. Observed:
5/5 valid and plausible.

## Controlled results

Detector: −40 dBFS, 10 ms frames, 200 ms sustained duration.

| Experiment | Events | Core Macro-F1 | AI_AGENT response | SPECIFIC_HUMAN false takeover | Premature | Core pair | Strict triplet |
|---|---:|---:|---:|---:|---:|---:|---:|
| Long original context, 10 groups x 3 seeds | 60 core / 90 total | 0.3939 | 29/30 = 0.9667 | 28/30 = 0.9333 | 90/90 = 1.0000 | 0/30 | 0/30 |
| Short neutral matched triplet, 3 seeds | 6 core / 9 total | 0.3333 | 3/3 = 1.0000 | 3/3 = 1.0000 | 9/9 = 1.0000 | 0/3 | 0/3 |
| Official casual prompt, matched group 1, 3 seeds | 6 core / 9 total | 0.3333 | 3/3 = 1.0000 | 3/3 = 1.0000 | 9/9 = 1.0000 | 0/3 | 0/3 |
| Silent-group prompt, matched group 1, 3 seeds | 6 core / 9 total | 0.3333 | 3/3 = 1.0000 | 3/3 = 1.0000 | 9/9 = 1.0000 | 0/3 | 0/3 |

The matched prompt comparison found identical predicted actions and premature
flags for all 9 paired events.

Per-seed long-context results:

| Seed | Core Macro-F1 | Core response | SPECIFIC_HUMAN false takeover | EVERYONE participation | Premature | Pair | Strict |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.3333 | 10/10 | 10/10 | 7/10 | 10/10 | 0/10 | 0/10 |
| 202 | 0.4357 | 10/10 | 9/10 | 9/10 | 10/10 | 0/10 | 0/10 |
| 303 | 0.4048 | 9/10 | 9/10 | 7/10 | 10/10 | 0/10 | 0/10 |

Aggregate confusion: core n=60 has TP=29, FP=28, FN=1, TN=2. Strict
original n=90 has TP=52, FP=28, FN=8, TN=2. Valid response latency is
0/30; median and P90 are N/A because every candidate response was premature.

## Timeline, audio, and error review

Reports and review artifacts:

- run_eval_turn_taking_personaplex_official_10triplets_seed101_20260913/report/report.json
- run_eval_turn_taking_personaplex_official_10triplets_seed202_20260913/report/report.json
- run_eval_turn_taking_personaplex_official_10triplets_seed303_20260913/report/report.json
- run_eval_turn_taking_personaplex_official_10triplets_3seeds_aggregate_20260914/report.json
- run_eval_turn_taking_personaplex_official_10triplets_3seeds_aggregate_20260914/manual_error_review.json

The error review enumerates all 90 events with output WAV, source transcript,
sustained intervals, first onset, decoded text, decision, and latency. Using a
fixed diagnostic initialization interval [0, 2.0) seconds:

| Diagnostic | Count |
|---|---:|
| Startup speech in [0, 2.0) | 86/90 |
| Context takeover after 2.0 s and before target | 90/90 |
| Speech crossing target boundary | 6/90 |
| Target-window START | 80/90 |
| Target-window WAIT | 10/90 |
| Valid response latency | 0/90 |

Representative references:

- Long AI_AGENT output: run_eval_turn_taking_personaplex_official_10triplets_seed101_20260913/tt_h3_ai_agent_0001/agent.wav; transcript: run_script_turn_taking_v2_delivery_canary_20260912/tt_h3_ai_agent_0001/transcript.txt; first sustained output 0.39 s.
- Long SPECIFIC_HUMAN output: run_eval_turn_taking_personaplex_official_10triplets_seed101_20260913/tt_h3_specific_human_0001/agent.wav; expected WAIT but predicted START with pre-target speech.
- Short-neutral SPECIFIC_HUMAN output: run_eval_turn_taking_personaplex_short_neutral_seed101_20260913/tt_h3_short_neutral_specific_human_0001/agent.wav; target boundary 4.82 s, first sustained output 0.39 s.

## Experimental confounds

run_eval_turn_taking_personaplex_prefix_audit_20260913.json shows that the
existing TTS triplet members are not byte-identical before target: exact
non-reference common-prefix matches are 0/20, all-prefix-identical groups are
0/10, mean common-prefix RMSE is 0.05203, and maximum target-time spread is
15.5539 s. These limit causal contrast, but do not explain away universal
strict prematurity; the short-neutral construction is a stronger diagnostic.

## Recommendation

Choose Outcome B, staged as follows:

1. Keep the causal audio stream feeding PersonaPlex, but gate audible output
   with a causal speaker/addressee and turn-taking classifier. It must not use
   labels, future audio, or post-window metadata. Measure controller latency,
   false opens, missed opens, and hidden-speech contamination of later state.
2. If the speech model itself must learn addressee-aware behavior, adapter-train
   or fine-tune with explicit speaker/addressee conditioning and schema-v2
   triplets. Preserve group splits and compare with this exact base checkpoint;
   shuffle/zero identity conditioning as an ablation.
3. Keep realistic human-to-human questions, stratified as a difficulty class.
   Do not relax strict metrics or remove difficult cases after seeing results.

## Code changes

Added the fail-closed official runner, official positive-control command,
pinned provenance/crop/output checks, explicit proxy-only legacy flag,
triplet aggregation, prefix audit, short-neutral construction, event-level
error review, and CPU regression tests. No changes were made to
/work/b12901015/mpmoshi or unrelated enrollment/named-addressee work.
Generated run directories and logs remain uncommitted.
