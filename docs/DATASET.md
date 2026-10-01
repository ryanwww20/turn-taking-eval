# Dataset contract

This repository contains evaluation code only. The schema-v2 turn-taking
dataset is produced by the `Audio_synthesize` pipeline
(`production/turn_taking_v2/`) and is **not stored in git**: audio is roughly
9 MB per dialogue (~270 MB per 30 dialogues), while the text side is under
1 MB per 30 dialogues.

## Status

There is no frozen, versioned evaluation set yet. The PersonaPlex baseline in
[`results/personaplex_7b_v1_20260914.md`](results/personaplex_7b_v1_20260914.md)
used 10 triplets from the delivery canary
(`run_script_turn_taking_v2_delivery_canary_20260912` +
`run_audio_turn_taking_v2_delivery_canary_nongreedy_20260913`).

Planned: generate the production set with fixed seeds, pass
`validate_generation.py` / `validate_audio.py`, then publish it as a versioned
archive (e.g. `turn-taking-eval-v2.0`) outside git with a SHA-256 manifest.
Benchmark results must name the dataset version they were computed on.

## Expected layout

Place (or symlink) a dataset under `data/` (git-ignored):

```text
data/
  scripts/                                   # "scripts root"
    tt_h3_g0001_group.json                   # minimal triplet: members per role
    tt_h3_ai_agent_0001/
      turn_taking_meta.json                  # minimal_group_id, num_humans,
                                             # is_complete_triplet_member, ...
      transcript.txt
      script.jsonl
    tt_h3_everyone_0001/ ...
    tt_h3_specific_human_0001/ ...
  audio/                                     # "audio root"
    tt_h3_ai_agent_0001/
      aligned_turn_taking_events.jsonl       # one schema-v2 event
      aligned_script.jsonl
      audio_qwen3-tts-1.7b-clone.wav         # mono 24 kHz human mix (model input)
      audio_qwen3-tts-1.7b-clone_<SPK>.wav   # per-speaker stems
```

The scorer only reads `aligned_turn_taking_events.jsonl`, `turn_taking_meta.json`
and the model's output WAV. Runners additionally read `*_group.json` and the
input mix.

Required event fields (`dataset_schema_version: "2.0"`, `legacy_format: false`):
`event_id`, `dialogue_id`, `addressee_type` (`AI_AGENT` / `EVERYONE` /
`SPECIFIC_HUMAN`), `expected_action`, `minimal_group_id`, `context_start`,
`aligned_evaluation_window_start`, `aligned_evaluation_window_end`.
See [`EVALUATION.md`](EVALUATION.md) for how they are scored.
