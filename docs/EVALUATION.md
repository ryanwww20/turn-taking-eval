# Turn-taking evaluation protocol v1

This evaluator measures audible START/WAIT behavior on **aligned schema v2**
events. It does not run a dialogue model. Feed the context to your model in
streaming order and record its **agent-only output audio**, including silence,
from `context_start` through `aligned_evaluation_window_end`. Never feed future
continuation, expected labels, or metadata to the model before the decision.
Use the same agent identity, prompt, decoding setup, output gain and detector
settings across compared runs; record these alongside each manifest.

## Run

From the repository root, use an environment with `numpy` and `soundfile`
(`pip install -e .` if needed):

```bash
python -m turn_taking_eval.evaluation model_outputs/manifest.jsonl \
  --out-dir evaluation_results/model_a \
  --threshold-dbfs -40 --frame-ms 10 --min-sound-ms 200
```

One manifest JSON object per event, with paths relative to the manifest:

```json
{"events":"../data/audio/tt_h4_everyone_0001/aligned_turn_taking_events.jsonl","meta":"../data/scripts/tt_h4_everyone_0001/turn_taking_meta.json","output_audio":"tt_h4_everyone_0001.agent.wav","audio_start_time":0.0}
```

`output_audio` is the model output, **not** the synthesized human input/mix.
`audio_start_time` is the input-timeline time corresponding to output sample 0;
it is required, never inferred. Account for cropping/recording offsets, but do
not subtract model response delay. If the event file has multiple events, add
`"event_id":"..."` to select exactly one. Use a separate manifest per model,
persona, or stochastic repeat; do not pool duplicate events into one run.

Missing files, truncated recordings, missing aligned timestamps, legacy events,
inconsistent metadata and duplicate events fail evaluation. A silent response
must be a full-duration silent audio file, not a missing or empty file.
The evaluator checks coverage before scoring results are emitted. Detector
intervals use the recorded timeline; input/output must already be synchronized.

Outputs: `report.json` (all metrics, counts, protocol settings and per-event
diagnostics) and `report.md` (meeting summary). Report rates as fractions in
[0,1], latency in seconds. Undefined rates and two-class Macro-F1 when either
ground-truth class is absent are `null` / `N/A`, not zero.

## What counts as opening the mouth

For this first version, sustained audible energy is sufficient: no ASR, content
judge or speech classifier. Frame RMS must reach **−40 dBFS** for at least
**200 ms** continuously, using **10 ms** frames. These are configurable starting
defaults, not calibrated claims about human perception. Quiet frames break a
run; there is no gap filling. Use the maximum channel RMS for multichannel
outputs. Noise, humming and backchannels count if they pass the same detector.
Onset resolution is approximately one frame; qualification by duration retains
the original onset rather than adding 200 ms to latency.

The aligned post-target window is `[start, end)`. Sustained audio overlapping
this window gives predicted START; otherwise predicted WAIT. The duration rule
applies to the entire detected sound run within the observation horizon
`[context_start, end)`. Audio after the deadline cannot qualify a short sound
before the deadline, even if the recording continues longer. Audio overlapping
`[context_start, start)` is also a **premature start**, including sound that
continues into the evaluation window. Pre-context sound is otherwise ignored.

## Five primary measurements

1. **Core Macro-F1 ↑**: average START and WAIT F1 on `AI_AGENT` (START) and
   `SPECIFIC_HUMAN` (WAIT). Binary decisions refer to sound in the evaluation
   window. Confusion matrix uses START as the positive class.
2. **Response Rate ↑**: START predictions / expected START events (START recall).
   Core uses AI_AGENT only. A sound run crossing the target can count as window
   activity, so always read this together with premature-start diagnostics.
3. **False Takeover Rate ↓**: START predictions / expected WAIT events (FPR).
   This is specifically takeover in the post-target window. Earlier takeovers
   are covered by Premature-start Rate, even when the window itself is silent.
4. **Response Latency ↓**: first sound onset in the window minus aligned target
   end, for expected START events **with no premature start**. Report median,
   P90 and eligible response count. Silence and premature responses have no
   valid latency; they remain in response-rate denominators. Do not rank models
   by latency alone. `timely_response_rate_at_seconds` reports 1/2/5/10-second
   success fractions including failures in the denominator; each threshold
   includes only events with a window at least that long and reports its count.
5. **Core Pair Accuracy ↑**: fraction of eligible groups where AI_AGENT starts
   and SPECIFIC_HUMAN waits, **both without premature starts**. This stricter
   criterion prevents early interruptions from counting as group successes.

Additional diagnostics: `premature_start_rate` over all events in each scope,
and `event_success_rate` requiring the correct window decision and no premature
start. Early sounds are never successful low/negative latency responses.

## EVERYONE and original-spec reporting

The dataset's original label remains **EVERYONE → START**. No relabeling occurs.
Natural group participation can depend on an agent's participation policy, so
the main `everyone` block reports **Participation Rate (no ↑/↓)**, latency and
premature-start rate independently. Silence alone does not establish a persona.

`strict_original` includes all three conditions under the original labels.
**Strict Triplet Accuracy ↑** requires AI_AGENT START, EVERYONE START and
SPECIFIC_HUMAN WAIT, all without premature starts. EVERYONE summaries under
`per_condition` also use its original START label; they are distinct from the
policy-neutral participation interpretation.

All five measurements are available by condition and human count where defined.
Single-condition Macro-F1 is undefined because only one true class exists.
Compare personality prompts using fixed models and matched groups in separate
runs. Check whether EVERYONE participation changes without degrading the core.

## Group and dataset integrity

Group metrics require matching `minimal_group_id` and
`is_complete_triplet_member=true` from metadata. Standalone extras are excluded.
Core pairs need both AI_AGENT and SPECIFIC_HUMAN evaluated; strict triplets need
all three. Denominators and incomplete-group counts are explicit. Duplicated
roles or inconsistent human counts within an eligible group are errors.

Keep every minimal group in one train/dev/test split. If combining independently
generated batches whose group IDs collide, namespace IDs consistently before
evaluation. Dataset semantic/audio validation remains a separate prerequisite;
this evaluator checks scoring inputs, not whether authored utterances actually
satisfy their intended labels or share identical context. Do not mix legacy
~600ms WAIT windows with v2 windows. Compare runs on the same event set and
window durations, and retain detector settings with the results.

This protocol covers a silent agent deciding START/WAIT. YIELD/CONTINUE and
response-content quality need different data or additional evaluators.

## Tests

```bash
python -m pytest tests -v
```

Tests synthesize deterministic sound/silence arrays and temporary WAVs. They do
not represent a real model benchmark result.
