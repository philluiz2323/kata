# Evaluator Versioning

PromptForge promotes prompts based on measured benchmark results. That only
works if each result can be traced back to the evaluator and benchmark state
that produced it.

## What should be versioned

Every meaningful eval or challenge run should record:

- evaluator version
- prompt hashes
- task-pool fingerprint
- benchmark task ids
- challenge or eval run id

PromptForge now records this metadata in:

- `run_summary.json`
- `challenge_summary.json`
- `frontier.json`

In the intended deployment model, `frontier.json` and the benchmark task files
live in the benchmark registry repo identified by
`promptforge-benchmark-registry.json`, while run artifacts stay in `runs/`.

## Why it matters

A prompt improvement is only trustworthy when the measurement basis is stable.

If any of these change, the result may no longer be comparable:

- task definitions
- `checks.sh`
- allowed or forbidden path rules
- evaluator scoring rules
- frontier prompt state

This is why PromptForge treats benchmark provenance as part of the result, not
just background context.

## Intended promotion flow

1. Add or refine benchmark tasks based on real repo work.
2. Run the incumbent frontier on the primary pool.
3. Run the challenger on the same primary pool under the same conditions.
4. Retest the challenger on the holdout pool if it wins the primary pool.
5. Promote only when the recorded benchmark provenance still matches the
   expected frontier configuration.

## MVP boundary

The current MVP records version and fingerprint metadata, but it does not yet
enforce a full signed or locked promotion pipeline. Maintainers should still
review benchmark changes carefully before treating results as comparable over
time.
