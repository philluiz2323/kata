# Contributing to PromptForge

PromptForge is an objective prompt-optimization repo. Contributions should make
the evaluator, benchmark packs, or prompt competition workflow more trustworthy
and more useful.

## Priorities

- Keep scoring objective and reproducible.
- Prefer pinned tasks and explicit checks over subjective judgment.
- Treat benchmark and evaluator correctness as higher priority than prompt style.
- Keep changes scoped and easy to audit.
- Preserve benchmark provenance so results stay comparable over time.

## Local checks

Run these before opening a PR:

```bash
uv run pytest
uv run ruff check
uv run python -m promptforge eval-pack validate --path <your-eval-pack>
```

If you change the benchmark runner or reporting logic, add or update tests.

## Benchmark packs

Eval-pack tasks should be based on real repo work where possible:

- real issues
- real PR-sized edits
- pinned commits
- explicit pass/fail checks
- explicit allowed and forbidden paths when task scope matters

Do not commit placeholder scaffold tasks as if they were live benchmarks.

Canonical benchmark packs should live in a benchmark registry repo that contains
`promptforge-benchmark-registry.json`, rather than in the main PromptForge repo.

## Prompt competition model

PromptForge uses three prompt roles:

- `baseline`: fixed generic control prompt
- `frontier`: current best verified prompt for a repo and mode
- `challenger`: new candidate prompt trying to replace the frontier

A challenger should only be promoted when it beats the frontier on the primary
pool and, when configured, on the holdout pool.

Benchmark changes should also preserve clear provenance:

- evaluator version
- prompt hashes
- task-pool fingerprints
- explicit task ids

## Scope guidance

Good contributions:

- stronger eval-pack checks
- better task coverage
- clearer baseline/frontier/challenger workflow
- safer reporting and anti-gaming logic
- better contributor and reviewer prompt initialization

Lower-priority contributions:

- broad prompt rewrites without benchmark evidence
- subjective prompt-style changes without measured improvement
- changes that treat prompt generation as the main product rather than as a
  source of frontier challengers
