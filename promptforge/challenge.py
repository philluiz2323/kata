from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from promptforge.benchmarks import resolve_eval_pack_path
from promptforge.eval_runner import EvalRunSummary, run_prompt_variants
from promptforge.frontier import (
    FrontierManifest,
    FrontierModeConfig,
    load_frontier_manifest,
)
from promptforge.provenance import EVALUATOR_VERSION, sha256_text, short_hash

PRIMARY_PROMOTION_MARGIN_POINTS = 3.0


@dataclass(frozen=True)
class ChallengePoolSummary:
    task_ids: list[str]
    eval_run_summary: str
    total_task_weight: float
    variant_successes: dict[str, int]
    variant_invalid_tasks: dict[str, int]
    variant_scores: dict[str, float]
    candidate_beats_frontier: bool
    candidate_score_delta: float


@dataclass(frozen=True)
class ChallengeSummary:
    schema_version: int
    run_id: str
    manifest_path: str
    mode: str
    evaluator_version: str
    baseline_prompt: str
    frontier_prompt: str
    candidate_prompt: str
    baseline_prompt_hash: str
    frontier_prompt_hash: str
    candidate_prompt_hash: str
    primary_pool_fingerprint: str | None
    holdout_pool_fingerprint: str | None
    promotion_margin_points: float
    created_at: str
    primary: ChallengePoolSummary
    holdout: ChallengePoolSummary | None
    promotion_ready: bool
    promotion_reason: str


def run_frontier_challenge(
    *,
    eval_pack_path: str,
    mode: str,
    candidate_prompt_path: str,
    agent_command: str,
    output_root: str | None = None,
    agent_timeout_seconds: int | None = None,
    checks_timeout_seconds: int | None = None,
) -> ChallengeSummary:
    eval_pack_root = resolve_eval_pack_path(eval_pack_path)
    manifest = load_frontier_manifest(eval_pack_path)
    mode_config = resolve_mode(manifest, mode)
    candidate_path = Path(candidate_prompt_path).expanduser().resolve()
    output_base = Path(output_root) if output_root else Path("runs")
    challenge_run_id = build_challenge_id(eval_pack_root.name, mode)
    challenge_root = output_base / challenge_run_id
    challenge_root.mkdir(parents=True, exist_ok=False)

    baseline_text = Path(mode_config.baseline_prompt).read_text(encoding="utf-8")
    frontier_text = Path(mode_config.frontier_prompt).read_text(encoding="utf-8")
    candidate_text = candidate_path.read_text(encoding="utf-8")
    evaluator_version = mode_config.evaluator_version or EVALUATOR_VERSION
    baseline_hash = mode_config.baseline_prompt_hash or sha256_text(baseline_text)
    frontier_hash = mode_config.frontier_prompt_hash or sha256_text(frontier_text)
    candidate_hash = sha256_text(candidate_text)

    primary_eval = run_prompt_variants(
        repo_ref=manifest.repo_ref,
        eval_pack_path=eval_pack_path,
        mode=mode,
        agent_command=agent_command,
        prompt_variants=[
            ("baseline", baseline_text),
            ("frontier", frontier_text),
            ("candidate", candidate_text),
        ],
        task_names=mode_config.primary_tasks,
        output_root=str(challenge_root / "primary"),
        run_label=f"{eval_pack_root.name}-{mode}-primary",
        run_kind="challenge-primary",
        metadata={
            "evaluator_version": evaluator_version,
            "pool_name": "primary",
            "pool_fingerprint": mode_config.primary_pool_fingerprint or "",
            "baseline_prompt_hash": baseline_hash,
            "frontier_prompt_hash": frontier_hash,
            "candidate_prompt_hash": candidate_hash,
        },
        agent_timeout_seconds=agent_timeout_seconds,
        checks_timeout_seconds=checks_timeout_seconds,
    )
    primary_summary = summarize_pool(primary_eval, mode_config.primary_tasks)

    holdout_summary: ChallengePoolSummary | None = None
    promotion_ready = False
    if primary_summary.candidate_beats_frontier and mode_config.holdout_tasks:
        holdout_eval = run_prompt_variants(
            repo_ref=manifest.repo_ref,
            eval_pack_path=eval_pack_path,
            mode=mode,
            agent_command=agent_command,
            prompt_variants=[
                ("baseline", baseline_text),
                ("frontier", frontier_text),
                ("candidate", candidate_text),
            ],
            task_names=mode_config.holdout_tasks,
            output_root=str(challenge_root / "holdout"),
            run_label=f"{eval_pack_root.name}-{mode}-holdout",
            run_kind="challenge-holdout",
            metadata={
                "evaluator_version": evaluator_version,
                "pool_name": "holdout",
                "pool_fingerprint": mode_config.holdout_pool_fingerprint or "",
                "baseline_prompt_hash": baseline_hash,
                "frontier_prompt_hash": frontier_hash,
                "candidate_prompt_hash": candidate_hash,
            },
            agent_timeout_seconds=agent_timeout_seconds,
            checks_timeout_seconds=checks_timeout_seconds,
        )
        holdout_summary = summarize_pool(holdout_eval, mode_config.holdout_tasks)
    promotion_ready, reason = evaluate_promotion(primary_summary, holdout_summary)
    summary = ChallengeSummary(
        schema_version=2,
        run_id=challenge_run_id,
        manifest_path=str(eval_pack_root / "frontier.json"),
        mode=mode,
        evaluator_version=evaluator_version,
        baseline_prompt=str(Path(mode_config.baseline_prompt).resolve()),
        frontier_prompt=str(Path(mode_config.frontier_prompt).resolve()),
        candidate_prompt=str(candidate_path),
        baseline_prompt_hash=baseline_hash,
        frontier_prompt_hash=frontier_hash,
        candidate_prompt_hash=candidate_hash,
        primary_pool_fingerprint=mode_config.primary_pool_fingerprint,
        holdout_pool_fingerprint=mode_config.holdout_pool_fingerprint,
        promotion_margin_points=PRIMARY_PROMOTION_MARGIN_POINTS,
        created_at=datetime.now(UTC).isoformat(),
        primary=primary_summary,
        holdout=holdout_summary,
        promotion_ready=promotion_ready,
        promotion_reason=reason,
    )
    write_challenge_summary(challenge_root / "challenge_summary.json", summary)
    return summary


def render_challenge_summary(summary: ChallengeSummary) -> str:
    lines: list[str] = []
    lines.append(f"Challenge run: {summary.run_id}")
    lines.append(f"Mode: {summary.mode}")
    lines.append(f"Manifest: `{summary.manifest_path}`")
    lines.append(f"Candidate prompt: `{summary.candidate_prompt}`")
    lines.append(f"Evaluator version: {summary.evaluator_version}")
    lines.append(f"Baseline prompt hash: {short_hash(summary.baseline_prompt_hash)}")
    lines.append(f"Frontier prompt hash: {short_hash(summary.frontier_prompt_hash)}")
    lines.append(f"Candidate prompt hash: {short_hash(summary.candidate_prompt_hash)}")
    if summary.primary_pool_fingerprint:
        lines.append(
            f"Primary pool fingerprint: {short_hash(summary.primary_pool_fingerprint)}"
        )
    if summary.holdout_pool_fingerprint:
        lines.append(
            f"Holdout pool fingerprint: {short_hash(summary.holdout_pool_fingerprint)}"
        )
    lines.append("")
    lines.append("Primary pool")
    lines.extend(render_pool(summary.primary))
    if summary.holdout is not None:
        lines.append("")
        lines.append("Holdout pool")
        lines.extend(render_pool(summary.holdout))
    lines.append("")
    lines.append(f"Promotion margin: {summary.promotion_margin_points:.1f} points")
    lines.append(f"Promotion ready: {'yes' if summary.promotion_ready else 'no'}")
    lines.append(f"Reason: {summary.promotion_reason}")
    return "\n".join(lines)


def load_challenge_summary(path: str) -> ChallengeSummary:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    holdout_payload = payload.get("holdout")
    return ChallengeSummary(
        schema_version=payload["schema_version"],
        run_id=payload["run_id"],
        manifest_path=payload["manifest_path"],
        mode=payload["mode"],
        evaluator_version=payload.get("evaluator_version", EVALUATOR_VERSION),
        baseline_prompt=payload["baseline_prompt"],
        frontier_prompt=payload["frontier_prompt"],
        candidate_prompt=payload["candidate_prompt"],
        baseline_prompt_hash=payload.get("baseline_prompt_hash", ""),
        frontier_prompt_hash=payload.get("frontier_prompt_hash", ""),
        candidate_prompt_hash=payload.get("candidate_prompt_hash", ""),
        primary_pool_fingerprint=payload.get("primary_pool_fingerprint"),
        holdout_pool_fingerprint=payload.get("holdout_pool_fingerprint"),
        promotion_margin_points=payload.get(
            "promotion_margin_points", PRIMARY_PROMOTION_MARGIN_POINTS
        ),
        created_at=payload["created_at"],
        primary=parse_challenge_pool(payload["primary"]),
        holdout=parse_challenge_pool(holdout_payload) if holdout_payload else None,
        promotion_ready=payload["promotion_ready"],
        promotion_reason=payload["promotion_reason"],
    )


def parse_challenge_pool(payload: dict[str, object]) -> ChallengePoolSummary:
    variant_scores = payload.get("variant_scores") or {}
    candidate_score = float(variant_scores.get("candidate", 0.0)) if variant_scores else 0.0
    frontier_score = float(variant_scores.get("frontier", 0.0)) if variant_scores else 0.0
    return ChallengePoolSummary(
        task_ids=list(payload["task_ids"]),
        eval_run_summary=str(payload["eval_run_summary"]),
        total_task_weight=float(payload.get("total_task_weight", len(payload["task_ids"]))),
        variant_successes=dict(payload.get("variant_successes") or {}),
        variant_invalid_tasks=dict(payload.get("variant_invalid_tasks") or {}),
        variant_scores={name: float(score) for name, score in variant_scores.items()},
        candidate_beats_frontier=bool(
            payload.get("candidate_beats_frontier", candidate_score > frontier_score)
        ),
        candidate_score_delta=float(
            payload.get("candidate_score_delta", round(candidate_score - frontier_score, 2))
        ),
    )


def summarize_pool(summary: EvalRunSummary, task_ids: list[str]) -> ChallengePoolSummary:
    successes = count_variant_successes(summary)
    invalid_tasks = count_variant_invalid_tasks(summary)
    scores = score_variants(summary)
    total_weight = sum(task.task_weight for task in summary.tasks)
    candidate_score = scores.get("candidate", 0.0)
    frontier_score = scores.get("frontier", 0.0)
    return ChallengePoolSummary(
        task_ids=task_ids,
        eval_run_summary=str(resolve_run_summary_path(summary)),
        total_task_weight=total_weight,
        variant_successes=successes,
        variant_invalid_tasks=invalid_tasks,
        variant_scores=scores,
        candidate_beats_frontier=candidate_score > frontier_score,
        candidate_score_delta=round(candidate_score - frontier_score, 2),
    )


def count_variant_successes(summary: EvalRunSummary) -> dict[str, int]:
    successes: dict[str, int] = {}
    for task in summary.tasks:
        for variant in task.variants:
            if variant.success:
                successes[variant.name] = successes.get(variant.name, 0) + 1
            else:
                successes.setdefault(variant.name, 0)
    return successes


def count_variant_invalid_tasks(summary: EvalRunSummary) -> dict[str, int]:
    invalid_tasks: dict[str, int] = {}
    for task in summary.tasks:
        for variant in task.variants:
            if not variant.validity_passed:
                invalid_tasks[variant.name] = invalid_tasks.get(variant.name, 0) + 1
            else:
                invalid_tasks.setdefault(variant.name, 0)
    return invalid_tasks


def score_variants(summary: EvalRunSummary) -> dict[str, float]:
    total_weight = sum(task.task_weight for task in summary.tasks)
    if total_weight <= 0:
        raise ValueError("Challenge pool has zero total task weight.")
    weighted_scores: dict[str, float] = {}
    for task in summary.tasks:
        for variant in task.variants:
            weighted_scores[variant.name] = (
                weighted_scores.get(variant.name, 0.0) + variant.weighted_task_score
            )
    return {
        name: round((weighted_score / total_weight) * 100, 2)
        for name, weighted_score in weighted_scores.items()
    }


def evaluate_promotion(
    primary: ChallengePoolSummary,
    holdout: ChallengePoolSummary | None,
) -> tuple[bool, str]:
    primary_delta = primary.candidate_score_delta
    if primary.variant_invalid_tasks.get("candidate", 0) > 0:
        return False, "candidate has invalid primary-pool task runs"
    if not primary.candidate_beats_frontier:
        return False, "candidate did not beat the current frontier on the primary score"
    if primary_delta < PRIMARY_PROMOTION_MARGIN_POINTS:
        return (
            False,
            "candidate improved the primary score but did not clear the promotion margin",
        )
    if holdout is None:
        return True, "candidate cleared the primary score margin"
    if holdout.variant_invalid_tasks.get("candidate", 0) > 0:
        return False, "candidate has invalid holdout-pool task runs"
    if holdout.variant_scores.get("candidate", 0.0) < holdout.variant_scores.get("frontier", 0.0):
        return False, "candidate cleared the primary score margin but regressed on holdout"
    return True, "candidate cleared the primary score margin and held on holdout"


def promotion_reason(primary: ChallengePoolSummary, holdout: ChallengePoolSummary | None) -> str:
    return evaluate_promotion(primary, holdout)[1]


def resolve_mode(manifest: FrontierManifest, mode: str) -> FrontierModeConfig:
    mode_config = manifest.modes.get(mode)
    if mode_config is None:
        raise ValueError(
            f"Mode is not configured in frontier manifest: {mode}. "
            "Run `promptforge frontier init` first."
        )
    return mode_config


def render_pool(pool: ChallengePoolSummary) -> list[str]:
    lines = [
        f"- Tasks: {', '.join(pool.task_ids)}",
        f"- Eval run: `{pool.eval_run_summary}`",
        f"- Total task weight: {pool.total_task_weight:g}",
    ]
    for variant_name in ("baseline", "frontier", "candidate"):
        lines.append(f"- {variant_name} solved: {pool.variant_successes.get(variant_name, 0)}")
        lines.append(
            f"- {variant_name} invalid tasks: {pool.variant_invalid_tasks.get(variant_name, 0)}"
        )
        lines.append(f"- {variant_name} score: {pool.variant_scores.get(variant_name, 0.0):.2f}")
    lines.append(
        f"- Candidate beats frontier: {'yes' if pool.candidate_beats_frontier else 'no'}"
    )
    lines.append(f"- Candidate score delta: {pool.candidate_score_delta:+.2f}")
    return lines


def resolve_run_summary_path(summary: EvalRunSummary) -> Path:
    if not summary.tasks:
        raise ValueError("Eval summary contains no tasks.")
    first_task_path = Path(summary.tasks[0].task_path).resolve()
    return first_task_path.parents[1] / "run_summary.json"


def build_challenge_id(eval_pack_name: str, mode: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"challenge-{eval_pack_name}-{mode}-{timestamp}"


def write_challenge_summary(path: Path, summary: ChallengeSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
