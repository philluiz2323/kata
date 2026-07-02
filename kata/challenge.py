from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from kata.agent_bundle import AGENT_ENTRY_FILENAME, load_bundle_files
from kata.benchmarks import resolve_eval_pack_path, resolve_private_eval_pack_path
from kata.config import resolve_validator_model
from kata.eval_pack import discover_live_eval_pack_tasks
from kata.eval_runner import ArtifactVariant, EvalRunSummary, run_artifact_variants
from kata.evaluators.sn60_bitsec import (
    DEFAULT_REPLICAS_PER_PROJECT,
    Sn60DuelSummary,
    Sn60EvaluationHook,
    Sn60ExecutionHook,
    Sn60SandboxSource,
    Sn60VariantSummary,
    bitsec_project_image,
    hash_bundle_root,
    resolve_sn60_sandbox_source,
    run_sn60_bitsec_duel,
)
from kata.frontier import (
    DEFAULT_PROMOTION_MARGIN_POINTS,
    PRIMARY_SELECTION_RANDOM_LIVE,
    FrontierManifest,
    FrontierModeConfig,
    load_frontier_manifest,
    resolve_frontier_artifact_hash,
)
from kata.lane_state import (
    BENCHMARK_SNAPSHOT_SCHEMA_VERSION,
    CHALLENGE_STATE_SCHEMA_VERSION,
    PROMOTION_RECORD_SCHEMA_VERSION,
    BenchmarkSnapshotState,
    ChallengeState,
    PromotionRecord,
    write_benchmark_snapshot,
    write_challenge_state,
    write_promotion_record,
)
from kata.live_progress import update_live_status
from kata.provenance import EVALUATOR_VERSION, pool_fingerprint, short_hash
from kata.public_artifacts import resolve_artifact_path
from kata.screening import (
    Sn60ScreeningHook,
    Sn60ScreeningResult,
    run_sn60_screening,
    screening_result_payload,
    sn60_screening_freshness_fingerprint,
    write_screening_result,
)

SUBMISSION_METADATA_FILENAME = "submission.json"
SN60_MINER_LANE_ID = "sn60__bitsec"
SN60_MINER_MODE = "miner"
SN60_VALIDATOR_MODEL = "sn60-bitsec-sandbox"


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
    validator_model: str
    frontier_artifact: str
    candidate_artifact: str
    frontier_artifact_hash: str
    candidate_artifact_hash: str
    primary_pool_fingerprint: str | None
    holdout_pool_fingerprint: str | None
    promotion_margin_points: float
    holdout_promotion_margin_points: float
    created_at: str
    primary: ChallengePoolSummary
    holdout: ChallengePoolSummary | None
    promotion_ready: bool
    promotion_reason: str


@dataclass(frozen=True)
class Sn60PromotionDecision:
    promotion_ready: bool
    final_winner: str
    reason: str


def run_frontier_challenge(
    *,
    eval_pack_path: str,
    mode: str,
    candidate_artifact_path: str,
    agent_command: str,
    output_root: str | None = None,
    agent_timeout_seconds: int | None = None,
    checks_timeout_seconds: int | None = None,
) -> ChallengeSummary:
    eval_pack_root = resolve_eval_pack_path(eval_pack_path)
    manifest = load_frontier_manifest(eval_pack_path)
    mode_config = resolve_mode(manifest, mode)
    candidate_path = Path(candidate_artifact_path).expanduser().resolve()
    if candidate_path.is_file():
        candidate_path = candidate_path.parent
    output_base = Path(output_root) if output_root else Path("runs")
    challenge_run_id = build_challenge_id(eval_pack_root.name, mode)
    challenge_root = output_base / challenge_run_id
    challenge_root.mkdir(parents=True, exist_ok=False)

    frontier_files = load_bundle_files(resolve_artifact_path(mode_config.frontier_artifact))
    candidate_files = load_bundle_files(candidate_path)
    evaluator_version = mode_config.evaluator_version or EVALUATOR_VERSION
    validator_model = resolve_validator_model()
    frontier_hash = resolve_frontier_artifact_hash(mode_config)
    candidate_hash = sha256_bundle_dict(candidate_files)
    promotion_margin_points = mode_config.promotion_margin_points
    selected_primary_tasks = resolve_primary_task_ids(eval_pack_path, mode_config)
    current_primary_fingerprint = current_primary_pool_fingerprint(
        eval_pack_path,
        mode_config,
        selected_task_ids=selected_primary_tasks,
    )
    if mode_config.holdout_task_count > 0 and not mode_config.holdout_tasks:
        raise ValueError(
            "Private holdout tasks are configured for this lane, but they are not available "
            "in the current validator environment. Set KATA_PRIVATE_BENCHMARKS_ROOT."
        )
    current_holdout_fingerprint = current_holdout_pool_fingerprint(
        eval_pack_path,
        mode_config,
    )
    update_live_status(
        {
            "state": "running",
            "phase": "primary",
            "repo_pack": eval_pack_root.name,
            "mode": mode,
            "candidate_submission_id": candidate_path.name,
            "candidate_author": resolve_candidate_author(candidate_path),
            "challenge_run_id": challenge_run_id,
            "evaluator_version": evaluator_version,
            "validator_model": validator_model,
            "frontier_artifact_hash": frontier_hash,
            "candidate_artifact_hash": candidate_hash,
            "primary_pool_fingerprint": current_primary_fingerprint,
            "holdout_pool_fingerprint": current_holdout_fingerprint,
            "pools": {
                "primary": queued_pool_status("primary", selected_primary_tasks),
                "holdout": queued_pool_status("holdout", mode_config.holdout_tasks),
            },
        }
    )

    primary_eval = run_artifact_variants(
        repo_ref=manifest.repo_ref,
        eval_pack_path=eval_pack_path,
        mode=mode,
        agent_command=agent_command,
        artifact_variants=[
            ArtifactVariant(
                name="frontier",
                files=frontier_files,
                entrypoint=AGENT_ENTRY_FILENAME,
            ),
            ArtifactVariant(
                name="candidate",
                files=candidate_files,
                entrypoint=AGENT_ENTRY_FILENAME,
            ),
        ],
        task_names=selected_primary_tasks,
        output_root=str(challenge_root / "primary"),
        run_label=f"{eval_pack_root.name}-{mode}-primary",
        run_kind="challenge-primary",
        metadata={
            "evaluator_version": evaluator_version,
            "pool_name": "primary",
            "pool_fingerprint": current_primary_fingerprint or "",
            "validator_model": validator_model,
            "frontier_artifact_hash": frontier_hash,
            "candidate_artifact_hash": candidate_hash,
        },
        agent_timeout_seconds=agent_timeout_seconds,
        checks_timeout_seconds=checks_timeout_seconds,
    )
    primary_summary = summarize_pool(primary_eval, selected_primary_tasks)

    holdout_summary: ChallengePoolSummary | None = None
    promotion_ready = False
    if primary_summary.candidate_beats_frontier and mode_config.holdout_tasks:
        update_live_status(
            {
                "state": "running",
                "phase": "holdout",
                "holdout_pool_fingerprint": current_holdout_fingerprint,
            }
        )
        holdout_eval_pack_path = (
            str(
                resolve_private_eval_pack_path(
                    mode_config.holdout_eval_pack or eval_pack_root.name
                )
            )
            if mode_config.holdout_is_private
            else (mode_config.holdout_eval_pack or eval_pack_path)
        )
        holdout_eval = run_artifact_variants(
            repo_ref=manifest.repo_ref,
            eval_pack_path=holdout_eval_pack_path,
            mode=mode,
            agent_command=agent_command,
            artifact_variants=[
                ArtifactVariant(
                    name="frontier",
                    files=frontier_files,
                    entrypoint=AGENT_ENTRY_FILENAME,
                ),
                ArtifactVariant(
                    name="candidate",
                    files=candidate_files,
                    entrypoint=AGENT_ENTRY_FILENAME,
                ),
            ],
            task_names=mode_config.holdout_tasks,
            output_root=str(challenge_root / "holdout"),
            run_label=f"{eval_pack_root.name}-{mode}-holdout",
            run_kind="challenge-holdout",
            metadata={
                "evaluator_version": evaluator_version,
                "pool_name": "holdout",
                "pool_fingerprint": current_holdout_fingerprint or "",
                "validator_model": validator_model,
                "frontier_artifact_hash": frontier_hash,
                "candidate_artifact_hash": candidate_hash,
            },
            agent_timeout_seconds=agent_timeout_seconds,
            checks_timeout_seconds=checks_timeout_seconds,
        )
        holdout_summary = summarize_pool(holdout_eval, mode_config.holdout_tasks)
    promotion_ready, reason = evaluate_promotion(
        primary_summary,
        holdout_summary,
        promotion_margin_points=promotion_margin_points,
        holdout_promotion_margin_points=mode_config.holdout_promotion_margin_points,
    )
    summary = ChallengeSummary(
        schema_version=4,
        run_id=challenge_run_id,
        manifest_path=str(eval_pack_root / "frontier.json"),
        mode=mode,
        evaluator_version=evaluator_version,
        validator_model=validator_model,
        frontier_artifact=str(resolve_artifact_path(mode_config.frontier_artifact)),
        candidate_artifact=str(candidate_path),
        frontier_artifact_hash=frontier_hash,
        candidate_artifact_hash=candidate_hash,
        primary_pool_fingerprint=current_primary_fingerprint,
        holdout_pool_fingerprint=current_holdout_fingerprint,
        promotion_margin_points=promotion_margin_points,
        holdout_promotion_margin_points=mode_config.holdout_promotion_margin_points,
        created_at=datetime.now(UTC).isoformat(),
        primary=primary_summary,
        holdout=holdout_summary,
        promotion_ready=promotion_ready,
        promotion_reason=reason,
    )
    write_challenge_summary(challenge_root / "challenge_summary.json", summary)
    update_live_status(
        {
            "state": "verifying",
            "phase": "verifying",
            "challenge_summary_path": str(challenge_root / "challenge_summary.json"),
            "promotion_ready": promotion_ready,
            "promotion_reason": reason,
        }
    )
    return summary


def run_sn60_challenge(
    *,
    frontier_artifact_path: str,
    candidate_artifact_path: str,
    project_keys: list[str],
    candidate_submission_id: str,
    lane_id: str = SN60_MINER_LANE_ID,
    output_root: str | None = None,
    replicas_per_project: int = DEFAULT_REPLICAS_PER_PROJECT,
    sandbox_root: str | None = None,
    benchmark_file: str | None = None,
    sandbox_commit: str | None = None,
    screening_result: dict[str, object] | None = None,
    public_root: str | None = None,
    screening_hook: Sn60ScreeningHook | None = None,
    execution_hook: Sn60ExecutionHook | None = None,
    evaluation_hook: Sn60EvaluationHook | None = None,
) -> ChallengeSummary:
    if not project_keys:
        raise ValueError("SN60 challenge requires at least one screening project key.")
    sandbox_source = resolve_sn60_sandbox_source(
        sandbox_root=sandbox_root,
        benchmark_file=benchmark_file,
        sandbox_commit=sandbox_commit,
        scorer_version="ScaBenchScorerV2",
    )
    screening = run_sn60_screening(
        candidate_artifact_path=candidate_artifact_path,
        project_key=project_keys[0],
        output_root=output_root or "runs",
        sandbox_source=sandbox_source,
        execution_hook=screening_hook,
    )
    effective_screening_result = screening_result_payload(screening)
    if screening_result:
        effective_screening_result["details"] = {
            **dict(effective_screening_result.get("details") or {}),
            "caller_context": screening_result,
        }
    if not screening.passed:
        summary = build_sn60_screening_failure_summary(
            frontier_artifact_path=frontier_artifact_path,
            candidate_artifact_path=candidate_artifact_path,
            project_keys=project_keys,
            lane_id=lane_id,
            screening=screening,
        )
        write_challenge_summary(
            Path(screening.result_path).with_name("challenge_summary.json"),
            summary,
        )
        record_sn60_screening_failure_provenance(
            lane_id=lane_id,
            candidate_submission_id=candidate_submission_id,
            frontier_artifact_path=frontier_artifact_path,
            project_keys=project_keys,
            replicas_per_project=replicas_per_project,
            screening=screening,
            public_root=public_root,
        )
        return summary

    duel_summary = run_sn60_bitsec_duel(
        frontier_artifact_path=frontier_artifact_path,
        candidate_artifact_path=candidate_artifact_path,
        project_keys=project_keys,
        output_root=output_root,
        replicas_per_project=replicas_per_project,
        sandbox_root=sandbox_source.sandbox_root,
        benchmark_file=sandbox_source.benchmark_file,
        sandbox_commit=sandbox_source.sandbox_commit,
        execution_hook=execution_hook,
        evaluation_hook=evaluation_hook,
    )
    write_screening_result(Path(duel_summary.output_root) / "screening_result.json", screening)
    summary = sn60_duel_to_challenge_summary(
        duel_summary,
        lane_id=lane_id,
        screening_result=effective_screening_result,
    )
    challenge_summary_path = Path(duel_summary.output_root) / "challenge_summary.json"
    write_challenge_summary(challenge_summary_path, summary)
    record_sn60_lane_provenance(
        lane_id=lane_id,
        candidate_submission_id=candidate_submission_id,
        duel_summary=duel_summary,
        screening_result=effective_screening_result,
        public_root=public_root,
    )
    return summary


def sn60_duel_to_challenge_summary(
    duel_summary: Sn60DuelSummary,
    *,
    lane_id: str = SN60_MINER_LANE_ID,
    screening_result: dict[str, object] | None = None,
) -> ChallengeSummary:
    decision = evaluate_sn60_promotion(
        frontier=duel_summary.frontier,
        candidate=duel_summary.candidate,
        screening_result=screening_result,
    )
    freshness_fingerprint = sn60_freshness_fingerprint(duel_summary)
    duel_summary_path = Path(duel_summary.output_root) / "duel_summary.json"
    return ChallengeSummary(
        schema_version=4,
        run_id=duel_summary.run_id,
        manifest_path=str(duel_summary_path),
        mode=SN60_MINER_MODE,
        evaluator_version=sn60_evaluator_version(duel_summary),
        validator_model=SN60_VALIDATOR_MODEL,
        frontier_artifact=duel_summary.frontier.artifact_path,
        candidate_artifact=duel_summary.candidate.artifact_path,
        frontier_artifact_hash=duel_summary.frontier.artifact_hash,
        candidate_artifact_hash=duel_summary.candidate.artifact_hash,
        primary_pool_fingerprint=freshness_fingerprint,
        holdout_pool_fingerprint=None,
        promotion_margin_points=0.0,
        holdout_promotion_margin_points=0.0,
        created_at=duel_summary.created_at,
        primary=sn60_duel_to_pool_summary(
            duel_summary,
            eval_run_summary=duel_summary_path,
            screening_result=screening_result,
        ),
        holdout=None,
        promotion_ready=decision.promotion_ready,
        promotion_reason=f"{lane_id}: {decision.reason}",
    )


def sn60_duel_to_pool_summary(
    duel_summary: Sn60DuelSummary,
    *,
    eval_run_summary: Path,
    screening_result: dict[str, object] | None = None,
) -> ChallengePoolSummary:
    frontier_score = round(duel_summary.frontier.aggregated_score * 100, 2)
    candidate_score = round(duel_summary.candidate.aggregated_score * 100, 2)
    decision = evaluate_sn60_promotion(
        frontier=duel_summary.frontier,
        candidate=duel_summary.candidate,
        screening_result=screening_result,
    )
    return ChallengePoolSummary(
        task_ids=list(duel_summary.project_keys),
        eval_run_summary=str(eval_run_summary),
        total_task_weight=float(len(duel_summary.project_keys)),
        variant_successes={
            "frontier": duel_summary.frontier.codebase_pass_count,
            "candidate": duel_summary.candidate.codebase_pass_count,
        },
        variant_invalid_tasks={
            "frontier": duel_summary.frontier.invalid_runs,
            "candidate": duel_summary.candidate.invalid_runs,
        },
        variant_scores={
            "frontier": frontier_score,
            "candidate": candidate_score,
        },
        candidate_beats_frontier=decision.final_winner == "candidate",
        candidate_score_delta=round(candidate_score - frontier_score, 2),
    )


def build_sn60_screening_failure_summary(
    *,
    frontier_artifact_path: str,
    candidate_artifact_path: str,
    project_keys: list[str],
    lane_id: str,
    screening: Sn60ScreeningResult,
) -> ChallengeSummary:
    frontier_root = Path(frontier_artifact_path).expanduser().resolve()
    candidate_root = Path(candidate_artifact_path).expanduser().resolve()
    frontier_hash = hash_bundle_root(frontier_root)
    freshness_fingerprint = sn60_screening_freshness_fingerprint(
        frontier_artifact_hash=frontier_hash,
        screening_result=screening,
    )
    reason = "; ".join(screening.reasons) if screening.reasons else "unknown screening failure"
    return ChallengeSummary(
        schema_version=4,
        run_id=screening.run_id,
        manifest_path=screening.result_path,
        mode=SN60_MINER_MODE,
        evaluator_version=(
            f"{screening.sandbox_source.scorer_version}"
            f"@{short_hash(screening.sandbox_source.sandbox_commit)}"
        ),
        validator_model=SN60_VALIDATOR_MODEL,
        frontier_artifact=str(frontier_root),
        candidate_artifact=str(candidate_root),
        frontier_artifact_hash=frontier_hash,
        candidate_artifact_hash=screening.artifact_hash,
        primary_pool_fingerprint=freshness_fingerprint,
        holdout_pool_fingerprint=None,
        promotion_margin_points=0.0,
        holdout_promotion_margin_points=0.0,
        created_at=screening.created_at,
        primary=ChallengePoolSummary(
            task_ids=list(project_keys),
            eval_run_summary=screening.result_path,
            total_task_weight=1.0,
            variant_successes={"frontier": 0, "candidate": 0},
            variant_invalid_tasks={"frontier": 0, "candidate": 1},
            variant_scores={"frontier": 0.0, "candidate": 0.0},
            candidate_beats_frontier=False,
            candidate_score_delta=0.0,
        ),
        holdout=None,
        promotion_ready=False,
        promotion_reason=f"{lane_id}: candidate failed SN60 screening: {reason}",
    )


def evaluate_sn60_promotion(
    *,
    frontier: Sn60VariantSummary,
    candidate: Sn60VariantSummary,
    screening_result: dict[str, object] | None = None,
) -> Sn60PromotionDecision:
    screening_status = screening_result.get("status") if screening_result is not None else None
    if screening_result is not None and screening_status not in {"passed", "pass", True}:
        return Sn60PromotionDecision(
            promotion_ready=False,
            final_winner="frontier",
            reason="candidate failed SN60 screening",
        )
    if candidate.invalid_runs > 0:
        return Sn60PromotionDecision(
            promotion_ready=False,
            final_winner="frontier",
            reason="candidate has invalid SN60 replica runs",
        )

    candidate_rank = sn60_variant_rank(candidate)
    frontier_rank = sn60_variant_rank(frontier)
    if candidate_rank <= frontier_rank:
        return Sn60PromotionDecision(
            promotion_ready=False,
            final_winner="frontier",
            reason="candidate did not beat the current SN60 king",
        )
    return Sn60PromotionDecision(
        promotion_ready=True,
        final_winner="candidate",
        reason="candidate beat the current SN60 king",
    )


def sn60_variant_rank(summary: Sn60VariantSummary) -> tuple[float, int, int, int]:
    # Promotion comparator per the frozen SN60 spec:
    # aggregated score first, codebase pass count second, true positives third.
    return (
        round(summary.aggregated_score, 8),
        summary.codebase_pass_count,
        summary.true_positives,
        -summary.invalid_runs,
    )


def record_sn60_lane_provenance(
    *,
    lane_id: str,
    candidate_submission_id: str,
    duel_summary: Sn60DuelSummary,
    screening_result: dict[str, object],
    public_root: str | None = None,
    reward_label_applied: str | None = None,
) -> tuple[Path, Path]:
    decision = evaluate_sn60_promotion(
        frontier=duel_summary.frontier,
        candidate=duel_summary.candidate,
        screening_result=screening_result,
    )
    freshness_fingerprint = sn60_freshness_fingerprint(duel_summary)
    record_sn60_benchmark_snapshot(
        lane_id=lane_id,
        sandbox_source=duel_summary.sandbox_source,
        project_keys=list(duel_summary.project_keys),
        public_root=public_root,
    )
    challenge_path = write_challenge_state(
        lane_id,
        ChallengeState(
            schema_version=CHALLENGE_STATE_SCHEMA_VERSION,
            candidate_submission_id=candidate_submission_id,
            candidate_artifact_hash=duel_summary.candidate.artifact_hash,
            king_artifact_hash=duel_summary.frontier.artifact_hash,
            screening_result=screening_result,
            selected_project_keys=list(duel_summary.project_keys),
            validator_replica_count=duel_summary.replicas_per_project,
            run_ids=[duel_summary.run_id],
            freshness_fingerprint=freshness_fingerprint,
            updated_at=datetime.now(UTC).isoformat(),
        ),
        public_root=public_root,
    )
    promotion_path = write_promotion_record(
        lane_id,
        PromotionRecord(
            schema_version=PROMOTION_RECORD_SCHEMA_VERSION,
            final_metrics=sn60_final_metrics(duel_summary, decision),
            local_replica_scores=sn60_local_replica_scores(duel_summary),
            pass_counts={
                "frontier": duel_summary.frontier.codebase_pass_count,
                "candidate": duel_summary.candidate.codebase_pass_count,
            },
            true_positives={
                "frontier": duel_summary.frontier.true_positives,
                "candidate": duel_summary.candidate.true_positives,
            },
            invalid_runs={
                "frontier": duel_summary.frontier.invalid_runs,
                "candidate": duel_summary.candidate.invalid_runs,
            },
            final_winner=decision.final_winner,
            reward_label_applied=reward_label_applied,
            recorded_at=datetime.now(UTC).isoformat(),
        ),
        public_root=public_root,
    )
    return challenge_path, promotion_path


def record_sn60_screening_failure_provenance(
    *,
    lane_id: str,
    candidate_submission_id: str,
    frontier_artifact_path: str,
    project_keys: list[str],
    replicas_per_project: int,
    screening: Sn60ScreeningResult,
    public_root: str | None = None,
) -> tuple[Path, Path]:
    frontier_hash = hash_bundle_root(Path(frontier_artifact_path).expanduser().resolve())
    freshness_fingerprint = sn60_screening_freshness_fingerprint(
        frontier_artifact_hash=frontier_hash,
        screening_result=screening,
    )
    screening_payload = screening_result_payload(screening)
    reason = "; ".join(screening.reasons) if screening.reasons else "unknown screening failure"
    record_sn60_benchmark_snapshot(
        lane_id=lane_id,
        sandbox_source=screening.sandbox_source,
        project_keys=list(project_keys),
        public_root=public_root,
    )
    challenge_path = write_challenge_state(
        lane_id,
        ChallengeState(
            schema_version=CHALLENGE_STATE_SCHEMA_VERSION,
            candidate_submission_id=candidate_submission_id,
            candidate_artifact_hash=screening.artifact_hash,
            king_artifact_hash=frontier_hash,
            screening_result=screening_payload,
            selected_project_keys=list(project_keys),
            validator_replica_count=replicas_per_project,
            run_ids=[screening.run_id],
            freshness_fingerprint=freshness_fingerprint,
            updated_at=datetime.now(UTC).isoformat(),
        ),
        public_root=public_root,
    )
    promotion_path = write_promotion_record(
        lane_id,
        PromotionRecord(
            schema_version=PROMOTION_RECORD_SCHEMA_VERSION,
            final_metrics={
                "run_id": screening.run_id,
                "promotion_ready": False,
                "promotion_reason": f"candidate failed SN60 screening: {reason}",
                "screening_status": screening.status,
                "screening_stage": screening.stage,
                "sandbox_commit": screening.sandbox_source.sandbox_commit,
                "benchmark_sha256": screening.sandbox_source.benchmark_sha256,
                "scorer_version": screening.sandbox_source.scorer_version,
            },
            local_replica_scores={"frontier": [], "candidate": []},
            pass_counts={"frontier": 0, "candidate": 0},
            true_positives={"frontier": 0, "candidate": 0},
            invalid_runs={"frontier": 0, "candidate": 1},
            final_winner="frontier",
            reward_label_applied=None,
            recorded_at=datetime.now(UTC).isoformat(),
        ),
        public_root=public_root,
    )
    return challenge_path, promotion_path


def sn60_final_metrics(
    duel_summary: Sn60DuelSummary,
    decision: Sn60PromotionDecision,
) -> dict[str, object]:
    frontier_aggregated = duel_summary.frontier.aggregated_score
    candidate_aggregated = duel_summary.candidate.aggregated_score
    return {
        "run_id": duel_summary.run_id,
        "promotion_ready": decision.promotion_ready,
        "promotion_reason": decision.reason,
        "frontier_aggregated_score": frontier_aggregated,
        "candidate_aggregated_score": candidate_aggregated,
        "candidate_aggregated_score_delta": candidate_aggregated - frontier_aggregated,
        # `average score` is defined as exactly equal to `aggregated score` per the
        # SN60 metric naming rule; kept for existing dashboard consumers.
        "frontier_average_score": frontier_aggregated,
        "candidate_average_score": candidate_aggregated,
        "candidate_score_delta": candidate_aggregated - frontier_aggregated,
        "sandbox_commit": duel_summary.sandbox_source.sandbox_commit,
        "benchmark_sha256": duel_summary.sandbox_source.benchmark_sha256,
        "scorer_version": duel_summary.sandbox_source.scorer_version,
    }


def sn60_local_replica_scores(duel_summary: Sn60DuelSummary) -> dict[str, list[float]]:
    return {
        "frontier": [result.score for result in duel_summary.frontier.replica_results],
        "candidate": [result.score for result in duel_summary.candidate.replica_results],
    }


def record_sn60_benchmark_snapshot(
    *,
    lane_id: str,
    sandbox_source: Sn60SandboxSource,
    project_keys: list[str],
    public_root: str | None = None,
) -> None:
    write_benchmark_snapshot(
        lane_id,
        BenchmarkSnapshotState(
            schema_version=BENCHMARK_SNAPSHOT_SCHEMA_VERSION,
            sandbox_mirror_source=sandbox_source.sandbox_root,
            sandbox_commit_hash=sandbox_source.sandbox_commit,
            benchmark_dataset_id=Path(sandbox_source.benchmark_file).name,
            benchmark_dataset_hash=sandbox_source.benchmark_sha256,
            project_list_hash=sn60_project_list_hash(project_keys),
            project_keys=list(project_keys),
            container_images=[
                bitsec_project_image(project_key) for project_key in project_keys
            ],
            scorer_version=sandbox_source.scorer_version,
            updated_at=datetime.now(UTC).isoformat(),
        ),
        public_root=public_root,
    )


def sn60_project_list_hash(project_keys: list[str]) -> str:
    payload = json.dumps(sorted(project_keys))
    return sha256(payload.encode("utf-8")).hexdigest()


def sn60_freshness_fingerprint(duel_summary: Sn60DuelSummary) -> str:
    payload = {
        "frontier_artifact_hash": duel_summary.frontier.artifact_hash,
        "candidate_artifact_hash": duel_summary.candidate.artifact_hash,
        "project_keys": duel_summary.project_keys,
        "replicas_per_project": duel_summary.replicas_per_project,
        "sandbox_commit": duel_summary.sandbox_source.sandbox_commit,
        "benchmark_sha256": duel_summary.sandbox_source.benchmark_sha256,
        "scorer_version": duel_summary.sandbox_source.scorer_version,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def sn60_evaluator_version(duel_summary: Sn60DuelSummary) -> str:
    return (
        f"{duel_summary.sandbox_source.scorer_version}"
        f"@{short_hash(duel_summary.sandbox_source.sandbox_commit)}"
    )


def queued_pool_status(pool_name: str, task_ids: list[str]) -> dict[str, object]:
    return {
        "name": pool_name,
        "state": "queued",
        "total_tasks": len(task_ids),
        "completed_tasks": 0,
        "task_statuses": [
            {
                "task_id": task_id,
                "status": "queued",
                "completed": False,
                "candidate": {"started": False, "finished": False},
                "frontier": {"started": False, "finished": False},
            }
            for task_id in task_ids
        ],
    }


def render_challenge_summary(summary: ChallengeSummary) -> str:
    lines: list[str] = []
    lines.append(f"Challenge run: {summary.run_id}")
    lines.append(f"Mode: {summary.mode}")
    lines.append(f"Manifest: `{summary.manifest_path}`")
    lines.append(f"Candidate artifact: `{summary.candidate_artifact}`")
    lines.append(f"Evaluator version: {summary.evaluator_version}")
    lines.append(f"Validator model: {summary.validator_model}")
    lines.append(f"Frontier artifact hash: {short_hash(summary.frontier_artifact_hash)}")
    lines.append(f"Candidate artifact hash: {short_hash(summary.candidate_artifact_hash)}")
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
    lines.append(
        f"Holdout margin: {summary.holdout_promotion_margin_points:.1f} points"
    )
    lines.append(f"Promotion ready: {'yes' if summary.promotion_ready else 'no'}")
    lines.append(f"Reason: {summary.promotion_reason}")
    return "\n".join(lines)


def load_challenge_summary(path: str) -> ChallengeSummary:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    holdout_payload = payload.get("holdout")
    frontier_artifact = payload.get("frontier_artifact")
    if frontier_artifact is None:
        frontier_artifact = payload["frontier_prompt"]
    candidate_artifact = payload.get("candidate_artifact")
    if candidate_artifact is None:
        candidate_artifact = payload["candidate_prompt"]
    return ChallengeSummary(
        schema_version=payload["schema_version"],
        run_id=payload["run_id"],
        manifest_path=payload["manifest_path"],
        mode=payload["mode"],
        evaluator_version=payload.get("evaluator_version", EVALUATOR_VERSION),
        validator_model=payload.get("validator_model", resolve_validator_model()),
        frontier_artifact=frontier_artifact,
        candidate_artifact=candidate_artifact,
        frontier_artifact_hash=payload.get(
            "frontier_artifact_hash",
            payload.get("frontier_prompt_hash", ""),
        ),
        candidate_artifact_hash=payload.get(
            "candidate_artifact_hash",
            payload.get("candidate_prompt_hash", ""),
        ),
        primary_pool_fingerprint=payload.get("primary_pool_fingerprint"),
        holdout_pool_fingerprint=payload.get("holdout_pool_fingerprint"),
        promotion_margin_points=payload.get(
            "promotion_margin_points", DEFAULT_PROMOTION_MARGIN_POINTS
        ),
        holdout_promotion_margin_points=payload.get(
            "holdout_promotion_margin_points", 0.0
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
    *,
    promotion_margin_points: float = DEFAULT_PROMOTION_MARGIN_POINTS,
    holdout_promotion_margin_points: float = 0.0,
) -> tuple[bool, str]:
    primary_delta = primary.candidate_score_delta
    if primary.variant_invalid_tasks.get("candidate", 0) > 0:
        return False, "candidate has invalid primary-pool task runs"
    if not primary.candidate_beats_frontier:
        return False, "candidate did not beat the current frontier on the primary score"
    if primary_delta < promotion_margin_points:
        return (
            False,
            "candidate improved the primary score but did not clear the promotion margin",
        )
    if holdout is None:
        return True, "candidate cleared the primary score margin"
    if holdout.variant_invalid_tasks.get("candidate", 0) > 0:
        return False, "candidate has invalid holdout-pool task runs"
    holdout_delta = holdout.candidate_score_delta
    if holdout_delta < holdout_promotion_margin_points:
        if holdout_delta < 0:
            return False, "candidate cleared the primary score margin but regressed on holdout"
        return (
            False,
            "candidate cleared the primary score margin but did not clear the holdout margin",
        )
    return True, "candidate cleared the primary score margin and holdout margin"


def promotion_reason(primary: ChallengePoolSummary, holdout: ChallengePoolSummary | None) -> str:
    return evaluate_promotion(primary, holdout)[1]


def resolve_mode(manifest: FrontierManifest, mode: str) -> FrontierModeConfig:
    mode_config = manifest.modes.get(mode)
    if mode_config is None:
        raise ValueError(
            f"Mode is not configured in frontier manifest: {mode}. "
            "Run `kata frontier init` first."
        )
    return mode_config


def resolve_primary_task_ids(
    eval_pack_path: str,
    mode_config: FrontierModeConfig,
) -> list[str]:
    if mode_config.primary_selection != PRIMARY_SELECTION_RANDOM_LIVE:
        return list(mode_config.primary_tasks)
    validations = discover_live_eval_pack_tasks(eval_pack_path)
    available = sorted(result.root.name for result in validations)
    requested_count = mode_config.primary_task_count or len(available)
    if requested_count <= 0:
        raise ValueError("Random public primary pool requires at least one task.")
    if len(available) < requested_count:
        raise ValueError(
            "Random public primary pool is underfilled. "
            f"Requested {requested_count} live tasks but only {len(available)} are available."
        )
    return sorted(secrets.SystemRandom().sample(available, requested_count))


def current_primary_pool_fingerprint(
    eval_pack_path: str,
    mode_config: FrontierModeConfig,
    *,
    selected_task_ids: list[str] | None = None,
) -> str | None:
    eval_pack_root = resolve_eval_pack_path(eval_pack_path)
    if mode_config.primary_selection == PRIMARY_SELECTION_RANDOM_LIVE:
        if selected_task_ids is not None:
            if not selected_task_ids:
                return None
            return pool_fingerprint(
                [
                    eval_pack_root / validate_selected_task_id(task_id)
                    for task_id in selected_task_ids
                ]
            )
        validations = discover_live_eval_pack_tasks(eval_pack_path)
        return pool_fingerprint([result.root for result in validations]) if validations else None
    if not mode_config.primary_tasks:
        return None
    return pool_fingerprint([eval_pack_root / task_id for task_id in mode_config.primary_tasks])


def validate_selected_task_id(task_id: str) -> str:
    task_path = Path(task_id)
    if task_path.is_absolute() or task_path.name != task_id or task_id in {"", ".", ".."}:
        raise ValueError(f"Invalid challenge task id: {task_id}")
    return task_id


def current_holdout_pool_fingerprint(
    eval_pack_path: str,
    mode_config: FrontierModeConfig,
) -> str | None:
    if not mode_config.holdout_tasks:
        return None
    holdout_pack_ref = mode_config.holdout_eval_pack or resolve_eval_pack_path(
        eval_pack_path
    ).name
    holdout_eval_pack_root = (
        resolve_private_eval_pack_path(holdout_pack_ref)
        if mode_config.holdout_is_private
        else resolve_eval_pack_path(mode_config.holdout_eval_pack or eval_pack_path)
    )
    holdout_task_roots = [
        holdout_eval_pack_root / task_id for task_id in mode_config.holdout_tasks
    ]
    return pool_fingerprint(holdout_task_roots)


def render_pool(pool: ChallengePoolSummary) -> list[str]:
    lines = [
        f"- Tasks: {', '.join(pool.task_ids)}",
        f"- Eval run: `{pool.eval_run_summary}`",
        f"- Total task weight: {pool.total_task_weight:g}",
    ]
    for variant_name in ("frontier", "candidate"):
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


def infer_submission_author(submission_id: str) -> str | None:
    if submission_id.startswith("kata-init"):
        return "Kata Seed"
    parts = submission_id.rsplit("-", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return submission_id or None


def resolve_candidate_author(candidate_path: Path) -> str | None:
    metadata_path = candidate_path / SUBMISSION_METADATA_FILENAME
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return infer_submission_author(candidate_path.name)
    if not isinstance(payload, dict):
        return infer_submission_author(candidate_path.name)
    author = payload.get("author")
    if isinstance(author, str) and author.strip():
        return author.strip()
    return infer_submission_author(candidate_path.name)


def sha256_bundle_dict(files: dict[str, str]) -> str:
    import hashlib

    hasher = hashlib.sha256()
    for relative_path in sorted(files):
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(files[relative_path].encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()
