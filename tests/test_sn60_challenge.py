from __future__ import annotations

import json
from pathlib import Path

from kata.challenge import (
    SN60_MINER_LANE_ID,
    evaluate_sn60_promotion,
    load_challenge_summary,
    run_sn60_challenge,
)
from kata.evaluators.sn60_bitsec import (
    Sn60ProjectAggregate,
    Sn60ReplicaContext,
    Sn60ReplicaResult,
    Sn60VariantSummary,
)
from kata.lane_state import (
    load_benchmark_snapshot,
    load_challenge_state,
    load_promotion_record,
)


def write_bundle(root: Path, title: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.py").write_text(
        "def agent_main(project_dir=None, inference_api=None):\n"
        f"    return {{'vulnerabilities': [{{'title': '{title}'}}]}}\n",
        encoding="utf-8",
    )


def write_sandbox_source(root: Path) -> Path:
    benchmark_path = root / "validator" / "curated-highs-only-2025-08-08.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(
        json.dumps(
            [
                {
                    "project_id": "project-alpha",
                    "vulnerabilities": [{"title": "expected"}],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return benchmark_path


def test_run_sn60_challenge_decides_winner_and_records_lane_provenance(
    tmp_path: Path,
) -> None:
    sandbox_root = tmp_path / "sandbox"
    benchmark_path = write_sandbox_source(sandbox_root)
    frontier_root = tmp_path / "frontier"
    candidate_root = tmp_path / "candidate"
    write_bundle(frontier_root, "frontier")
    write_bundle(candidate_root, "candidate")

    def execute(context: Sn60ReplicaContext) -> dict[str, object]:
        return {
            "success": True,
            "report": {
                "vulnerabilities": [
                    {"title": f"{context.variant_name}-{context.replica_index}"}
                ],
            },
        }

    def evaluate(
        context: Sn60ReplicaContext,
        report_payload: dict[str, object],
    ) -> dict[str, object]:
        detection_rate = 1.0 if context.variant_name == "candidate" else 0.25
        return {
            "status": "success",
            "result": {
                "project": context.project_key,
                "timestamp": "2026-07-01T00:00:00+00:00",
                "total_expected": 4,
                "total_found": len(report_payload["report"]["vulnerabilities"]),
                "true_positives": int(detection_rate * 4),
                "false_negatives": 4 - int(detection_rate * 4),
                "false_positives": 0,
                "detection_rate": detection_rate,
                "precision": 1.0,
                "f1_score": detection_rate,
                "result": "PASS" if detection_rate == 1.0 else "FAIL",
            },
        }

    def screen(context: Sn60ReplicaContext) -> dict[str, object]:
        return {"success": True, "report": {"vulnerabilities": []}}

    summary = run_sn60_challenge(
        frontier_artifact_path=str(frontier_root),
        candidate_artifact_path=str(candidate_root),
        project_keys=["project-alpha"],
        candidate_submission_id="miner-sn60-1",
        output_root=str(tmp_path / "runs"),
        replicas_per_project=2,
        sandbox_root=str(sandbox_root),
        benchmark_file=str(benchmark_path),
        sandbox_commit="sandbox-commit-1",
        public_root=str(tmp_path / "public"),
        screening_hook=screen,
        execution_hook=execute,
        evaluation_hook=evaluate,
    )

    assert summary.mode == "miner"
    assert summary.promotion_ready
    assert summary.primary.variant_scores == {"frontier": 0.0, "candidate": 100.0}
    assert summary.primary.variant_successes == {"frontier": 0, "candidate": 1}
    assert summary.primary.total_task_weight == 1.0
    assert summary.primary.candidate_beats_frontier
    assert summary.primary_pool_fingerprint

    persisted = load_challenge_summary(
        str(Path(summary.manifest_path).with_name("challenge_summary.json"))
    )
    assert persisted.run_id == summary.run_id
    assert persisted.promotion_ready

    challenge_state = load_challenge_state(
        SN60_MINER_LANE_ID,
        public_root=str(tmp_path / "public"),
    )
    promotion_record = load_promotion_record(
        SN60_MINER_LANE_ID,
        public_root=str(tmp_path / "public"),
    )
    assert challenge_state.candidate_submission_id == "miner-sn60-1"
    assert challenge_state.freshness_fingerprint == summary.primary_pool_fingerprint
    assert promotion_record.final_winner == "candidate"
    assert promotion_record.final_metrics["promotion_ready"] is True
    assert promotion_record.final_metrics["candidate_aggregated_score"] == 1.0
    assert promotion_record.final_metrics["frontier_aggregated_score"] == 0.0
    assert promotion_record.final_metrics["candidate_average_score"] == 1.0
    assert promotion_record.pass_counts == {"frontier": 0, "candidate": 1}
    assert promotion_record.local_replica_scores["candidate"] == [1.0, 1.0]

    snapshot = load_benchmark_snapshot(
        SN60_MINER_LANE_ID,
        public_root=str(tmp_path / "public"),
    )
    assert snapshot.sandbox_commit_hash == "sandbox-commit-1"
    assert snapshot.project_keys == ["project-alpha"]
    assert snapshot.benchmark_dataset_id == "curated-highs-only-2025-08-08.json"
    assert snapshot.benchmark_dataset_hash
    assert snapshot.project_list_hash
    assert snapshot.container_images == ["ghcr.io/bitsec-ai/project-alpha:latest"]
    assert snapshot.scorer_version == "ScaBenchScorerV2"
    assert (
        Path(summary.manifest_path).with_name("screening_result.json")
    ).exists()


def test_evaluate_sn60_promotion_rejects_invalid_candidate() -> None:
    frontier = build_variant(
        "frontier", aggregated_score=0.5, codebase_pass_count=1, invalid_runs=0
    )
    candidate = build_variant(
        "candidate", aggregated_score=1.0, codebase_pass_count=2, invalid_runs=1
    )

    decision = evaluate_sn60_promotion(frontier=frontier, candidate=candidate)

    assert not decision.promotion_ready
    assert decision.final_winner == "frontier"
    assert decision.reason == "candidate has invalid SN60 replica runs"


def test_evaluate_sn60_promotion_uses_pass_count_as_score_tiebreaker() -> None:
    frontier = build_variant(
        "frontier",
        aggregated_score=0.5,
        codebase_pass_count=1,
        true_positives=4,
    )
    candidate = build_variant(
        "candidate",
        aggregated_score=0.5,
        codebase_pass_count=2,
        true_positives=4,
    )

    decision = evaluate_sn60_promotion(frontier=frontier, candidate=candidate)

    assert decision.promotion_ready
    assert decision.final_winner == "candidate"


def test_evaluate_sn60_promotion_uses_true_positives_as_final_tiebreaker() -> None:
    frontier = build_variant(
        "frontier",
        aggregated_score=0.5,
        codebase_pass_count=1,
        true_positives=4,
    )
    candidate = build_variant(
        "candidate",
        aggregated_score=0.5,
        codebase_pass_count=1,
        true_positives=6,
    )

    decision = evaluate_sn60_promotion(frontier=frontier, candidate=candidate)

    assert decision.promotion_ready
    assert decision.final_winner == "candidate"


def test_sn60_freshness_fingerprint_changes_with_sandbox_commit(tmp_path: Path) -> None:
    sandbox_root = tmp_path / "sandbox"
    benchmark_path = write_sandbox_source(sandbox_root)
    frontier_root = tmp_path / "frontier"
    candidate_root = tmp_path / "candidate"
    write_bundle(frontier_root, "frontier")
    write_bundle(candidate_root, "candidate")

    def execute(context: Sn60ReplicaContext) -> dict[str, object]:
        return {"success": True, "report": {"vulnerabilities": []}}

    def evaluate(
        context: Sn60ReplicaContext,
        report_payload: dict[str, object],
    ) -> dict[str, object]:
        return {
            "status": "success",
            "result": {
                "project": context.project_key,
                "timestamp": "2026-07-01T00:00:00+00:00",
                "total_expected": 1,
                "total_found": 0,
                "true_positives": 0,
                "false_negatives": 1,
                "false_positives": 0,
                "detection_rate": 0.0,
                "precision": 0.0,
                "f1_score": 0.0,
                "result": "FAIL",
            },
        }

    def screen(context: Sn60ReplicaContext) -> dict[str, object]:
        return {"success": True, "report": {"vulnerabilities": []}}

    first = run_sn60_challenge(
        frontier_artifact_path=str(frontier_root),
        candidate_artifact_path=str(candidate_root),
        project_keys=["project-alpha"],
        candidate_submission_id="miner-sn60-1",
        output_root=str(tmp_path / "runs-a"),
        replicas_per_project=1,
        sandbox_root=str(sandbox_root),
        benchmark_file=str(benchmark_path),
        sandbox_commit="commit-a",
        public_root=str(tmp_path / "public-a"),
        screening_hook=screen,
        execution_hook=execute,
        evaluation_hook=evaluate,
    )
    second = run_sn60_challenge(
        frontier_artifact_path=str(frontier_root),
        candidate_artifact_path=str(candidate_root),
        project_keys=["project-alpha"],
        candidate_submission_id="miner-sn60-1",
        output_root=str(tmp_path / "runs-b"),
        replicas_per_project=1,
        sandbox_root=str(sandbox_root),
        benchmark_file=str(benchmark_path),
        sandbox_commit="commit-b",
        public_root=str(tmp_path / "public-b"),
        screening_hook=screen,
        execution_hook=execute,
        evaluation_hook=evaluate,
    )

    assert first.primary_pool_fingerprint != second.primary_pool_fingerprint


def test_run_sn60_challenge_stops_before_duel_when_screening_fails(
    tmp_path: Path,
) -> None:
    sandbox_root = tmp_path / "sandbox"
    benchmark_path = write_sandbox_source(sandbox_root)
    frontier_root = tmp_path / "frontier"
    candidate_root = tmp_path / "candidate"
    write_bundle(frontier_root, "frontier")
    write_bundle(candidate_root, "candidate")

    execution_called = False

    def screen(context: Sn60ReplicaContext) -> dict[str, object]:
        return {"success": False, "error": "screening timeout"}

    def execute(context: Sn60ReplicaContext) -> dict[str, object]:
        nonlocal execution_called
        execution_called = True
        return {"success": True, "report": {"vulnerabilities": []}}

    summary = run_sn60_challenge(
        frontier_artifact_path=str(frontier_root),
        candidate_artifact_path=str(candidate_root),
        project_keys=["project-alpha"],
        candidate_submission_id="miner-sn60-1",
        output_root=str(tmp_path / "runs"),
        replicas_per_project=1,
        sandbox_root=str(sandbox_root),
        benchmark_file=str(benchmark_path),
        sandbox_commit="sandbox-commit-1",
        public_root=str(tmp_path / "public"),
        screening_hook=screen,
        execution_hook=execute,
        evaluation_hook=lambda context, report: {"status": "success", "result": {}},
    )

    assert not execution_called
    assert not summary.promotion_ready
    assert "candidate failed SN60 screening" in summary.promotion_reason
    assert Path(summary.manifest_path).name == "screening_result.json"

    challenge_summary_path = Path(summary.manifest_path).with_name("challenge_summary.json")
    assert challenge_summary_path.exists()

    challenge_state = load_challenge_state(
        SN60_MINER_LANE_ID,
        public_root=str(tmp_path / "public"),
    )
    promotion_record = load_promotion_record(
        SN60_MINER_LANE_ID,
        public_root=str(tmp_path / "public"),
    )
    assert challenge_state.screening_result["status"] == "failed"
    assert promotion_record.final_winner == "frontier"

    snapshot = load_benchmark_snapshot(
        SN60_MINER_LANE_ID,
        public_root=str(tmp_path / "public"),
    )
    assert snapshot.sandbox_commit_hash == "sandbox-commit-1"
    assert snapshot.project_keys == ["project-alpha"]


def build_variant(
    variant_name: str,
    *,
    aggregated_score: float,
    codebase_pass_count: int,
    true_positives: int = 0,
    invalid_runs: int = 0,
) -> Sn60VariantSummary:
    replica_results = [
        Sn60ReplicaResult(
            project_key="project-alpha",
            replica_index=1,
            report_path="/tmp/report.json",
            evaluation_path="/tmp/evaluation.json",
            execution_success=True,
            evaluation_status="success" if invalid_runs == 0 else "error",
            score=aggregated_score,
            detection_rate=aggregated_score,
            result="PASS" if codebase_pass_count else "FAIL",
            true_positives=true_positives,
            total_expected=4,
            total_found=true_positives,
        )
    ]
    return Sn60VariantSummary(
        variant_name=variant_name,
        artifact_path=f"/tmp/{variant_name}",
        artifact_hash=f"{variant_name}-hash",
        successful_runs=1 - invalid_runs,
        invalid_runs=invalid_runs,
        pass_count=codebase_pass_count,
        codebase_pass_count=codebase_pass_count,
        aggregated_score=aggregated_score,
        average_detection_rate=aggregated_score,
        true_positives=true_positives,
        total_expected=4,
        total_found=true_positives,
        project_summaries=[
            Sn60ProjectAggregate(
                project_key="project-alpha",
                replica_count=1,
                successful_runs=1 - invalid_runs,
                invalid_runs=invalid_runs,
                pass_count=codebase_pass_count,
                passed=bool(codebase_pass_count),
                average_detection_rate=aggregated_score,
                true_positives=true_positives,
                total_expected=4,
                total_found=true_positives,
            )
        ],
        replica_results=replica_results,
    )
