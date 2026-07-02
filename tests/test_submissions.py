from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kata.agent_bundle import (
    AGENT_ENTRY_FILENAME,
    AGENT_MANIFEST_FILENAME,
    write_agent_manifest,
)
from kata.challenge import current_primary_pool_fingerprint, run_sn60_challenge
from kata.frontier import (
    FRONTIER_SCHEMA_VERSION,
    PRIMARY_SELECTION_RANDOM_LIVE,
    FrontierManifest,
    FrontierModeConfig,
    write_frontier_manifest,
)
from kata.lane_state import (
    KING_STATE_SCHEMA_VERSION,
    LANE_METADATA_SCHEMA_VERSION,
    EvaluatorLaneMetadata,
    LaneKingState,
    load_benchmark_snapshot,
    load_challenge_state,
    load_lane_king_state,
    write_benchmark_snapshot,
    write_challenge_state,
    write_lane_king_state,
    write_lane_metadata,
)
from kata.provenance import pool_fingerprint, sha256_directory, sha256_text
from kata.submissions import (
    PR_ACTION_CLOSE_INVALID,
    PR_ACTION_CLOSE_LOSING,
    PR_ACTION_EVALUATE,
    PR_ACTION_MERGE,
    PR_ACTION_RERUN_STALE,
    decide_submission_action,
    evaluate_submission,
    hash_submission_bundle,
    init_submission,
    inspect_pull_request,
    promote_submission_result,
    validate_submission,
    verify_submission_result,
)

VALID_AGENT = (
    "def solve(repo_path, issue, model, api_base, api_key):\n"
    "    return {\"success\": True, \"diff\": \"\"}\n"
)
VALID_MINER_AGENT = (
    "def agent_main(project_dir=None, inference_api=None):\n"
    "    return {\"vulnerabilities\": []}\n"
)
SEED_MINER_AGENT = (
    "def agent_main(project_dir=None, inference_api=None):\n"
    "    return {\n"
    "        \"vulnerabilities\": [{\"title\": \"seed finding\"}],\n"
    "    }\n"
)
SEED_AGENT = (
    "def solve(repo_path, issue, model, api_base, api_key):\n"
    "    return {\"diff\": \"\"}\n"
)


def write_registry(
    root: Path,
    *,
    active_repo_packs: list[str] | None = None,
    default_repo_pack: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "registry_name": "test-registry",
        "benchmarks_dir": "benchmarks",
    }
    if active_repo_packs is not None:
        payload["active_repo_packs"] = active_repo_packs
    if default_repo_pack is not None:
        payload["default_repo_pack"] = default_repo_pack
    (root / "kata-benchmark-registry.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    (root / "benchmarks").mkdir(parents=True, exist_ok=True)


def write_frontier_pack(
    registry_root: Path,
    repo_pack: str,
    repo_ref: str,
    *,
    modes: tuple[str, ...] = ("contributor",),
) -> Path:
    pack_root = registry_root / "benchmarks" / repo_pack
    write_eval_task(pack_root / "task-a")
    frontier_modes: dict[str, FrontierModeConfig] = {}
    for mode in modes:
        artifact_root = pack_root / "agents" / mode
        baseline_root = artifact_root / "baseline"
        frontier_root = artifact_root / "frontier"
        baseline_root.mkdir(parents=True, exist_ok=True)
        frontier_root.mkdir(parents=True, exist_ok=True)
        seed_text = SEED_MINER_AGENT if mode == "miner" else SEED_AGENT
        write_agent_manifest(baseline_root / AGENT_MANIFEST_FILENAME)
        write_agent_manifest(frontier_root / AGENT_MANIFEST_FILENAME)
        (baseline_root / AGENT_ENTRY_FILENAME).write_text(seed_text, encoding="utf-8")
        (frontier_root / AGENT_ENTRY_FILENAME).write_text(seed_text, encoding="utf-8")
        frontier_modes[mode] = FrontierModeConfig(
            baseline_artifact=str(baseline_root.resolve()),
            frontier_artifact=str(frontier_root.resolve()),
            primary_tasks=["task-a"],
            holdout_tasks=[],
            evaluator_version="2026-06-29.v1",
            baseline_artifact_hash=sha256_directory(
                baseline_root,
                include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
            ),
            frontier_artifact_hash=sha256_directory(
                frontier_root,
                include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
            ),
            primary_pool_fingerprint=pool_fingerprint([pack_root / "task-a"]),
            holdout_pool_fingerprint=None,
            frontier_updated_at="2026-06-29T00:00:00+00:00",
            frontier_source="seed",
        )
    manifest = FrontierManifest(
        schema_version=FRONTIER_SCHEMA_VERSION,
        repo_ref=repo_ref,
        eval_pack=str(pack_root),
        updated_at="2026-06-29T00:00:00+00:00",
        modes=frontier_modes,
    )
    write_frontier_manifest(str(pack_root), manifest)
    return pack_root


def challenge_summary_payload(
    *,
    pack_root: Path,
    submission_root: Path,
    frontier_artifact_hash: str,
    candidate_artifact_hash: str,
    validator_model: str = "Qwen3-32B",
) -> dict[str, object]:
    baseline_artifact = pack_root / "agents" / "contributor" / "baseline"
    frontier_artifact = pack_root / "agents" / "contributor" / "frontier"
    candidate_artifact = submission_root
    primary_fingerprint = pool_fingerprint([pack_root / "task-a"])
    return {
        "schema_version": 4,
        "run_id": "challenge-1",
        "manifest_path": str((pack_root / "frontier.json").resolve()),
        "mode": "contributor",
        "evaluator_version": "2026-06-29.v1",
        "validator_model": validator_model,
        "baseline_artifact": str(baseline_artifact.resolve()),
        "frontier_artifact": str(frontier_artifact.resolve()),
        "candidate_artifact": str(candidate_artifact.resolve()),
        "baseline_artifact_hash": sha256_directory(
            baseline_artifact,
            include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
        ),
        "frontier_artifact_hash": frontier_artifact_hash,
        "candidate_artifact_hash": candidate_artifact_hash,
        "primary_pool_fingerprint": primary_fingerprint,
        "holdout_pool_fingerprint": None,
        "promotion_margin_points": 3.0,
        "created_at": "2026-06-29T00:00:00+00:00",
        "primary": {
            "task_ids": ["task-a"],
            "eval_run_summary": "run_summary.json",
            "total_task_weight": 1.0,
            "variant_successes": {"baseline": 0, "frontier": 0, "candidate": 1},
            "variant_invalid_tasks": {"baseline": 0, "frontier": 0, "candidate": 0},
            "variant_scores": {"baseline": 0.0, "frontier": 0.0, "candidate": 100.0},
            "candidate_beats_frontier": True,
            "candidate_score_delta": 100.0,
        },
        "holdout": None,
        "promotion_ready": True,
        "promotion_reason": "candidate cleared the primary score margin",
    }


def write_eval_task(task_root: Path) -> None:
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "task.md").write_text("# task\n", encoding="utf-8")
    (task_root / "repo_ref.txt").write_text(
        "https://github.com/example/repo.git@test\n",
        encoding="utf-8",
    )
    (task_root / "checks.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ntrue\n",
        encoding="utf-8",
    )
    (task_root / "rubric.md").write_text("# rubric\n", encoding="utf-8")
    (task_root / "allowed_paths.txt").write_text("src/\n", encoding="utf-8")
    (task_root / "forbidden_paths.txt").write_text("", encoding="utf-8")


def test_validate_submission_accepts_scoped_submission_pr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-1",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")

    result = validate_submission(
        str(submission_root),
        changed_paths=[
            "submissions/example__repo/contributor/miner-1/agent.py",
            "submissions/example__repo/contributor/miner-1/submission.json",
        ],
        repo_root=str(repo_root),
    )

    assert result.is_valid
    assert result.reasons == []


def test_validate_submission_rejects_symlink_before_reading_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-symlink",
        output_root=str(repo_root / "submissions"),
    )
    outside_target = tmp_path / "outside_agent.py"
    outside_target.write_text(
        "KATA_VALIDATOR_API_KEY\nthis is not valid python\n",
        encoding="utf-8",
    )
    agent_path = submission_root / "agent.py"
    agent_path.unlink()
    agent_path.symlink_to(outside_target)

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert result.reasons == ["Submission bundle must not contain symlinks: agent.py"]
    assert result.off_scope_paths == []


def test_validate_submission_rejects_off_scope_pr_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-2",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")

    result = validate_submission(
        str(submission_root),
        changed_paths=[
            "submissions/example__repo/contributor/miner-2/agent.py",
            "README.md",
        ],
        repo_root=str(repo_root),
    )

    assert not result.is_valid
    assert "Submission PR touches paths outside the allowed submission scope." in result.reasons
    assert result.off_scope_paths == ["README.md"]


def test_validate_submission_rejects_scaffold_agent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-2b",
        output_root=str(repo_root / "submissions"),
    )

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert "Submission agent still contains scaffold placeholder text." in result.reasons


def test_validate_submission_rejects_missing_solve(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-nosolve",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text("print('hello')\n", encoding="utf-8")

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert "Submission agent must define solve(...)." in result.reasons


def test_validate_submission_accepts_miner_submission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(
        registry_root,
        "example__repo",
        "/tmp/repo",
        modes=("contributor", "miner"),
    )
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="miner",
        submission_id="miner-sn60-1",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_MINER_AGENT, encoding="utf-8")

    result = validate_submission(
        str(submission_root),
        changed_paths=[
            "submissions/example__repo/miner/miner-sn60-1/agent.py",
            "submissions/example__repo/miner/miner-sn60-1/submission.json",
        ],
        repo_root=str(repo_root),
    )

    assert result.is_valid
    assert result.reasons == []


def test_validate_submission_rejects_missing_agent_main_for_miner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(
        registry_root,
        "example__repo",
        "/tmp/repo",
        modes=("contributor", "miner"),
    )
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="miner",
        submission_id="miner-no-agent-main",
        output_root=str(tmp_path / "Kata" / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert "Submission agent must define agent_main(...)." in result.reasons


def test_validate_submission_rejects_commented_agent_main_for_miner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(
        registry_root,
        "example__repo",
        "/tmp/repo",
        modes=("contributor", "miner"),
    )
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="miner",
        submission_id="miner-commented-agent-main",
        output_root=str(tmp_path / "Kata" / "submissions"),
    )
    (submission_root / "agent.py").write_text(
        "# def agent_main(project_dir=None, inference_api=None):\n"
        "print('not a real entrypoint')\n",
        encoding="utf-8",
    )

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert "Submission agent must define agent_main(...)." in result.reasons


def test_validate_submission_rejects_required_agent_main_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(
        registry_root,
        "example__repo",
        "/tmp/repo",
        modes=("contributor", "miner"),
    )
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="miner",
        submission_id="miner-required-arg",
        output_root=str(tmp_path / "Kata" / "submissions"),
    )
    (submission_root / "agent.py").write_text(
        "def agent_main(project_dir):\n"
        "    return {\"vulnerabilities\": []}\n",
        encoding="utf-8",
    )

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert "Submission agent must support no-argument invocation: agent_main()." in result.reasons


def test_validate_submission_rejects_helper_files_for_miner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(
        registry_root,
        "example__repo",
        "/tmp/repo",
        modes=("contributor", "miner"),
    )
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="miner",
        submission_id="miner-helpers",
        output_root=str(tmp_path / "Kata" / "submissions"),
    )
    helpers_root = submission_root / "helpers"
    helpers_root.mkdir()
    (helpers_root / "planner.py").write_text("def plan():\n    return 'ok'\n", encoding="utf-8")
    (submission_root / "agent.py").write_text(VALID_MINER_AGENT, encoding="utf-8")

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert any("do not support helper files in V1" in reason for reason in result.reasons)


def test_validate_submission_rejects_non_bitsec_report_contract_for_miner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(
        registry_root,
        "example__repo",
        "/tmp/repo",
        modes=("contributor", "miner"),
    )
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="miner",
        submission_id="miner-bad-report",
        output_root=str(tmp_path / "Kata" / "submissions"),
    )
    (submission_root / "agent.py").write_text(
        "def agent_main(project_dir=None, inference_api=None):\n"
        "    return {\"success\": True}\n",
        encoding="utf-8",
    )

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert any("Bitsec-compatible report" in reason for reason in result.reasons)


def test_evaluate_submission_routes_sn60_miner_to_sn60_challenge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root, active_repo_packs=["sn60__bitsec"])
    pack_root = write_frontier_pack(
        registry_root,
        "sn60__bitsec",
        "/tmp/repo",
        modes=("miner",),
    )
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    # No pack registry in this root: exercises the legacy frontier-manifest fallback.
    monkeypatch.setenv("KATA_ROOT", str(tmp_path / "kata-root"))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="sn60__bitsec",
        mode="miner",
        submission_id="miner-sn60-route",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_MINER_AGENT, encoding="utf-8")
    sentinel = object()
    calls: dict[str, object] = {}

    def fake_run_sn60_challenge(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr("kata.submissions.run_sn60_challenge", fake_run_sn60_challenge)

    summary = evaluate_submission(
        str(submission_root),
        agent_command="/bin/true",
        output_root=str(tmp_path / "runs"),
        sn60_project_keys=["project-a"],
        sn60_replicas_per_project=2,
        sn60_sandbox_root=str(tmp_path / "sandbox"),
        sn60_benchmark_file="benchmark.json",
        sn60_sandbox_commit="sandbox-commit",
    )

    assert summary is sentinel
    assert calls["frontier_artifact_path"] == str(
        (pack_root / "agents" / "miner" / "frontier").resolve()
    )
    assert calls["candidate_artifact_path"] == str(submission_root.resolve())
    assert calls["project_keys"] == ["project-a"]
    assert calls["candidate_submission_id"] == "miner-sn60-route"
    assert calls["lane_id"] == "sn60__bitsec"
    assert calls["replicas_per_project"] == 2
    assert calls["sandbox_root"] == str(tmp_path / "sandbox")
    assert calls["benchmark_file"] == "benchmark.json"
    assert calls["sandbox_commit"] == "sandbox-commit"


def test_evaluate_submission_keeps_contributor_on_legacy_frontier_challenge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-legacy-route",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")
    sentinel = object()
    calls: dict[str, object] = {}

    def fake_run_frontier_challenge(**kwargs):
        calls.update(kwargs)
        return sentinel

    def fake_run_sn60_challenge(**_kwargs):
        raise AssertionError("legacy contributor lane must not use SN60 challenge")

    monkeypatch.setattr("kata.submissions.run_frontier_challenge", fake_run_frontier_challenge)
    monkeypatch.setattr("kata.submissions.run_sn60_challenge", fake_run_sn60_challenge)

    summary = evaluate_submission(
        str(submission_root),
        agent_command="/bin/true",
        output_root=str(tmp_path / "runs"),
        sn60_project_keys=["project-a"],
    )

    assert summary is sentinel
    assert calls["eval_pack_path"] == "example__repo"
    assert calls["mode"] == "contributor"
    assert calls["candidate_artifact_path"] == str(submission_root.resolve())
    assert calls["agent_command"] == "/bin/true"


def test_verify_submission_result_accepts_sn60_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root, active_repo_packs=["sn60__bitsec"])
    pack_root = write_frontier_pack(
        registry_root,
        "sn60__bitsec",
        "/tmp/repo",
        modes=("miner",),
    )
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="sn60__bitsec",
        mode="miner",
        submission_id="miner-sn60-verify",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_MINER_AGENT, encoding="utf-8")
    frontier_root = pack_root / "agents" / "miner" / "frontier"
    summary_path = tmp_path / "challenge_summary.json"
    duel_summary_path = tmp_path / "duel_summary.json"
    duel_summary_path.write_text("{}\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "run_id": "sn60-run",
                "manifest_path": str(duel_summary_path),
                "mode": "miner",
                "evaluator_version": "ScaBenchScorerV2@test",
                "validator_model": "sn60-bitsec-sandbox",
                "frontier_artifact": str(frontier_root.resolve()),
                "candidate_artifact": str(submission_root.resolve()),
                "frontier_artifact_hash": sha256_directory(
                    frontier_root,
                    include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
                ),
                "candidate_artifact_hash": hash_submission_bundle(submission_root),
                "primary_pool_fingerprint": "sn60-fingerprint",
                "holdout_pool_fingerprint": None,
                "promotion_margin_points": 0.0,
                "holdout_promotion_margin_points": 0.0,
                "created_at": "2026-07-01T00:00:00+00:00",
                "primary": {
                    "task_ids": ["project-a"],
                    "eval_run_summary": str(duel_summary_path),
                    "total_task_weight": 2.0,
                    "variant_successes": {"frontier": 1, "candidate": 2},
                    "variant_invalid_tasks": {"frontier": 0, "candidate": 0},
                    "variant_scores": {"frontier": 50.0, "candidate": 80.0},
                    "candidate_beats_frontier": True,
                    "candidate_score_delta": 30.0,
                },
                "holdout": None,
                "promotion_ready": True,
                "promotion_reason": "sn60__bitsec: candidate beat the current SN60 king",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = verify_submission_result(str(submission_root), str(summary_path))

    assert result.auto_merge_ready
    assert result.benchmark_is_current
    assert result.reasons == []


def test_init_submission_creates_agent_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))

    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-manifest",
        output_root=str(tmp_path / "Kata" / "submissions"),
    )

    assert (submission_root / AGENT_MANIFEST_FILENAME).exists()


def test_validate_submission_accepts_helper_only_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-helper",
        output_root=str(repo_root / "submissions"),
    )
    helpers_root = submission_root / "helpers"
    helpers_root.mkdir()
    (helpers_root / "planner.py").write_text("def plan():\n    return 'ok'\n", encoding="utf-8")
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")

    result = validate_submission(
        str(submission_root),
        changed_paths=[
            "submissions/example__repo/contributor/miner-helper/helpers/planner.py",
        ],
        repo_root=str(repo_root),
    )

    assert result.is_valid


def test_validate_submission_rejects_unexpected_bundle_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-badfile",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")
    (submission_root / "notes.txt").write_text("bad\n", encoding="utf-8")

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert "Submission bundle contains unsupported files: notes.txt" in result.reasons


def test_validate_submission_ignores_local_python_cache_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-cache",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")
    pycache_root = submission_root / "__pycache__"
    pycache_root.mkdir()
    (pycache_root / "agent.cpython-312.pyc").write_bytes(b"cache")

    result = validate_submission(str(submission_root))

    assert result.is_valid


def test_validate_submission_rejects_frontier_copycat_agent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-copycat",
        output_root=str(repo_root / "submissions"),
    )
    helpers_root = submission_root / "helpers"
    helpers_root.mkdir()
    (helpers_root / "extra.py").write_text("VALUE = 1\n", encoding="utf-8")
    (submission_root / "agent.py").write_text(SEED_AGENT, encoding="utf-8")

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert (
        "Submission agent duplicates the current frontier agent implementation."
        in result.reasons
    )


def test_validate_submission_rejects_validator_env_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-env",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(
        "import os\n\n"
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    key = os.environ.get('OPENAI_API_KEY', '')\n"
        "    return {\"success\": bool(key), \"diff\": \"\"}\n",
        encoding="utf-8",
    )

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert any("OPENAI_API_KEY" in reason for reason in result.reasons)


def test_validate_submission_rejects_hardcoded_secret_like_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-secret",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    token = 'sk-1234567890abcdef'\n"
        "    return {\"success\": bool(token), \"diff\": \"\"}\n",
        encoding="utf-8",
    )

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert any("hardcoded secret token" in reason for reason in result.reasons)


def test_validate_submission_rejects_invalid_helper_python_syntax(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-badhelper",
        output_root=str(repo_root / "submissions"),
    )
    helpers_root = submission_root / "helpers"
    helpers_root.mkdir()
    (helpers_root / "planner.py").write_text("def plan(:\n    return 'bad'\n", encoding="utf-8")
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert any(
        "Submission bundle contains invalid Python syntax in helpers/planner.py" in reason
        for reason in result.reasons
    )


def test_validate_submission_rejects_solve_signature_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-signature",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(
        "def solve(issue, repo_path, model, api_base, api_key):\n"
        "    return {\"success\": True, \"diff\": \"\"}\n",
        encoding="utf-8",
    )

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert any("validator solve signature" in reason for reason in result.reasons)


def test_validate_submission_rejects_sampling_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-sampling",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    helper(model=model, temperature=0.7)\n"
        "    return {\"success\": True, \"diff\": \"\"}\n",
        encoding="utf-8",
    )

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert any("sampling parameters" in reason for reason in result.reasons)


def test_validate_submission_rejects_direct_provider_endpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-provider-url",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(
        "API_URL = 'https://api.openai.com/v1/chat/completions'\n\n"
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"success\": True, \"diff\": API_URL}\n",
        encoding="utf-8",
    )

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert any("provider endpoints" in reason for reason in result.reasons)


def test_validate_submission_reports_malformed_metadata_instead_of_crashing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-bad-metadata",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")
    (submission_root / "submission.json").write_text("{\"schema_version\": 2}\n", encoding="utf-8")

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert any("missing required field: repo_pack" in reason for reason in result.reasons)


def test_validate_submission_rejects_inactive_repo_pack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(
        registry_root,
        active_repo_packs=["e35ventura__taopedia-articles"],
        default_repo_pack="e35ventura__taopedia-articles",
    )
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = repo_root / "submissions" / "example__repo" / "contributor" / "miner-inactive"
    submission_root.mkdir(parents=True, exist_ok=True)
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")
    (submission_root / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "repo_pack": "example__repo",
                "mode": "contributor",
                "submission_id": "miner-inactive",
                "created_at": "2026-06-29T00:00:00+00:00",
                "author": "miner",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_submission(str(submission_root))

    assert not result.is_valid
    assert any("Repo pack is not active" in reason for reason in result.reasons)


def test_init_submission_rejects_inactive_repo_pack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(
        registry_root,
        active_repo_packs=["e35ventura__taopedia-articles"],
        default_repo_pack="e35ventura__taopedia-articles",
    )
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))

    try:
        init_submission(
            repo_pack="example__repo",
            mode="contributor",
            submission_id="miner-inactive-init",
            output_root=str(tmp_path / "Kata" / "submissions"),
        )
    except ValueError as exc:
        assert "Repo pack is not active" in str(exc)
    else:
        raise AssertionError("Expected init_submission to reject inactive repo pack.")


def test_inspect_pull_request_rejects_non_submission_pr(tmp_path: Path) -> None:
    repo_root = tmp_path / "Kata"
    repo_root.mkdir()

    result = inspect_pull_request(
        repo_root=str(repo_root),
        changed_paths=["README.md"],
    )

    assert result.action == PR_ACTION_CLOSE_INVALID
    assert result.submission_path is None


def test_inspect_pull_request_accepts_single_submission_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    repo_root.mkdir()
    submission_root = repo_root / "submissions" / "example__repo" / "contributor" / "miner-9"

    result = inspect_pull_request(
        repo_root=str(repo_root),
        changed_paths=[
            "submissions/example__repo/contributor/miner-9/agent.py",
            "submissions/example__repo/contributor/miner-9/submission.json",
        ],
    )

    assert result.action == PR_ACTION_EVALUATE
    assert result.submission_path == str(submission_root.resolve())


def test_inspect_pull_request_rejects_inactive_repo_pack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(
        registry_root,
        active_repo_packs=["e35ventura__taopedia-articles"],
        default_repo_pack="e35ventura__taopedia-articles",
    )
    write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    repo_root.mkdir()

    result = inspect_pull_request(
        repo_root=str(repo_root),
        changed_paths=[
            "submissions/example__repo/contributor/miner-9/agent.py",
            "submissions/example__repo/contributor/miner-9/submission.json",
        ],
    )

    assert result.action == PR_ACTION_CLOSE_INVALID
    assert any("Repo pack is not active" in reason for reason in result.reasons)


def test_verify_submission_result_accepts_current_promotion_ready_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-3",
        output_root=str(repo_root / "submissions"),
    )
    candidate_text = (
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"success\": True, \"diff\": \"winner\"}\n"
    )
    (submission_root / "agent.py").write_text(candidate_text, encoding="utf-8")
    candidate_hash = hash_submission_bundle(submission_root)
    summary_path = tmp_path / "challenge_summary.json"
    summary_path.write_text(
        json.dumps(
            challenge_summary_payload(
                pack_root=pack_root,
                submission_root=submission_root,
                frontier_artifact_hash=sha256_directory(
                    pack_root / "agents" / "contributor" / "frontier",
                    include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
                ),
                candidate_artifact_hash=candidate_hash,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = verify_submission_result(str(submission_root), str(summary_path))

    assert result.submission_matches_challenge
    assert result.frontier_is_current
    assert result.benchmark_is_current
    assert result.auto_merge_ready
    assert result.reasons == []


def test_verify_submission_result_detects_stale_frontier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-4",
        output_root=str(repo_root / "submissions"),
    )
    candidate_text = (
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"success\": True, \"diff\": \"winner\"}\n"
    )
    (submission_root / "agent.py").write_text(candidate_text, encoding="utf-8")
    summary_path = tmp_path / "challenge_summary.json"
    summary_path.write_text(
        json.dumps(
            challenge_summary_payload(
                pack_root=pack_root,
                submission_root=submission_root,
                frontier_artifact_hash=sha256_text("# older-frontier\n"),
                candidate_artifact_hash=hash_submission_bundle(submission_root),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = verify_submission_result(str(submission_root), str(summary_path))

    assert not result.frontier_is_current
    assert not result.auto_merge_ready
    assert "Challenge result is stale because the frontier artifact has changed." in result.reasons


def test_verify_submission_result_detects_validator_model_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    monkeypatch.setenv("KATA_VALIDATOR_MODEL", "Qwen3-32B")
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-model",
        output_root=str(repo_root / "submissions"),
    )
    candidate_text = (
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"success\": True, \"diff\": \"winner\"}\n"
    )
    (submission_root / "agent.py").write_text(candidate_text, encoding="utf-8")
    summary_path = tmp_path / "challenge_summary.json"
    summary_path.write_text(
        json.dumps(
            challenge_summary_payload(
                pack_root=pack_root,
                submission_root=submission_root,
                frontier_artifact_hash=sha256_directory(
                    pack_root / "agents" / "contributor" / "frontier",
                    include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
                ),
                candidate_artifact_hash=hash_submission_bundle(submission_root),
                validator_model="OldModel-32B",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = verify_submission_result(str(submission_root), str(summary_path))

    assert not result.benchmark_is_current
    assert not result.auto_merge_ready
    assert "Challenge result is stale because the validator model has changed." in result.reasons


def test_verify_submission_result_detects_selected_task_change_in_random_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    write_eval_task(pack_root / "task-b")
    artifact_root = pack_root / "agents" / "contributor"
    manifest = FrontierManifest(
        schema_version=FRONTIER_SCHEMA_VERSION,
        repo_ref="/tmp/repo",
        eval_pack=str(pack_root),
        updated_at="2026-06-29T00:00:00+00:00",
        modes={
            "contributor": FrontierModeConfig(
                baseline_artifact=str((artifact_root / "baseline").resolve()),
                frontier_artifact=str((artifact_root / "frontier").resolve()),
                primary_tasks=[],
                primary_task_count=1,
                primary_selection=PRIMARY_SELECTION_RANDOM_LIVE,
                holdout_tasks=[],
                evaluator_version="2026-06-29.v1",
                baseline_artifact_hash=sha256_directory(
                    artifact_root / "baseline",
                    include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
                ),
                frontier_artifact_hash=sha256_directory(
                    artifact_root / "frontier",
                    include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
                ),
                primary_pool_fingerprint=pool_fingerprint([pack_root / "task-a"]),
                holdout_pool_fingerprint=None,
                frontier_updated_at="2026-06-29T00:00:00+00:00",
                frontier_source="seed",
            )
        },
    )
    write_frontier_manifest(str(pack_root), manifest)
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-random",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_AGENT, encoding="utf-8")
    summary_payload = challenge_summary_payload(
        pack_root=pack_root,
        submission_root=submission_root,
        frontier_artifact_hash=sha256_directory(
            pack_root / "agents" / "contributor" / "frontier",
            include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
        ),
        candidate_artifact_hash=hash_submission_bundle(submission_root),
    )
    summary_payload["primary_pool_fingerprint"] = pool_fingerprint([pack_root / "task-a"])
    summary_path = tmp_path / "challenge_summary.json"
    summary_path.write_text(json.dumps(summary_payload) + "\n", encoding="utf-8")
    (pack_root / "task-a" / "task.md").write_text("# changed task\n", encoding="utf-8")

    result = verify_submission_result(str(submission_root), str(summary_path))

    assert not result.benchmark_is_current
    assert "Challenge result is stale because the benchmark lane has changed." in result.reasons


def test_random_live_primary_fingerprint_uses_selected_tasks(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    write_eval_task(pack_root / "task-b")
    mode_config = FrontierModeConfig(
        baseline_artifact=str((pack_root / "agents" / "contributor" / "baseline").resolve()),
        frontier_artifact=str((pack_root / "agents" / "contributor" / "frontier").resolve()),
        primary_tasks=[],
        primary_task_count=1,
        primary_selection=PRIMARY_SELECTION_RANDOM_LIVE,
        holdout_tasks=[],
    )

    task_a_fingerprint = current_primary_pool_fingerprint(
        str(pack_root),
        mode_config,
        selected_task_ids=["task-a"],
    )
    task_b_fingerprint = current_primary_pool_fingerprint(
        str(pack_root),
        mode_config,
        selected_task_ids=["task-b"],
    )

    assert task_a_fingerprint == pool_fingerprint([pack_root / "task-a"])
    assert task_b_fingerprint == pool_fingerprint([pack_root / "task-b"])
    assert task_a_fingerprint != task_b_fingerprint


def test_decide_submission_action_returns_merge_for_verified_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-merge",
        output_root=str(repo_root / "submissions"),
    )
    candidate_text = (
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"success\": True, \"diff\": \"winner\"}\n"
    )
    (submission_root / "agent.py").write_text(candidate_text, encoding="utf-8")
    summary_path = tmp_path / "challenge_summary.json"
    summary_path.write_text(
        json.dumps(
            challenge_summary_payload(
                pack_root=pack_root,
                submission_root=submission_root,
                frontier_artifact_hash=sha256_directory(
                    pack_root / "agents" / "contributor" / "frontier",
                    include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
                ),
                candidate_artifact_hash=hash_submission_bundle(submission_root),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = decide_submission_action(str(submission_root), str(summary_path))

    assert result.action == PR_ACTION_MERGE
    assert result.auto_merge_ready


def test_decide_submission_action_returns_rerun_for_stale_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-rerun",
        output_root=str(repo_root / "submissions"),
    )
    candidate_text = (
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"success\": True, \"diff\": \"winner\"}\n"
    )
    (submission_root / "agent.py").write_text(candidate_text, encoding="utf-8")
    summary_path = tmp_path / "challenge_summary.json"
    summary_path.write_text(
        json.dumps(
            challenge_summary_payload(
                pack_root=pack_root,
                submission_root=submission_root,
                frontier_artifact_hash=sha256_text("# stale-frontier\n"),
                candidate_artifact_hash=hash_submission_bundle(submission_root),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = decide_submission_action(str(submission_root), str(summary_path))

    assert result.action == PR_ACTION_RERUN_STALE


def test_decide_submission_action_returns_close_for_loser(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-lose",
        output_root=str(repo_root / "submissions"),
    )
    candidate_text = (
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"success\": False, \"diff\": \"loser\"}\n"
    )
    (submission_root / "agent.py").write_text(candidate_text, encoding="utf-8")
    summary_path = tmp_path / "challenge_summary.json"
    payload = challenge_summary_payload(
        pack_root=pack_root,
        submission_root=submission_root,
        frontier_artifact_hash=sha256_directory(
            pack_root / "agents" / "contributor" / "frontier",
            include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
        ),
        candidate_artifact_hash=hash_submission_bundle(submission_root),
    )
    payload["promotion_ready"] = False
    payload["promotion_reason"] = "candidate did not beat the current frontier on the primary score"
    payload["primary"]["variant_successes"] = {
        "baseline": 0,
        "frontier": 1,
        "candidate": 0,
    }
    payload["primary"]["variant_scores"] = {
        "baseline": 0.0,
        "frontier": 100.0,
        "candidate": 0.0,
    }
    payload["primary"]["candidate_beats_frontier"] = False
    payload["primary"]["candidate_score_delta"] = -100.0
    summary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = decide_submission_action(str(submission_root), str(summary_path))

    assert result.action == PR_ACTION_CLOSE_LOSING


def test_promote_submission_result_updates_frontier_for_verified_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-promote",
        output_root=str(repo_root / "submissions"),
    )
    candidate_text = (
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"success\": True, \"diff\": \"promoted\"}\n"
    )
    (submission_root / "agent.py").write_text(candidate_text, encoding="utf-8")
    candidate_hash = hash_submission_bundle(submission_root)
    summary_path = tmp_path / "challenge_summary.json"
    summary_path.write_text(
        json.dumps(
            challenge_summary_payload(
                pack_root=pack_root,
                submission_root=submission_root,
                frontier_artifact_hash=sha256_directory(
                    pack_root / "agents" / "contributor" / "frontier",
                    include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
                ),
                candidate_artifact_hash=candidate_hash,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = promote_submission_result(str(submission_root), str(summary_path))

    frontier_agent = pack_root / "agents" / "contributor" / "frontier" / "agent.py"
    assert frontier_agent.read_text(encoding="utf-8") == candidate_text
    assert (
        manifest.modes["contributor"].frontier_artifact_hash
        == candidate_hash
    )
    assert manifest.modes["contributor"].frontier_source == "challenge-1"


def test_promote_submission_result_updates_public_king_mirror(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    public_kata_root = tmp_path / "public-kata"
    public_kata_root.mkdir()
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-public-king",
        output_root=str(public_kata_root / "submissions"),
    )
    candidate_text = (
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"success\": True, \"diff\": \"public-king\"}\n"
    )
    (submission_root / "agent.py").write_text(candidate_text, encoding="utf-8")
    candidate_hash = hash_submission_bundle(submission_root)
    summary_path = tmp_path / "challenge_summary.json"
    summary_path.write_text(
        json.dumps(
            challenge_summary_payload(
                pack_root=pack_root,
                submission_root=submission_root,
                frontier_artifact_hash=sha256_directory(
                    pack_root / "agents" / "contributor" / "frontier",
                    include=[AGENT_ENTRY_FILENAME, AGENT_MANIFEST_FILENAME],
                ),
                candidate_artifact_hash=candidate_hash,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    promote_submission_result(
        str(submission_root),
        str(summary_path),
        public_root=str(public_kata_root),
    )

    public_agent = public_kata_root / "kings" / "example__repo" / "contributor" / "agent.py"
    public_metadata = (
        public_kata_root / "kings" / "example__repo" / "contributor" / "king.json"
    )
    assert public_agent.read_text(encoding="utf-8") == candidate_text
    assert (
        json.loads(public_metadata.read_text(encoding="utf-8"))["submission_id"]
        == "miner-public-king"
    )


def test_promote_submission_result_rejects_stale_submission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry(registry_root)
    pack_root = write_frontier_pack(registry_root, "example__repo", "/tmp/repo")
    monkeypatch.setenv("KATA_BENCHMARKS_ROOT", str(registry_root))
    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="example__repo",
        mode="contributor",
        submission_id="miner-stale-promote",
        output_root=str(repo_root / "submissions"),
    )
    candidate_text = (
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"success\": True, \"diff\": \"stale\"}\n"
    )
    (submission_root / "agent.py").write_text(candidate_text, encoding="utf-8")
    summary_path = tmp_path / "challenge_summary.json"
    summary_path.write_text(
        json.dumps(
            challenge_summary_payload(
                pack_root=pack_root,
                submission_root=submission_root,
                frontier_artifact_hash=sha256_text("# stale-frontier\n"),
                candidate_artifact_hash=hash_submission_bundle(submission_root),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        promote_submission_result(str(submission_root), str(summary_path))
    except ValueError as exc:
        assert "Submission is not safe to promote." in str(exc)
        assert "frontier artifact has changed" in str(exc)
    else:
        raise AssertionError("Expected stale submission promotion to be rejected.")


def write_evaluator_lane(public_root: Path, *, active: bool = True) -> None:
    write_lane_metadata(
        EvaluatorLaneMetadata(
            schema_version=LANE_METADATA_SCHEMA_VERSION,
            lane_id="sn60__bitsec",
            repo_pack="sn60__bitsec",
            mode="miner",
            evaluator_id="sn60_bitsec",
            evaluator_policy_version="v1",
            active=active,
            created_at="2026-07-01T00:00:00+00:00",
            updated_at="2026-07-01T00:00:00+00:00",
        ),
        public_root=str(public_root),
    )


def test_validate_submission_accepts_miner_submission_for_registry_lane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_root = tmp_path / "kata-root"
    write_evaluator_lane(public_root)
    monkeypatch.setenv("KATA_ROOT", str(public_root))
    monkeypatch.delenv("KATA_BENCHMARKS_ROOT", raising=False)

    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="sn60__bitsec",
        mode="miner",
        submission_id="alice-20260702-01",
        output_root=str(repo_root / "submissions"),
        author="alice",
    )
    (submission_root / "agent.py").write_text(VALID_MINER_AGENT, encoding="utf-8")

    result = validate_submission(str(submission_root), repo_root=str(repo_root))

    assert result.reasons == []
    assert result.is_valid
    assert result.evaluator_id == "sn60_bitsec"


def test_init_submission_rejects_inactive_registry_lane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_root = tmp_path / "kata-root"
    write_evaluator_lane(public_root, active=False)
    monkeypatch.setenv("KATA_ROOT", str(public_root))
    monkeypatch.delenv("KATA_BENCHMARKS_ROOT", raising=False)

    with pytest.raises(ValueError, match="not active in the pack registry"):
        init_submission(
            repo_pack="sn60__bitsec",
            mode="miner",
            submission_id="alice-20260702-01",
            output_root=str(tmp_path / "Kata" / "submissions"),
        )


def test_validate_submission_rejects_copy_of_lane_king(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_root = tmp_path / "kata-root"
    write_evaluator_lane(public_root)
    monkeypatch.setenv("KATA_ROOT", str(public_root))
    monkeypatch.delenv("KATA_BENCHMARKS_ROOT", raising=False)

    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="sn60__bitsec",
        mode="miner",
        submission_id="alice-20260702-01",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_MINER_AGENT, encoding="utf-8")

    write_lane_king_state(
        "sn60__bitsec",
        LaneKingState(
            schema_version=KING_STATE_SCHEMA_VERSION,
            current_king_submission_id="king-1",
            current_king_artifact_hash=hash_submission_bundle(submission_root),
            promotion_source_pr=None,
            promotion_timestamp=None,
            updated_at="2026-07-01T00:00:00+00:00",
        ),
        public_root=str(public_root),
    )

    result = validate_submission(str(submission_root), repo_root=str(repo_root))

    assert any("exact copy of the current lane king" in reason for reason in result.reasons)
    assert not result.is_valid


def seed_lane_king(public_root: Path, repo_pack: str) -> Path:
    king_root = public_root / "kings" / repo_pack / "miner"
    king_root.mkdir(parents=True)
    (king_root / "agent.py").write_text(SEED_MINER_AGENT, encoding="utf-8")
    write_agent_manifest(king_root / AGENT_MANIFEST_FILENAME)
    return king_root


def test_evaluate_submission_uses_seeded_lane_king_for_registry_lane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_root = tmp_path / "kata-root"
    write_evaluator_lane(public_root)
    monkeypatch.setenv("KATA_ROOT", str(public_root))
    monkeypatch.delenv("KATA_BENCHMARKS_ROOT", raising=False)
    king_root = seed_lane_king(public_root, "sn60__bitsec")

    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="sn60__bitsec",
        mode="miner",
        submission_id="alice-20260702-02",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_MINER_AGENT, encoding="utf-8")

    sentinel = object()
    calls: dict[str, object] = {}

    def fake_run_sn60_challenge(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr("kata.submissions.run_sn60_challenge", fake_run_sn60_challenge)

    summary = evaluate_submission(
        str(submission_root),
        agent_command="/bin/true",
        sn60_project_keys=["project-a"],
    )

    assert summary is sentinel
    assert calls["frontier_artifact_path"] == str(king_root.resolve())
    assert calls["lane_id"] == "sn60__bitsec"


def test_evaluate_submission_requires_seeded_king_for_registry_lane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_root = tmp_path / "kata-root"
    write_evaluator_lane(public_root)
    monkeypatch.setenv("KATA_ROOT", str(public_root))
    monkeypatch.delenv("KATA_BENCHMARKS_ROOT", raising=False)

    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="sn60__bitsec",
        mode="miner",
        submission_id="alice-20260702-03",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_MINER_AGENT, encoding="utf-8")

    with pytest.raises(ValueError, match="king artifact is not seeded"):
        evaluate_submission(
            str(submission_root),
            agent_command="/bin/true",
            sn60_project_keys=["project-a"],
        )


def test_evaluate_submission_selects_sn60_adapter_by_registry_evaluator_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_root = tmp_path / "kata-root"
    write_lane_metadata(
        EvaluatorLaneMetadata(
            schema_version=LANE_METADATA_SCHEMA_VERSION,
            lane_id="sn99__custom",
            repo_pack="sn99__custom",
            mode="miner",
            evaluator_id="sn60_bitsec",
            evaluator_policy_version="v1",
            active=True,
            created_at="2026-07-01T00:00:00+00:00",
            updated_at="2026-07-01T00:00:00+00:00",
        ),
        public_root=str(public_root),
    )
    monkeypatch.setenv("KATA_ROOT", str(public_root))
    monkeypatch.delenv("KATA_BENCHMARKS_ROOT", raising=False)
    king_root = seed_lane_king(public_root, "sn99__custom")

    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="sn99__custom",
        mode="miner",
        submission_id="alice-20260702-04",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_MINER_AGENT, encoding="utf-8")

    sentinel = object()
    calls: dict[str, object] = {}

    def fake_run_sn60_challenge(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr("kata.submissions.run_sn60_challenge", fake_run_sn60_challenge)

    summary = evaluate_submission(
        str(submission_root),
        agent_command="/bin/true",
        sn60_project_keys=["project-a"],
    )

    assert summary is sentinel
    assert calls["lane_id"] == "sn99__custom"
    assert calls["frontier_artifact_path"] == str(king_root.resolve())


def run_registry_lane_sn60_duel(tmp_path: Path, monkeypatch):
    public_root = tmp_path / "kata-root"
    write_evaluator_lane(public_root)
    monkeypatch.setenv("KATA_ROOT", str(public_root))
    monkeypatch.delenv("KATA_BENCHMARKS_ROOT", raising=False)
    king_root = seed_lane_king(public_root, "sn60__bitsec")

    sandbox_root = tmp_path / "sandbox"
    benchmark_path = sandbox_root / "validator" / "curated-highs-only-2025-08-08.json"
    benchmark_path.parent.mkdir(parents=True)
    benchmark_path.write_text(
        json.dumps([{"project_id": "project-alpha", "vulnerabilities": []}]) + "\n",
        encoding="utf-8",
    )

    repo_root = tmp_path / "Kata"
    submission_root = init_submission(
        repo_pack="sn60__bitsec",
        mode="miner",
        submission_id="alice-20260702-10",
        output_root=str(repo_root / "submissions"),
    )
    (submission_root / "agent.py").write_text(VALID_MINER_AGENT, encoding="utf-8")

    def execute(context):
        return {"success": True, "report": {"vulnerabilities": []}}

    def evaluate(context, report_payload):
        rate = 1.0 if context.variant_name == "candidate" else 0.0
        return {
            "status": "success",
            "result": {
                "detection_rate": rate,
                "true_positives": int(rate * 2),
                "total_expected": 2,
                "total_found": 1,
                "result": "PASS" if rate == 1.0 else "FAIL",
            },
        }

    summary = run_sn60_challenge(
        frontier_artifact_path=str(king_root),
        candidate_artifact_path=str(submission_root),
        project_keys=["project-alpha"],
        candidate_submission_id="alice-20260702-10",
        output_root=str(tmp_path / "runs"),
        replicas_per_project=2,
        sandbox_root=str(sandbox_root),
        benchmark_file=str(benchmark_path),
        sandbox_commit="commit-a",
        public_root=str(public_root),
        screening_hook=lambda ctx: {"success": True, "report": {"vulnerabilities": []}},
        execution_hook=execute,
        evaluation_hook=evaluate,
    )
    summary_path = Path(summary.manifest_path).with_name("challenge_summary.json")
    return public_root, submission_root, summary, summary_path


def test_verify_and_promote_sn60_registry_lane_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_root, submission_root, summary, summary_path = run_registry_lane_sn60_duel(
        tmp_path, monkeypatch
    )

    verification = verify_submission_result(str(submission_root), str(summary_path))
    assert verification.submission_matches_challenge
    assert verification.frontier_is_current
    assert verification.benchmark_is_current
    assert verification.promotion_ready
    assert verification.auto_merge_ready

    result = promote_submission_result(
        str(submission_root),
        str(summary_path),
        public_root=str(public_root),
    )
    assert result.lane_id == "sn60__bitsec"
    king_state = load_lane_king_state("sn60__bitsec", public_root=str(public_root))
    assert king_state.current_king_submission_id == "alice-20260702-10"
    assert king_state.current_king_artifact_hash == summary.candidate_artifact_hash
    promoted_agent = public_root / "kings" / "sn60__bitsec" / "miner" / "agent.py"
    assert promoted_agent.read_text(encoding="utf-8").strip() == VALID_MINER_AGENT.strip()

    # After promotion the candidate IS the king, so re-verifying the same
    # submission must fail validation as a copy of the current lane king.
    with pytest.raises(ValueError, match="exact copy of the current lane king"):
        verify_submission_result(str(submission_root), str(summary_path))


def test_verify_sn60_registry_lane_detects_stale_benchmark_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_root, submission_root, summary, summary_path = run_registry_lane_sn60_duel(
        tmp_path, monkeypatch
    )

    snapshot = load_benchmark_snapshot("sn60__bitsec", public_root=str(public_root))
    write_benchmark_snapshot(
        "sn60__bitsec",
        replace(snapshot, sandbox_commit_hash="commit-b"),
        public_root=str(public_root),
    )

    verification = verify_submission_result(str(submission_root), str(summary_path))
    assert verification.submission_matches_challenge
    assert verification.frontier_is_current
    assert not verification.benchmark_is_current
    assert not verification.auto_merge_ready
    assert any("SN60 benchmark lane has changed" in reason for reason in verification.reasons)


def test_verify_sn60_registry_lane_detects_superseded_challenge_fingerprint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_root, submission_root, summary, summary_path = run_registry_lane_sn60_duel(
        tmp_path, monkeypatch
    )

    state = load_challenge_state("sn60__bitsec", public_root=str(public_root))
    write_challenge_state(
        "sn60__bitsec",
        replace(state, freshness_fingerprint="0" * 64),
        public_root=str(public_root),
    )

    verification = verify_submission_result(str(submission_root), str(summary_path))
    assert not verification.benchmark_is_current
    assert not verification.auto_merge_ready
