from __future__ import annotations

import json
from pathlib import Path

from kata.evaluators.sn60_bitsec import Sn60ReplicaContext, resolve_sn60_sandbox_source
from kata.screening import (
    SN60_SCREENING_STAGE_EXECUTION,
    SN60_SCREENING_STAGE_STATIC,
    run_sn60_screening,
    validate_sn60_static_screening,
)


def write_bundle(root: Path, agent_source: str, *, helper_source: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.py").write_text(agent_source, encoding="utf-8")
    if helper_source is not None:
        helpers_root = root / "helpers"
        helpers_root.mkdir()
        (helpers_root / "planner.py").write_text(helper_source, encoding="utf-8")


def write_sandbox_source(root: Path) -> Path:
    benchmark_path = root / "validator" / "curated-highs-only-2025-08-08.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(
        json.dumps([{"project_id": "project-alpha", "vulnerabilities": []}]) + "\n",
        encoding="utf-8",
    )
    return benchmark_path


def test_validate_sn60_static_screening_rejects_helper_files_and_leak_tokens(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "candidate"
    write_bundle(
        bundle_root,
        "KNOWN = 'curated-highs-only'\n"
        "def agent_main(project_dir=None, inference_api=None):\n"
        "    return {'vulnerabilities': []}\n",
        helper_source="VALUE = 1\n",
    )

    reasons = validate_sn60_static_screening(bundle_root)

    assert any("do not support helper files in V1" in reason for reason in reasons)
    assert any("benchmark-answer leakage token" in reason for reason in reasons)


def test_validate_sn60_static_screening_rejects_async_agent_main(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "candidate"
    write_bundle(
        bundle_root,
        "async def agent_main(project_dir=None, inference_api=None):\n"
        "    return {'vulnerabilities': []}\n",
    )

    reasons = validate_sn60_static_screening(bundle_root)

    assert any("must be a synchronous function" in reason for reason in reasons)


def test_validate_sn60_static_screening_rejects_sampling_override(tmp_path: Path) -> None:
    bundle_root = tmp_path / "candidate"
    write_bundle(
        bundle_root,
        "def agent_main(project_dir=None, inference_api=None):\n"
        "    call(model='x', temperature=0.0)\n"
        "    return {'vulnerabilities': []}\n",
    )

    reasons = validate_sn60_static_screening(bundle_root)

    assert any("sampling parameters" in reason for reason in reasons)


def test_run_sn60_screening_persists_static_failure_without_execution(
    tmp_path: Path,
) -> None:
    sandbox_root = tmp_path / "sandbox"
    benchmark_path = write_sandbox_source(sandbox_root)
    source = resolve_sn60_sandbox_source(
        sandbox_root=str(sandbox_root),
        benchmark_file=str(benchmark_path),
        sandbox_commit="commit-1",
        scorer_version="ScaBenchScorerV2",
    )
    bundle_root = tmp_path / "candidate"
    write_bundle(
        bundle_root,
        "def agent_main(project_dir):\n"
        "    return {'vulnerabilities': []}\n",
    )
    execution_called = False

    def execute(context: Sn60ReplicaContext) -> dict[str, object]:
        nonlocal execution_called
        execution_called = True
        return {"success": True, "report": {"vulnerabilities": []}}

    result = run_sn60_screening(
        candidate_artifact_path=str(bundle_root),
        project_key="project-alpha",
        output_root=str(tmp_path / "runs"),
        sandbox_source=source,
        execution_hook=execute,
    )

    assert not execution_called
    assert not result.passed
    assert result.stage == SN60_SCREENING_STAGE_STATIC
    assert Path(result.result_path).exists()
    assert result.report_path is None


def test_run_sn60_screening_rejects_bad_execution_report(tmp_path: Path) -> None:
    sandbox_root = tmp_path / "sandbox"
    benchmark_path = write_sandbox_source(sandbox_root)
    source = resolve_sn60_sandbox_source(
        sandbox_root=str(sandbox_root),
        benchmark_file=str(benchmark_path),
        sandbox_commit="commit-1",
        scorer_version="ScaBenchScorerV2",
    )
    bundle_root = tmp_path / "candidate"
    write_bundle(
        bundle_root,
        "def agent_main(project_dir=None, inference_api=None):\n"
        "    return {'vulnerabilities': []}\n",
    )

    def execute(context: Sn60ReplicaContext) -> dict[str, object]:
        return {"success": True, "report": {"findings": []}}

    result = run_sn60_screening(
        candidate_artifact_path=str(bundle_root),
        project_key="project-alpha",
        output_root=str(tmp_path / "runs"),
        sandbox_source=source,
        execution_hook=execute,
    )

    assert not result.passed
    assert result.stage == SN60_SCREENING_STAGE_EXECUTION
    assert any("top-level `vulnerabilities` list" in reason for reason in result.reasons)
    assert Path(result.report_path or "").exists()


def test_validate_sn60_static_screening_rejects_expanded_leak_tokens(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "candidate"
    write_bundle(
        bundle_root,
        "GROUND = 'ground_truth'\n"
        "def agent_main(project_dir=None, inference_api=None):\n"
        "    return {'vulnerabilities': []}\n",
    )

    reasons = validate_sn60_static_screening(bundle_root)

    assert any(
        "benchmark-answer leakage token" in reason and "ground_truth" in reason
        for reason in reasons
    )


def test_validate_sn60_static_screening_rejects_validator_secret_reference(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "candidate"
    write_bundle(
        bundle_root,
        "import os\n"
        "def agent_main(project_dir=None, inference_api=None):\n"
        "    os.environ.get('CHUTES_API_KEY')\n"
        "    return {'vulnerabilities': []}\n",
    )

    reasons = validate_sn60_static_screening(bundle_root)

    assert any("validator secret reference" in reason for reason in reasons)


def test_validate_sn60_static_screening_rejects_hardcoded_chutes_key(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "candidate"
    write_bundle(
        bundle_root,
        "KEY = 'cpk_abcdefghij1234567890'\n"
        "def agent_main(project_dir=None, inference_api=None):\n"
        "    return {'vulnerabilities': []}\n",
    )

    reasons = validate_sn60_static_screening(bundle_root)

    assert any("hardcoded secret token" in reason for reason in reasons)
