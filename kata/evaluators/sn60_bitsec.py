from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Callable

from kata.agent_bundle import AGENT_ENTRY_FILENAME, load_bundle_files, write_bundle_files
from kata.provenance import sha256_directory

DEFAULT_SN60_DUEL_SCHEMA_VERSION = 1
DEFAULT_SANDBOX_PROXY_NETWORK = "bitsec-net"
DEFAULT_SANDBOX_PROXY_URL = "http://localhost:8087"
DEFAULT_SANDBOX_INFERENCE_API = "http://bitsec_proxy:8000"
DEFAULT_EVAL_MAX_VULNS = 100
DEFAULT_REPLICAS_PER_PROJECT = 3
DEFAULT_BENCHMARK_FILENAME = "curated-highs-only-2025-08-08.json"


@dataclass(frozen=True)
class Sn60SandboxSource:
    sandbox_root: str
    benchmark_file: str
    benchmark_sha256: str
    sandbox_commit: str
    scorer_version: str


@dataclass(frozen=True)
class Sn60ReplicaContext:
    run_id: str
    variant_name: str
    project_key: str
    replica_index: int
    bundle_root: str
    reports_root: str
    report_path: str
    evaluation_path: str
    sandbox_source: Sn60SandboxSource


@dataclass(frozen=True)
class Sn60ReplicaResult:
    project_key: str
    replica_index: int
    report_path: str
    evaluation_path: str
    execution_success: bool
    evaluation_status: str
    score: float
    detection_rate: float
    result: str | None
    true_positives: int
    total_expected: int
    total_found: int


@dataclass(frozen=True)
class Sn60ProjectAggregate:
    project_key: str
    replica_count: int
    successful_runs: int
    invalid_runs: int
    pass_count: int
    average_score: float
    average_detection_rate: float
    true_positives: int
    total_expected: int
    total_found: int


@dataclass(frozen=True)
class Sn60VariantSummary:
    variant_name: str
    artifact_path: str
    artifact_hash: str
    successful_runs: int
    invalid_runs: int
    pass_count: int
    average_score: float
    average_detection_rate: float
    true_positives: int
    total_expected: int
    total_found: int
    project_summaries: list[Sn60ProjectAggregate]
    replica_results: list[Sn60ReplicaResult]


@dataclass(frozen=True)
class Sn60DuelSummary:
    schema_version: int
    run_id: str
    created_at: str
    output_root: str
    project_keys: list[str]
    replicas_per_project: int
    sandbox_source: Sn60SandboxSource
    frontier: Sn60VariantSummary
    candidate: Sn60VariantSummary


Sn60ExecutionHook = Callable[[Sn60ReplicaContext], dict[str, object]]
Sn60EvaluationHook = Callable[[Sn60ReplicaContext, dict[str, object]], dict[str, object]]


def run_sn60_bitsec_duel(
    *,
    frontier_artifact_path: str,
    candidate_artifact_path: str,
    project_keys: list[str],
    output_root: str | None = None,
    replicas_per_project: int = DEFAULT_REPLICAS_PER_PROJECT,
    sandbox_root: str | None = None,
    benchmark_file: str | None = None,
    sandbox_commit: str | None = None,
    scorer_version: str = "ScaBenchScorerV2",
    execution_hook: Sn60ExecutionHook | None = None,
    evaluation_hook: Sn60EvaluationHook | None = None,
) -> Sn60DuelSummary:
    if not project_keys:
        raise ValueError("SN60 duel requires at least one project key.")
    if replicas_per_project <= 0:
        raise ValueError("SN60 duel replicas_per_project must be positive.")

    source = resolve_sn60_sandbox_source(
        sandbox_root=sandbox_root,
        benchmark_file=benchmark_file,
        sandbox_commit=sandbox_commit,
        scorer_version=scorer_version,
    )
    frontier_root = Path(frontier_artifact_path).expanduser().resolve()
    candidate_root = Path(candidate_artifact_path).expanduser().resolve()
    output_base = (
        Path(output_root).expanduser().resolve()
        if output_root
        else Path("runs").resolve()
    )
    run_id = build_sn60_duel_id()
    run_root = output_base / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    frontier_summary = evaluate_variant(
        run_id=run_id,
        run_root=run_root,
        variant_name="frontier",
        artifact_root=frontier_root,
        project_keys=project_keys,
        replicas_per_project=replicas_per_project,
        sandbox_source=source,
        execution_hook=execution_hook or build_default_execution_hook(source),
        evaluation_hook=evaluation_hook or build_default_evaluation_hook(source),
    )
    candidate_summary = evaluate_variant(
        run_id=run_id,
        run_root=run_root,
        variant_name="candidate",
        artifact_root=candidate_root,
        project_keys=project_keys,
        replicas_per_project=replicas_per_project,
        sandbox_source=source,
        execution_hook=execution_hook or build_default_execution_hook(source),
        evaluation_hook=evaluation_hook or build_default_evaluation_hook(source),
    )

    summary = Sn60DuelSummary(
        schema_version=DEFAULT_SN60_DUEL_SCHEMA_VERSION,
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        output_root=str(run_root),
        project_keys=list(project_keys),
        replicas_per_project=replicas_per_project,
        sandbox_source=source,
        frontier=frontier_summary,
        candidate=candidate_summary,
    )
    write_sn60_duel_summary(run_root / "duel_summary.json", summary)
    return summary


def evaluate_variant(
    *,
    run_id: str,
    run_root: Path,
    variant_name: str,
    artifact_root: Path,
    project_keys: list[str],
    replicas_per_project: int,
    sandbox_source: Sn60SandboxSource,
    execution_hook: Sn60ExecutionHook,
    evaluation_hook: Sn60EvaluationHook,
) -> Sn60VariantSummary:
    variant_root = run_root / variant_name
    artifact_hash = hash_bundle_root(artifact_root)
    replica_results: list[Sn60ReplicaResult] = []

    for project_key in project_keys:
        for replica_index in range(1, replicas_per_project + 1):
            replica_root = variant_root / project_key / f"replica-{replica_index:02d}"
            bundle_root = replica_root / "bundle"
            project_reports_root = replica_root / "reports" / project_key
            project_reports_root.mkdir(parents=True, exist_ok=True)
            stage_bundle(artifact_root, bundle_root)

            context = Sn60ReplicaContext(
                run_id=run_id,
                variant_name=variant_name,
                project_key=project_key,
                replica_index=replica_index,
                bundle_root=str(bundle_root),
                reports_root=str(project_reports_root),
                report_path=str(project_reports_root / "report.json"),
                evaluation_path=str(project_reports_root / "evaluation.json"),
                sandbox_source=sandbox_source,
            )
            report_payload = execution_hook(context)
            write_json_file(Path(context.report_path), report_payload)
            evaluation_payload = evaluation_hook(context, report_payload)
            write_json_file(Path(context.evaluation_path), evaluation_payload)
            replica_results.append(
                build_replica_result(context, report_payload, evaluation_payload)
            )

    return summarize_variant(
        variant_name=variant_name,
        artifact_root=artifact_root,
        artifact_hash=artifact_hash,
        replica_results=replica_results,
    )


def resolve_sn60_sandbox_source(
    *,
    sandbox_root: str | None = None,
    benchmark_file: str | None = None,
    sandbox_commit: str | None = None,
    scorer_version: str,
) -> Sn60SandboxSource:
    resolved_sandbox_root = (
        Path(sandbox_root).expanduser().resolve()
        if sandbox_root
        else default_sandbox_root()
    )
    resolved_benchmark_file = (
        Path(benchmark_file).expanduser().resolve()
        if benchmark_file
        else resolved_sandbox_root / "validator" / DEFAULT_BENCHMARK_FILENAME
    )
    if not resolved_benchmark_file.exists():
        raise FileNotFoundError(
            f"SN60 benchmark snapshot does not exist: {resolved_benchmark_file}"
        )
    resolved_commit = sandbox_commit or resolve_git_commit(resolved_sandbox_root)
    return Sn60SandboxSource(
        sandbox_root=str(resolved_sandbox_root),
        benchmark_file=str(resolved_benchmark_file),
        benchmark_sha256=sha256_directory(
            resolved_benchmark_file.parent,
            include=[resolved_benchmark_file.name],
        ),
        sandbox_commit=resolved_commit,
        scorer_version=scorer_version,
    )


def default_sandbox_root() -> Path:
    return workspace_root() / "sandbox"


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_sn60_duel_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"sn60-duel-{timestamp}-{secrets.token_hex(3)}"


def stage_bundle(source_root: Path, destination_root: Path) -> None:
    bundle_files = load_bundle_files(source_root)
    if not bundle_files:
        raise ValueError(f"SN60 artifact bundle is empty: {source_root}")
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    write_bundle_files(destination_root, bundle_files)


def hash_bundle_root(bundle_root: Path) -> str:
    bundle_files = load_bundle_files(bundle_root)
    if not bundle_files:
        raise ValueError(f"SN60 artifact bundle is empty: {bundle_root}")
    return sha256_directory(bundle_root, include=sorted(bundle_files))


def write_sn60_duel_summary(path: Path, summary: Sn60DuelSummary) -> None:
    write_json_file(path, asdict(summary))


def write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def summarize_variant(
    *,
    variant_name: str,
    artifact_root: Path,
    artifact_hash: str,
    replica_results: list[Sn60ReplicaResult],
) -> Sn60VariantSummary:
    project_keys = sorted({result.project_key for result in replica_results})
    project_summaries = [
        summarize_project(
            project_key=project_key,
            replica_results=[
                result for result in replica_results if result.project_key == project_key
            ],
        )
        for project_key in project_keys
    ]
    scores = [result.score for result in replica_results]
    detection_rates = [result.detection_rate for result in replica_results]
    return Sn60VariantSummary(
        variant_name=variant_name,
        artifact_path=str(artifact_root),
        artifact_hash=artifact_hash,
        successful_runs=sum(
            1 for result in replica_results if result.evaluation_status == "success"
        ),
        invalid_runs=sum(1 for result in replica_results if result.evaluation_status != "success"),
        pass_count=sum(1 for result in replica_results if result.result == "PASS"),
        average_score=fmean(scores) if scores else 0.0,
        average_detection_rate=fmean(detection_rates) if detection_rates else 0.0,
        true_positives=sum(result.true_positives for result in replica_results),
        total_expected=sum(result.total_expected for result in replica_results),
        total_found=sum(result.total_found for result in replica_results),
        project_summaries=project_summaries,
        replica_results=replica_results,
    )


def summarize_project(
    *,
    project_key: str,
    replica_results: list[Sn60ReplicaResult],
) -> Sn60ProjectAggregate:
    scores = [result.score for result in replica_results]
    detection_rates = [result.detection_rate for result in replica_results]
    return Sn60ProjectAggregate(
        project_key=project_key,
        replica_count=len(replica_results),
        successful_runs=sum(
            1 for result in replica_results if result.evaluation_status == "success"
        ),
        invalid_runs=sum(1 for result in replica_results if result.evaluation_status != "success"),
        pass_count=sum(1 for result in replica_results if result.result == "PASS"),
        average_score=fmean(scores) if scores else 0.0,
        average_detection_rate=fmean(detection_rates) if detection_rates else 0.0,
        true_positives=sum(result.true_positives for result in replica_results),
        total_expected=sum(result.total_expected for result in replica_results),
        total_found=sum(result.total_found for result in replica_results),
    )


def build_replica_result(
    context: Sn60ReplicaContext,
    report_payload: dict[str, object],
    evaluation_payload: dict[str, object],
) -> Sn60ReplicaResult:
    metrics = extract_evaluation_metrics(evaluation_payload)
    return Sn60ReplicaResult(
        project_key=context.project_key,
        replica_index=context.replica_index,
        report_path=context.report_path,
        evaluation_path=context.evaluation_path,
        execution_success=bool(report_payload.get("success")),
        evaluation_status=metrics["evaluation_status"],
        score=metrics["score"],
        detection_rate=metrics["detection_rate"],
        result=metrics["result"],
        true_positives=metrics["true_positives"],
        total_expected=metrics["total_expected"],
        total_found=metrics["total_found"],
    )


def extract_evaluation_metrics(evaluation_payload: dict[str, object]) -> dict[str, object]:
    status_value = str(evaluation_payload.get("status", "error")).lower()
    result_payload = evaluation_payload.get("result")
    if not isinstance(result_payload, dict):
        result_payload = {}
    detection_rate = float(result_payload.get("detection_rate", 0.0) or 0.0)
    return {
        "evaluation_status": status_value,
        "score": detection_rate if status_value == "success" else 0.0,
        "detection_rate": detection_rate if status_value == "success" else 0.0,
        "result": (
            str(result_payload["result"])
            if result_payload.get("result") is not None
            else None
        ),
        "true_positives": int(result_payload.get("true_positives", 0) or 0),
        "total_expected": int(result_payload.get("total_expected", 0) or 0),
        "total_found": int(result_payload.get("total_found", 0) or 0),
    }


def build_default_execution_hook(source: Sn60SandboxSource) -> Sn60ExecutionHook:
    def _execute(context: Sn60ReplicaContext) -> dict[str, object]:
        command = build_bitsec_execution_command(context)
        env = {
            "INFERENCE_API_KEY": required_env("INFERENCE_API_KEY"),
        }
        completed = subprocess.run(
            command,
            cwd=source.sandbox_root,
            capture_output=True,
            text=True,
            env={**default_subprocess_env(), **env},
        )
        report_path = Path(context.report_path)
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "success": False,
            "error": (
                f"Bitsec execution command failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            ),
        }

    return _execute


def build_default_evaluation_hook(source: Sn60SandboxSource) -> Sn60EvaluationHook:
    def _evaluate(
        context: Sn60ReplicaContext,
        report_payload: dict[str, object],
    ) -> dict[str, object]:
        if not Path(context.report_path).exists():
            write_json_file(Path(context.report_path), report_payload)
        completed = subprocess.run(
            build_bitsec_evaluation_command(context),
            cwd=source.sandbox_root,
            capture_output=True,
            text=True,
            env={
                **default_subprocess_env(),
                "KATA_SN60_JOB_RUN_REPORTS_DIR": str(
                    Path(context.reports_root).parent.resolve()
                ),
                "KATA_SN60_PROJECT_KEY": context.project_key,
                "KATA_SN60_EVAL_MAX_VULNS": str(DEFAULT_EVAL_MAX_VULNS),
                "CHUTES_API_KEY": required_env("CHUTES_API_KEY"),
                "PROXY_URL": DEFAULT_SANDBOX_PROXY_URL,
            },
        )
        if completed.returncode == 0:
            try:
                return json.loads(completed.stdout.strip())
            except json.JSONDecodeError:
                pass
        evaluation_path = Path(context.evaluation_path)
        if evaluation_path.exists():
            return json.loads(evaluation_path.read_text(encoding="utf-8"))
        return {
            "status": "error",
            "error": (
                f"Bitsec evaluation command failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            ),
            "result": {},
        }

    return _evaluate


def build_bitsec_execution_command(
    context: Sn60ReplicaContext,
    *,
    proxy_network: str = DEFAULT_SANDBOX_PROXY_NETWORK,
    inference_api: str = DEFAULT_SANDBOX_INFERENCE_API,
) -> list[str]:
    bundle_root = Path(context.bundle_root).resolve()
    reports_root = Path(context.reports_root).resolve()
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        proxy_network,
        "--volume",
        f"{bundle_root}:/kata_bundle:ro",
        "--volume",
        f"{reports_root}:/kata_output",
        "--env",
        f"AGENT_FILE=/kata_bundle/{AGENT_ENTRY_FILENAME}",
        "--env",
        "PYTHONPATH=/kata_bundle",
        "--env",
        "REPORT_FILE=/kata_output/report.json",
        "--env",
        f"JOB_RUN_ID={context.run_id}",
        "--env",
        f"PROJECT_KEY={context.project_key}",
        "--env",
        f"INFERENCE_API={inference_api}",
        "--env",
        "INFERENCE_API_KEY",
        f"ghcr.io/bitsec-ai/{context.project_key}:latest",
    ]


def build_bitsec_evaluation_command(context: Sn60ReplicaContext) -> list[str]:
    script = (
        "import json; "
        "from validator.executor import AgentExecutor; "
        "from validator.models.platform import MockJobRun; "
        "from validator.platform_client import MockPlatformClient; "
        "executor = AgentExecutor("
        "job_run=MockJobRun(id=1, job_id=1, validator_id=1, agent_id=1), "
        "agent_filepath='', "
        "project_key='"
        + context.project_key
        + "', "
        "job_run_reports_dir='"
        + str(Path(context.reports_root).parent.resolve())
        + "', "
        "platform_client=MockPlatformClient(), "
        "eval_max_vulns="
        + str(DEFAULT_EVAL_MAX_VULNS)
        + "); "
        "print(json.dumps(executor.eval_job_run(), default=str))"
    )
    return ["uv", "run", "python", "-c", script]


def default_subprocess_env() -> dict[str, str]:
    return dict(os.environ)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Required environment variable is not set: {name}")
    return value
