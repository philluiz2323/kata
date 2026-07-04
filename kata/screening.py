from __future__ import annotations

import ast
import json
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Callable

from kata.agent_bundle import AGENT_ENTRY_FILENAME, load_bundle_files
from kata.ast_utils import (
    find_module_async_function_def,
    find_module_function_def,
    function_supports_no_arg_invocation,
)
from kata.evaluators.sn60_bitsec import (
    Sn60ReplicaContext,
    Sn60SandboxSource,
    build_default_execution_hook,
    hash_bundle_root,
    stage_bundle,
)
from kata.util import dedupe, write_json

SN60_SCREENING_SCHEMA_VERSION = 1
SN60_SCREENING_STATUS_PASSED = "passed"
SN60_SCREENING_STATUS_FAILED = "failed"
SN60_SCREENING_STAGE_STATIC = "static"
SN60_SCREENING_STAGE_EXECUTION = "execution"

BENCHMARK_LEAK_TOKENS = (
    "curated-highs-only",
    "known_solution",
    "known solution",
    "expected_findings",
    "expected findings",
    "expected_vulnerabilities",
    "expected vulnerabilities",
    "ground_truth",
    "ground truth",
    "answer_key",
    "answer key",
    "scabench",
    "hardsteer",
)

# Screener security checks: submitted agents must never reference validator
# scoring secrets or ship hardcoded provider keys.
VALIDATOR_SECRET_ENV_TOKENS = (
    "CHUTES_API_KEY",
    "KATA_VALIDATOR_API_KEY",
)
HARDCODED_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{10,}|cpk_[A-Za-z0-9]{10,})"
)


@dataclass(frozen=True)
class Sn60ScreeningResult:
    schema_version: int
    run_id: str
    status: str
    stage: str
    artifact_path: str
    artifact_hash: str
    project_key: str
    report_path: str | None
    result_path: str
    reasons: list[str]
    details: dict[str, object]
    sandbox_source: Sn60SandboxSource
    created_at: str

    @property
    def passed(self) -> bool:
        return self.status == SN60_SCREENING_STATUS_PASSED


Sn60ScreeningHook = Callable[[Sn60ReplicaContext], dict[str, object]]


def run_sn60_screening(
    *,
    candidate_artifact_path: str,
    project_key: str,
    output_root: str,
    sandbox_source: Sn60SandboxSource,
    execution_hook: Sn60ScreeningHook | None = None,
) -> Sn60ScreeningResult:
    artifact_root = Path(candidate_artifact_path).expanduser().resolve()
    output_base = Path(output_root).expanduser().resolve()
    run_id = build_sn60_screening_id()
    run_root = output_base / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    artifact_hash = hash_bundle_root(artifact_root)
    static_reasons = validate_sn60_static_screening(artifact_root)
    if static_reasons:
        result = build_screening_result(
            run_id=run_id,
            status=SN60_SCREENING_STATUS_FAILED,
            stage=SN60_SCREENING_STAGE_STATIC,
            artifact_root=artifact_root,
            artifact_hash=artifact_hash,
            project_key=project_key,
            report_path=None,
            result_path=run_root / "screening_result.json",
            reasons=static_reasons,
            details={"static_checks": "failed"},
            sandbox_source=sandbox_source,
        )
        write_screening_result(Path(result.result_path), result)
        return result

    bundle_root = run_root / "bundle"
    reports_root = run_root / "reports" / project_key
    reports_root.mkdir(parents=True, exist_ok=True)
    stage_bundle(artifact_root, bundle_root)
    context = Sn60ReplicaContext(
        run_id=run_id,
        variant_name="screening",
        project_key=project_key,
        replica_index=1,
        bundle_root=str(bundle_root),
        reports_root=str(reports_root),
        report_path=str(reports_root / "report.json"),
        evaluation_path=str(reports_root / "evaluation.json"),
        sandbox_source=sandbox_source,
    )
    execute = execution_hook or build_default_execution_hook(sandbox_source)
    try:
        report_payload = execute(context)
    except Exception as exc:
        report_payload = {
            "success": False,
            "error": f"SN60 screening execution failed before report creation: {exc}",
        }
    write_json(Path(context.report_path), report_payload)
    execution_reasons = validate_sn60_screening_report(report_payload)
    result = build_screening_result(
        run_id=run_id,
        status=(
            SN60_SCREENING_STATUS_FAILED
            if execution_reasons
            else SN60_SCREENING_STATUS_PASSED
        ),
        stage=SN60_SCREENING_STAGE_EXECUTION,
        artifact_root=artifact_root,
        artifact_hash=artifact_hash,
        project_key=project_key,
        report_path=Path(context.report_path),
        result_path=run_root / "screening_result.json",
        reasons=execution_reasons,
        details={"execution_report_success": bool(report_payload.get("success"))},
        sandbox_source=sandbox_source,
    )
    write_screening_result(Path(result.result_path), result)
    return result


def validate_sn60_static_screening(candidate_root: str | Path) -> list[str]:
    root = Path(candidate_root).expanduser().resolve()
    reasons: list[str] = []
    bundle_files = load_bundle_files(root)
    helper_paths = [
        relative_path
        for relative_path in sorted(bundle_files)
        if Path(relative_path).parts and Path(relative_path).parts[0] == "helpers"
    ]
    if helper_paths:
        reasons.append(
            "SN60 miner submissions do not support helper files in V1: "
            + ", ".join(helper_paths)
        )

    for relative_path, content in sorted(bundle_files.items()):
        if not relative_path.endswith(".py"):
            continue
        for token in VALIDATOR_SECRET_ENV_TOKENS:
            if token in content:
                reasons.append(
                    "SN60 screening rejected a validator secret reference: "
                    f"{relative_path} references `{token}`."
                )
        if HARDCODED_SECRET_PATTERN.search(content):
            reasons.append(
                f"SN60 screening rejected a hardcoded secret token in {relative_path}."
            )

    agent_source = bundle_files.get(AGENT_ENTRY_FILENAME)
    if agent_source is None:
        reasons.append("Submission agent must define agent_main(...).")
        return reasons

    try:
        tree = ast.parse(agent_source, filename=AGENT_ENTRY_FILENAME)
    except SyntaxError as exc:
        line_number = exc.lineno or 1
        reasons.append(
            f"Submission bundle contains invalid Python syntax in agent.py:{line_number}."
        )
        return reasons

    agent_main = find_module_function_def(tree, "agent_main")
    if agent_main is None:
        if find_module_async_function_def(tree, "agent_main") is not None:
            reasons.append(
                "Submission agent_main must be a synchronous function; the SN60 "
                "sandbox runner calls agent_main() directly and does not await "
                "coroutines."
            )
        else:
            reasons.append("Submission agent must define agent_main(...).")
    elif not function_supports_no_arg_invocation(agent_main):
        reasons.append("Submission agent must support no-argument invocation: agent_main().")

    lowered_source = agent_source.lower()
    for token in BENCHMARK_LEAK_TOKENS:
        if token in lowered_source:
            reasons.append(
                "SN60 screening rejected benchmark-answer leakage token: "
                f"`{token}`."
            )

    parsed_trees: dict[str, ast.AST] = {}
    for relative_path, content in sorted(bundle_files.items()):
        if not relative_path.endswith(".py"):
            continue
        try:
            parsed_trees[relative_path] = ast.parse(content, filename=relative_path)
        except SyntaxError:
            continue
    if parsed_trees:
        # submissions imports this module; lazy import avoids a circular import.
        from kata.submissions import validate_bundle_sampling_policy

        reasons.extend(validate_bundle_sampling_policy(parsed_trees))
    return dedupe(reasons)


def validate_sn60_screening_report(report_payload: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if not report_payload.get("success"):
        reasons.append(
            "SN60 screening execution did not complete successfully: "
            + str(report_payload.get("error", "unknown error"))
        )
    report = report_payload.get("report")
    if not isinstance(report, dict):
        reasons.append("SN60 screening report must be a JSON object.")
        return dedupe(reasons)
    vulnerabilities = report.get("vulnerabilities")
    if not isinstance(vulnerabilities, list):
        reasons.append("SN60 screening report must contain a top-level `vulnerabilities` list.")
    return dedupe(reasons)


def build_screening_result(
    *,
    run_id: str,
    status: str,
    stage: str,
    artifact_root: Path,
    artifact_hash: str,
    project_key: str,
    report_path: Path | None,
    result_path: Path,
    reasons: list[str],
    details: dict[str, object],
    sandbox_source: Sn60SandboxSource,
) -> Sn60ScreeningResult:
    return Sn60ScreeningResult(
        schema_version=SN60_SCREENING_SCHEMA_VERSION,
        run_id=run_id,
        status=status,
        stage=stage,
        artifact_path=str(artifact_root),
        artifact_hash=artifact_hash,
        project_key=project_key,
        report_path=str(report_path) if report_path is not None else None,
        result_path=str(result_path),
        reasons=reasons,
        details=details,
        sandbox_source=sandbox_source,
        created_at=datetime.now(UTC).isoformat(),
    )


def write_screening_result(path: Path, result: Sn60ScreeningResult) -> Path:
    write_json(path, asdict(result))
    return path




def screening_result_payload(result: Sn60ScreeningResult) -> dict[str, object]:
    return asdict(result)


def sn60_screening_freshness_fingerprint(
    *,
    king_artifact_hash: str,
    screening_result: Sn60ScreeningResult,
) -> str:
    payload = {
        "king_artifact_hash": king_artifact_hash,
        "candidate_artifact_hash": screening_result.artifact_hash,
        "project_key": screening_result.project_key,
        "screening_status": screening_result.status,
        "screening_stage": screening_result.stage,
        "sandbox_commit": screening_result.sandbox_source.sandbox_commit,
        "benchmark_sha256": screening_result.sandbox_source.benchmark_sha256,
        "scorer_version": screening_result.sandbox_source.scorer_version,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_sn60_screening_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"sn60-screening-{timestamp}-{secrets.token_hex(3)}"








