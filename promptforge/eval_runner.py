from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from promptforge.baseline import generate_baseline_prompt_from_repository
from promptforge.benchmarks import resolve_eval_pack_path
from promptforge.config import resolve_registry_url
from promptforge.eval_pack import EvalPackValidationResult, discover_eval_pack_tasks
from promptforge.generator import generate_prompt_from_repository
from promptforge.provenance import EVALUATOR_VERSION, pool_fingerprint
from promptforge.repository import resolve_repository
from promptforge.scoring import read_task_weight, score_variant_run

IGNORED_COPY_DIRS = (
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
)


@dataclass(frozen=True)
class VariantResult:
    name: str
    prompt_path: str
    workspace: str
    agent_stdout: str
    agent_stderr: str
    checks_stdout: str
    checks_stderr: str
    agent_exit_code: int
    checks_exit_code: int
    task_solved: bool
    validity_passed: bool
    verifier_score: float
    quality_score: float
    weighted_task_score: float
    score_source: str
    changed_paths: list[str]
    path_policy_passed: bool
    path_policy_violations: list[str]
    success: bool


@dataclass(frozen=True)
class TaskRunSummary:
    task_id: str
    task_path: str
    task_repo_ref: str
    task_weight: float
    variants: list[VariantResult]


@dataclass(frozen=True)
class EvalRunSummary:
    run_id: str
    requested_repo_ref: str
    eval_pack: str
    mode: str
    registry_url: str | None
    agent_command: str
    created_at: str
    tasks: list[TaskRunSummary]
    run_kind: str = "eval"
    metadata: dict[str, str] = field(default_factory=dict)


def run_eval(
    *,
    repo_ref: str,
    eval_pack_path: str,
    mode: str,
    agent_command: str,
    registry_url: str | None = None,
    output_root: str | None = None,
    agent_timeout_seconds: int | None = None,
    checks_timeout_seconds: int | None = None,
) -> EvalRunSummary:
    eval_pack_root = resolve_eval_pack_path(eval_pack_path)
    validations = discover_eval_pack_tasks(eval_pack_path)
    invalid = [result for result in validations if not result.is_valid]
    if invalid:
        invalid_names = ", ".join(result.root.name for result in invalid)
        raise ValueError(
            "Eval pack is invalid. Run `promptforge eval-pack validate` first. "
            f"Invalid task directories: {invalid_names}"
        )

    resolved_registry_url = resolve_registry_url(registry_url)
    return run_prompt_variants(
        repo_ref=repo_ref,
        eval_pack_path=str(eval_pack_root),
        mode=mode,
        agent_command=agent_command,
        prompt_variants=[
            ("baseline", None),
            ("generated", resolved_registry_url),
        ],
        output_root=output_root,
        agent_timeout_seconds=agent_timeout_seconds,
        checks_timeout_seconds=checks_timeout_seconds,
        run_label=None,
        run_kind="eval",
        metadata={"reference_workflow": "baseline-vs-generated"},
    )


def run_prompt_variants(
    *,
    repo_ref: str,
    eval_pack_path: str,
    mode: str,
    agent_command: str,
    prompt_variants: list[tuple[str, str | None]],
    task_names: list[str] | None = None,
    output_root: str | None = None,
    agent_timeout_seconds: int | None = None,
    checks_timeout_seconds: int | None = None,
    run_label: str | None = None,
    run_kind: str = "eval",
    metadata: dict[str, str] | None = None,
) -> EvalRunSummary:
    eval_pack_root = resolve_eval_pack_path(eval_pack_path)
    validations = discover_eval_pack_tasks(eval_pack_path)
    invalid = [result for result in validations if not result.is_valid]
    if invalid:
        invalid_names = ", ".join(result.root.name for result in invalid)
        raise ValueError(
            "Eval pack is invalid. Run `promptforge eval-pack validate` first. "
            f"Invalid task directories: {invalid_names}"
        )
    selected_validations = select_task_validations(validations, task_names)
    selected_roots = [result.root for result in selected_validations]
    task_ids = [result.root.name for result in selected_validations]
    if run_label is not None:
        run_name = run_label
    elif len(task_ids) > 1:
        run_name = selected_validations[0].root.parent.name
    else:
        run_name = task_ids[0]
    run_id = build_run_id(run_name)
    runs_root = Path(output_root) if output_root else Path("runs")
    run_root = runs_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    task_summaries: list[TaskRunSummary] = []
    for validation in selected_validations:
        task_root = validation.root
        task_run_root = run_root / "tasks" / task_root.name
        task_run_root.mkdir(parents=True, exist_ok=False)
        task_snapshot = task_run_root / "eval_pack"
        shutil.copytree(task_root, task_snapshot)
        task_weight = read_task_weight(task_snapshot)

        task_repo_ref = read_task_repo_ref(task_snapshot / "repo_ref.txt", fallback=repo_ref)
        with resolve_repository(task_repo_ref) as repo:
            repo_snapshot = task_run_root / "repo_snapshot"
            copy_repository(repo.root, repo_snapshot)
            variants = [
                run_variant(
                    variant_name=variant_name,
                    prompt_text=resolve_prompt_text(
                        variant_name=variant_name,
                        prompt_value=prompt_value,
                        repo_ref=task_repo_ref,
                        repo=repo,
                        mode=mode,
                    ),
                    variant_root=task_run_root / variant_name,
                    repo_snapshot=repo_snapshot,
                    eval_pack_root=task_snapshot,
                    repo_ref=task_repo_ref,
                    mode=mode,
                    agent_command=agent_command,
                    agent_timeout_seconds=agent_timeout_seconds,
                    checks_timeout_seconds=checks_timeout_seconds,
                )
                for variant_name, prompt_value in prompt_variants
            ]

        task_summaries.append(
            TaskRunSummary(
                task_id=task_root.name,
                task_path=str(task_run_root),
                task_repo_ref=task_repo_ref,
                task_weight=task_weight,
                variants=variants,
            )
        )

    summary = EvalRunSummary(
        run_id=run_id,
        requested_repo_ref=repo_ref,
        eval_pack=str(eval_pack_root),
        mode=mode,
        registry_url=None,
        agent_command=agent_command,
        created_at=datetime.now(UTC).isoformat(),
        tasks=task_summaries,
        run_kind=run_kind,
        metadata=build_run_metadata(
            task_ids=task_ids,
            task_roots=selected_roots,
            extra=metadata,
        ),
    )
    write_summary(run_root / "run_summary.json", summary)
    return summary


def run_variant(
    *,
    variant_name: str,
    prompt_text: str,
    variant_root: Path,
    repo_snapshot: Path,
    eval_pack_root: Path,
    repo_ref: str,
    mode: str,
    agent_command: str,
    agent_timeout_seconds: int | None,
    checks_timeout_seconds: int | None,
) -> VariantResult:
    workspace = variant_root / "workspace"
    shutil.copytree(repo_snapshot, workspace)

    prompt_path = variant_root / "prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text + "\n", encoding="utf-8")

    agent_stdout = variant_root / "agent.stdout.txt"
    agent_stderr = variant_root / "agent.stderr.txt"
    checks_stdout = variant_root / "checks.stdout.txt"
    checks_stderr = variant_root / "checks.stderr.txt"
    score_file = variant_root / "score.txt"

    agent_exit_code = run_agent_command(
        command=agent_command,
        workspace=workspace,
        prompt_path=prompt_path,
        eval_pack_root=eval_pack_root,
        repo_snapshot=repo_snapshot,
        mode=mode,
        repo_ref=repo_ref,
        stdout_path=agent_stdout,
        stderr_path=agent_stderr,
        score_file=score_file,
        timeout_seconds=agent_timeout_seconds,
    )
    checks_exit_code = run_checks(
        checks_path=eval_pack_root / "checks.sh",
        workspace=workspace,
        prompt_path=prompt_path,
        eval_pack_root=eval_pack_root,
        repo_snapshot=repo_snapshot,
        mode=mode,
        repo_ref=repo_ref,
        stdout_path=checks_stdout,
        stderr_path=checks_stderr,
        score_file=score_file,
        timeout_seconds=checks_timeout_seconds,
    )
    benchmark_score = score_variant_run(
        repo_snapshot=repo_snapshot,
        eval_pack_root=eval_pack_root,
        workspace=workspace,
        agent_exit_code=agent_exit_code,
        checks_exit_code=checks_exit_code,
        score_file=score_file,
    )
    return VariantResult(
        name=variant_name,
        prompt_path=str(prompt_path),
        workspace=str(workspace),
        agent_stdout=str(agent_stdout),
        agent_stderr=str(agent_stderr),
        checks_stdout=str(checks_stdout),
        checks_stderr=str(checks_stderr),
        agent_exit_code=agent_exit_code,
        checks_exit_code=checks_exit_code,
        task_solved=benchmark_score.task_solved,
        validity_passed=benchmark_score.validity_passed,
        verifier_score=benchmark_score.verifier_score,
        quality_score=benchmark_score.quality_score,
        weighted_task_score=benchmark_score.weighted_task_score,
        score_source=benchmark_score.score_source,
        changed_paths=benchmark_score.changed_paths,
        path_policy_passed=benchmark_score.path_policy_passed,
        path_policy_violations=benchmark_score.path_policy_violations,
        success=(
            benchmark_score.agent_ok
            and benchmark_score.checks_passed
            and benchmark_score.path_policy_passed
        ),
    )


def run_agent_command(
    *,
    command: str,
    workspace: Path,
    prompt_path: Path,
    eval_pack_root: Path,
    repo_snapshot: Path,
    mode: str,
    repo_ref: str,
    stdout_path: Path,
    stderr_path: Path,
    score_file: Path,
    timeout_seconds: int | None,
) -> int:
    env = build_env(
        workspace=workspace,
        prompt_path=prompt_path,
        eval_pack_root=eval_pack_root,
        repo_snapshot=repo_snapshot,
        mode=mode,
        repo_ref=repo_ref,
        score_file=score_file,
    )
    return run_shell_command(
        command,
        workspace,
        env,
        stdout_path,
        stderr_path,
        timeout_seconds=timeout_seconds,
    )


def run_checks(
    *,
    checks_path: Path,
    workspace: Path,
    prompt_path: Path,
    eval_pack_root: Path,
    repo_snapshot: Path,
    mode: str,
    repo_ref: str,
    stdout_path: Path,
    stderr_path: Path,
    score_file: Path,
    timeout_seconds: int | None,
) -> int:
    env = build_env(
        workspace=workspace,
        prompt_path=prompt_path,
        eval_pack_root=eval_pack_root,
        repo_snapshot=repo_snapshot,
        mode=mode,
        repo_ref=repo_ref,
        score_file=score_file,
    )
    return run_process(
        ["bash", str(checks_path.resolve())],
        workspace,
        env,
        stdout_path,
        stderr_path,
        timeout_seconds=timeout_seconds,
    )


def run_shell_command(
    command: str,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int | None,
) -> int:
    return run_process(
        ["bash", "-lc", command],
        cwd,
        env,
        stdout_path,
        stderr_path,
        timeout_seconds=timeout_seconds,
    )


def run_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int | None,
) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            stderr_file.write(
                f"PromptForge timeout after {timeout_seconds} seconds for command: "
                f"{' '.join(command)}\n"
            )
            return 124
    return completed.returncode


def build_run_metadata(
    *,
    task_ids: list[str],
    task_roots: list[Path],
    extra: dict[str, str] | None,
) -> dict[str, str]:
    metadata = {
        "evaluator_version": EVALUATOR_VERSION,
        "task_count": str(len(task_ids)),
        "task_ids": ",".join(task_ids),
        "task_pool_fingerprint": pool_fingerprint(task_roots),
    }
    if extra:
        metadata.update(extra)
    return metadata


def build_env(
    *,
    workspace: Path,
    prompt_path: Path,
    eval_pack_root: Path,
    repo_snapshot: Path,
    mode: str,
    repo_ref: str,
    score_file: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    env["PROMPTFORGE_WORKSPACE"] = str(workspace.resolve())
    env["PROMPTFORGE_PROMPT_FILE"] = str(prompt_path.resolve())
    env["PROMPTFORGE_MODE"] = mode
    env["PROMPTFORGE_REPO_REF"] = repo_ref
    env["PROMPTFORGE_REPO_SNAPSHOT"] = str(repo_snapshot.resolve())
    env["PROMPTFORGE_EVAL_TASK_DIR"] = str(eval_pack_root.resolve())
    env["PROMPTFORGE_SCORE_FILE"] = str(score_file.resolve())
    env["PROMPTFORGE_TASK_FILE"] = str((eval_pack_root / "task.md").resolve())
    env["PROMPTFORGE_RUBRIC_FILE"] = str((eval_pack_root / "rubric.md").resolve())
    env["PROMPTFORGE_ALLOWED_PATHS_FILE"] = str((eval_pack_root / "allowed_paths.txt").resolve())
    env["PROMPTFORGE_FORBIDDEN_PATHS_FILE"] = str(
        (eval_pack_root / "forbidden_paths.txt").resolve()
    )
    return env


def read_task_repo_ref(path: Path, *, fallback: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped
    return fallback


def build_run_id(task_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{task_id}-{timestamp}"


def copy_repository(source: Path, target: Path) -> None:
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*IGNORED_COPY_DIRS))


def write_summary(path: Path, summary: EvalRunSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")


def resolve_prompt_text(
    *,
    variant_name: str,
    prompt_value: str | None,
    repo_ref: str,
    repo: object,
    mode: str,
) -> str:
    if variant_name == "baseline" and prompt_value is None:
        return generate_baseline_prompt_from_repository(repo, mode)
    if variant_name == "generated":
        return generate_prompt_from_repository(repo, mode, prompt_value)
    if prompt_value is None:
        raise ValueError(f"Prompt text is required for variant: {variant_name}")
    return prompt_value


def select_task_validations(
    validations: list[EvalPackValidationResult],
    task_names: list[str] | None,
) -> list[EvalPackValidationResult]:
    if not task_names:
        return validations
    by_name = {validation.root.name: validation for validation in validations}
    missing = [task_name for task_name in task_names if task_name not in by_name]
    if missing:
        raise ValueError(f"Unknown eval-pack tasks: {', '.join(missing)}")
    return [by_name[task_name] for task_name in task_names]
