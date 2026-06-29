from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path

from promptforge.benchmarks import resolve_benchmarks_root, resolve_eval_pack_path
from promptforge.repository import github_full_name, is_github_url

REQUIRED_FILES = (
    "task.md",
    "repo_ref.txt",
    "checks.sh",
    "rubric.md",
    "allowed_paths.txt",
    "forbidden_paths.txt",
)

PLACEHOLDER_MARKERS = {
    "task.md": (
        "Describe the exact change the agent must make.",
        "Describe what a valid final diff should achieve.",
    ),
    "checks.sh": ('echo "TODO: add repo-specific checks"',),
    "rubric.md": (
        "- Task goal is completed.",
        "- Wrong task solved.",
    ),
}


@dataclass(frozen=True)
class EvalPackValidationResult:
    root: Path
    missing_files: list[str]
    empty_files: list[str]
    placeholder_files: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing_files and not self.empty_files and not self.placeholder_files


def init_eval_pack(repo_ref: str, task_id: str, output_root: str | None = None) -> Path:
    repo_slug = repository_slug(repo_ref)
    normalized_task_id = normalize_task_id(task_id)
    base_dir = resolve_benchmarks_root(output_root, require_exists=False)
    pack_dir = base_dir / repo_slug / normalized_task_id
    pack_dir.mkdir(parents=True, exist_ok=False)

    write_file(pack_dir / "task.md", default_task_md(repo_ref, normalized_task_id))
    write_file(pack_dir / "repo_ref.txt", default_repo_ref(repo_ref))
    write_file(pack_dir / "checks.sh", default_checks_sh())
    write_file(pack_dir / "rubric.md", default_rubric_md())
    write_file(pack_dir / "allowed_paths.txt", default_allowed_paths())
    write_file(pack_dir / "forbidden_paths.txt", default_forbidden_paths())
    make_executable(pack_dir / "checks.sh")
    return pack_dir


def validate_eval_pack(path: str) -> EvalPackValidationResult:
    root = resolve_eval_pack_path(path)
    missing_files: list[str] = []
    empty_files: list[str] = []
    placeholder_files: list[str] = []

    for filename in REQUIRED_FILES:
        file_path = root / filename
        if not file_path.exists():
            missing_files.append(filename)
            continue
        if file_path.stat().st_size == 0:
            empty_files.append(filename)
            continue
        if file_contains_placeholder(file_path, filename):
            placeholder_files.append(filename)

    return EvalPackValidationResult(
        root=root,
        missing_files=missing_files,
        empty_files=empty_files,
        placeholder_files=placeholder_files,
    )


def discover_eval_pack_tasks(path: str) -> list[EvalPackValidationResult]:
    root = resolve_eval_pack_path(path)
    direct_result = validate_eval_pack(str(root))
    if direct_result.is_valid:
        return [direct_result]
    if not root.is_dir():
        return [direct_result]

    task_results: list[EvalPackValidationResult] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not looks_like_eval_pack(child):
            continue
        task_results.append(validate_eval_pack(str(child)))
    return task_results or [direct_result]


def render_validation_result(result: EvalPackValidationResult) -> str:
    lines: list[str] = []
    lines.append(f"Eval pack: {result.root}")
    if result.is_valid:
        lines.append("Status: valid")
        lines.append("Required files are present and non-empty.")
        return "\n".join(lines)

    lines.append("Status: invalid")
    if result.missing_files:
        lines.append("Missing files:")
        lines.extend(f"- {name}" for name in result.missing_files)
    if result.empty_files:
        lines.append("Empty files:")
        lines.extend(f"- {name}" for name in result.empty_files)
    if result.placeholder_files:
        lines.append("Placeholder scaffold content still present:")
        lines.extend(f"- {name}" for name in result.placeholder_files)
    return "\n".join(lines)


def repository_slug(repo_ref: str) -> str:
    if is_github_url(repo_ref):
        return github_full_name(repo_ref).replace("/", "__")
    return normalize_task_id(Path(repo_ref).name)


def normalize_task_id(task_id: str) -> str:
    cleaned = []
    for char in task_id.strip().lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {"-", "_", ".", "/", " "}:
            cleaned.append("-")
    normalized = "".join(cleaned).strip("-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    if not normalized:
        raise ValueError("Task id must contain at least one alphanumeric character.")
    return normalized


def default_task_md(repo_ref: str, task_id: str) -> str:
    return (
        f"# Eval Task: {task_id}\n\n"
        f"Repository: `{repo_ref}`\n\n"
        "## Goal\n"
        "- Describe the exact change the agent must make.\n\n"
        "## Constraints\n"
        "- List repo-specific rules the solution must follow.\n"
        "- Reference protected or forbidden paths if relevant.\n\n"
        "## Expected Outcome\n"
        "- Describe what a valid final diff should achieve.\n"
    )


def default_repo_ref(repo_ref: str) -> str:
    return (
        "# Repo source for this eval task.\n"
        "# Optional format: <repo-url-or-path>@<git-ref>\n"
        f"{repo_ref}\n"
    )


def default_checks_sh() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "# Replace these placeholder checks with repo-specific validation commands.\n"
        "echo \"TODO: add repo-specific checks\"\n"
    )


def default_rubric_md() -> str:
    return (
        "# Rubric\n\n"
        "## Pass Conditions\n"
        "- Task goal is completed.\n"
        "- Required checks pass.\n"
        "- Protected or forbidden paths are not touched.\n\n"
        "## Failure Conditions\n"
        "- Wrong task solved.\n"
        "- Required checks fail.\n"
        "- Forbidden paths are edited.\n"
    )


def default_allowed_paths() -> str:
    return "# One path per line. Leave comments if the task has no path restrictions.\n"


def default_forbidden_paths() -> str:
    return "# One path per line. Add maintainer-owned or out-of-scope paths here.\n"


def looks_like_eval_pack(path: Path) -> bool:
    return any((path / filename).exists() for filename in REQUIRED_FILES)


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def make_executable(path: Path) -> None:
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def file_contains_placeholder(path: Path, filename: str) -> bool:
    markers = PLACEHOLDER_MARKERS.get(filename, ())
    if not markers:
        return False
    content = path.read_text(encoding="utf-8")
    return any(marker in content for marker in markers)
