from __future__ import annotations

import json
from pathlib import Path

from promptforge.benchmarks import (
    BENCHMARKS_ROOT_ENV,
    REGISTRY_MARKER_FILENAME,
    resolve_benchmark_registry,
    resolve_benchmarks_root,
    resolve_eval_pack_path,
)
from promptforge.eval_pack import discover_eval_pack_tasks, init_eval_pack


def write_registry_marker(root: Path, *, benchmarks_dir: str = "benchmarks") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / REGISTRY_MARKER_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "registry_name": "test-registry",
                "benchmarks_dir": benchmarks_dir,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / benchmarks_dir).mkdir(parents=True, exist_ok=True)


def write_task(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.md").write_text("# Task\n\nReal task text.\n", encoding="utf-8")
    (root / "repo_ref.txt").write_text(
        "https://github.com/example/repo.git@main\n",
        encoding="utf-8",
    )
    (root / "checks.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ntrue\n",
        encoding="utf-8",
    )
    (root / "checks.sh").chmod(0o755)
    (root / "rubric.md").write_text(
        "# Rubric\n\n## Pass Conditions\n- Real pass condition.\n",
        encoding="utf-8",
    )
    (root / "allowed_paths.txt").write_text("src/\n", encoding="utf-8")
    (root / "forbidden_paths.txt").write_text("docs/\n", encoding="utf-8")


def test_resolve_benchmark_registry_from_env_repo_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry_marker(registry_root)
    monkeypatch.setenv(BENCHMARKS_ROOT_ENV, str(registry_root))

    registry = resolve_benchmark_registry()

    assert registry.root == registry_root.resolve()
    assert registry.benchmarks_dir == (registry_root / "benchmarks").resolve()
    assert registry.registry_name == "test-registry"


def test_resolve_benchmarks_root_from_explicit_benchmarks_dir(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry_marker(registry_root)

    resolved = resolve_benchmarks_root(str(registry_root / "benchmarks"))

    assert resolved == (registry_root / "benchmarks").resolve()


def test_resolve_eval_pack_path_accepts_pack_id_from_registry_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry_marker(registry_root)
    pack_root = registry_root / "benchmarks" / "example__repo"
    pack_root.mkdir(parents=True)
    monkeypatch.setenv(BENCHMARKS_ROOT_ENV, str(registry_root))

    resolved = resolve_eval_pack_path("example__repo")

    assert resolved == pack_root.resolve()


def test_init_eval_pack_defaults_to_registry_benchmarks_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry_marker(registry_root)
    monkeypatch.setenv(BENCHMARKS_ROOT_ENV, str(registry_root))

    pack_dir = init_eval_pack("https://github.com/example/repo.git", "fix-cli-flag")

    assert pack_dir == (
        registry_root / "benchmarks" / "example__repo" / "fix-cli-flag"
    ).resolve()


def test_discover_eval_pack_tasks_accepts_pack_id_from_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_root = tmp_path / "registry"
    write_registry_marker(registry_root)
    monkeypatch.setenv(BENCHMARKS_ROOT_ENV, str(registry_root))
    task_root = registry_root / "benchmarks" / "example__repo" / "task-a"
    write_task(task_root)

    results = discover_eval_pack_tasks("example__repo")

    assert len(results) == 1
    assert results[0].root == task_root.resolve()
    assert results[0].is_valid


def test_resolve_benchmark_registry_discovers_marker_from_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    promptforge_root = workspace_root / "PromptForge"
    registry_root = workspace_root / "bench-any-name"
    promptforge_root.mkdir(parents=True)
    write_registry_marker(registry_root)
    monkeypatch.delenv(BENCHMARKS_ROOT_ENV, raising=False)
    monkeypatch.chdir(promptforge_root)

    registry = resolve_benchmark_registry()

    assert registry.root == registry_root.resolve()
