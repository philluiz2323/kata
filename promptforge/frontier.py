from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from promptforge.baseline import generate_baseline_prompt
from promptforge.benchmarks import resolve_eval_pack_path
from promptforge.eval_pack import discover_eval_pack_tasks
from promptforge.generator import generate_prompt
from promptforge.provenance import (
    EVALUATOR_VERSION,
    pool_fingerprint,
    sha256_file,
    sha256_text,
    short_hash,
)

FRONTIER_SCHEMA_VERSION = 2
FRONTIER_FILENAME = "frontier.json"


@dataclass(frozen=True)
class FrontierModeConfig:
    baseline_prompt: str
    frontier_prompt: str
    primary_tasks: list[str]
    holdout_tasks: list[str] = field(default_factory=list)
    evaluator_version: str | None = None
    baseline_prompt_hash: str | None = None
    frontier_prompt_hash: str | None = None
    primary_pool_fingerprint: str | None = None
    holdout_pool_fingerprint: str | None = None
    frontier_updated_at: str | None = None
    frontier_source: str | None = None


@dataclass(frozen=True)
class FrontierManifest:
    schema_version: int
    repo_ref: str
    eval_pack: str
    modes: dict[str, FrontierModeConfig]
    updated_at: str


def frontier_manifest_path(eval_pack_path: str) -> Path:
    return resolve_eval_pack_path(eval_pack_path) / FRONTIER_FILENAME


def load_frontier_manifest(eval_pack_path: str) -> FrontierManifest:
    path = frontier_manifest_path(eval_pack_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    modes = {
        mode: FrontierModeConfig(**config)
        for mode, config in (payload.get("modes") or {}).items()
    }
    return FrontierManifest(
        schema_version=payload["schema_version"],
        repo_ref=payload["repo_ref"],
        eval_pack=payload["eval_pack"],
        modes=modes,
        updated_at=payload["updated_at"],
    )


def write_frontier_manifest(eval_pack_path: str, manifest: FrontierManifest) -> Path:
    path = frontier_manifest_path(eval_pack_path)
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return path


def init_frontier(
    *,
    repo_ref: str,
    eval_pack_path: str,
    mode: str,
    registry_url: str | None = None,
    primary_tasks: list[str] | None = None,
    holdout_tasks: list[str] | None = None,
) -> FrontierManifest:
    validations = discover_eval_pack_tasks(eval_pack_path)
    invalid = [result.root.name for result in validations if not result.is_valid]
    if invalid:
        raise ValueError(
            "Eval pack is invalid. Run `promptforge eval-pack validate` first. "
            f"Invalid task directories: {', '.join(invalid)}"
        )

    available_tasks = [result.root.name for result in validations]
    task_roots_by_name = {result.root.name: result.root for result in validations}
    selected_primary = primary_tasks or available_tasks
    selected_holdout = holdout_tasks or []
    ensure_known_tasks(selected_primary, available_tasks, label="primary")
    ensure_known_tasks(selected_holdout, available_tasks, label="holdout")
    overlap = sorted(set(selected_primary) & set(selected_holdout))
    if overlap:
        raise ValueError(
            "Primary and holdout pools must not overlap. "
            f"Overlapping task ids: {', '.join(overlap)}"
        )
    if not selected_primary:
        raise ValueError("Frontier init requires at least one primary task.")

    eval_pack_root = resolve_eval_pack_path(eval_pack_path)
    prompt_dir = eval_pack_root / "prompts" / mode
    prompt_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = prompt_dir / "baseline.md"
    frontier_path = prompt_dir / "frontier.md"
    baseline_path.write_text(generate_baseline_prompt(repo_ref, mode) + "\n", encoding="utf-8")
    frontier_path.write_text(generate_prompt(repo_ref, mode, registry_url) + "\n", encoding="utf-8")
    primary_pool = [task_roots_by_name[task_id] for task_id in selected_primary]
    holdout_pool = [task_roots_by_name[task_id] for task_id in selected_holdout]

    manifest = existing_or_new_manifest(repo_ref=repo_ref, eval_pack_path=eval_pack_path)
    updated_modes = dict(manifest.modes)
    updated_modes[mode] = FrontierModeConfig(
        baseline_prompt=str(baseline_path.resolve()),
        frontier_prompt=str(frontier_path.resolve()),
        primary_tasks=selected_primary,
        holdout_tasks=selected_holdout,
        evaluator_version=EVALUATOR_VERSION,
        baseline_prompt_hash=sha256_file(baseline_path),
        frontier_prompt_hash=sha256_file(frontier_path),
        primary_pool_fingerprint=pool_fingerprint(primary_pool),
        holdout_pool_fingerprint=pool_fingerprint(holdout_pool) if holdout_pool else None,
        frontier_updated_at=timestamp_now(),
        frontier_source="promptforge-init",
    )
    updated_manifest = FrontierManifest(
        schema_version=FRONTIER_SCHEMA_VERSION,
        repo_ref=repo_ref,
        eval_pack=str(eval_pack_root),
        modes=updated_modes,
        updated_at=timestamp_now(),
    )
    write_frontier_manifest(eval_pack_path, updated_manifest)
    return updated_manifest


def update_frontier_prompt(
    *,
    eval_pack_path: str,
    mode: str,
    new_prompt_text: str,
    source: str,
    evaluator_version: str | None = None,
) -> FrontierManifest:
    manifest = load_frontier_manifest(eval_pack_path)
    if mode not in manifest.modes:
        raise ValueError(f"Mode is not configured in frontier manifest: {mode}")
    mode_config = manifest.modes[mode]
    frontier_path = Path(mode_config.frontier_prompt)
    frontier_path.write_text(new_prompt_text.rstrip() + "\n", encoding="utf-8")
    frontier_hash = sha256_text(new_prompt_text.rstrip() + "\n")
    updated_modes = dict(manifest.modes)
    updated_modes[mode] = FrontierModeConfig(
        baseline_prompt=mode_config.baseline_prompt,
        frontier_prompt=mode_config.frontier_prompt,
        primary_tasks=mode_config.primary_tasks,
        holdout_tasks=mode_config.holdout_tasks,
        evaluator_version=evaluator_version or mode_config.evaluator_version or EVALUATOR_VERSION,
        baseline_prompt_hash=mode_config.baseline_prompt_hash,
        frontier_prompt_hash=frontier_hash,
        primary_pool_fingerprint=mode_config.primary_pool_fingerprint,
        holdout_pool_fingerprint=mode_config.holdout_pool_fingerprint,
        frontier_updated_at=timestamp_now(),
        frontier_source=source,
    )
    updated_manifest = FrontierManifest(
        schema_version=manifest.schema_version,
        repo_ref=manifest.repo_ref,
        eval_pack=manifest.eval_pack,
        modes=updated_modes,
        updated_at=timestamp_now(),
    )
    write_frontier_manifest(eval_pack_path, updated_manifest)
    return updated_manifest


def render_frontier_manifest(manifest: FrontierManifest, mode: str | None = None) -> str:
    lines: list[str] = []
    lines.append(f"Frontier manifest: `{manifest.eval_pack}`")
    lines.append(f"Repo: `{manifest.repo_ref}`")
    lines.append(f"Updated: {manifest.updated_at}")
    lines.append("")
    modes = [mode] if mode else sorted(manifest.modes)
    for selected_mode in modes:
        mode_config = manifest.modes.get(selected_mode)
        if mode_config is None:
            raise ValueError(f"Mode is not configured in frontier manifest: {selected_mode}")
        lines.append(f"Mode: {selected_mode}")
        lines.append(f"- Baseline prompt: `{mode_config.baseline_prompt}`")
        lines.append(f"- Frontier prompt: `{mode_config.frontier_prompt}`")
        lines.append(f"- Primary tasks: {', '.join(mode_config.primary_tasks)}")
        lines.append(
            "- Holdout tasks: "
            + (", ".join(mode_config.holdout_tasks) if mode_config.holdout_tasks else "none")
        )
        if mode_config.frontier_updated_at:
            lines.append(f"- Frontier updated: {mode_config.frontier_updated_at}")
        if mode_config.frontier_source:
            lines.append(f"- Frontier source: {mode_config.frontier_source}")
        if mode_config.evaluator_version:
            lines.append(f"- Evaluator version: {mode_config.evaluator_version}")
        if mode_config.baseline_prompt_hash:
            lines.append(
                f"- Baseline prompt hash: {short_hash(mode_config.baseline_prompt_hash)}"
            )
        if mode_config.frontier_prompt_hash:
            lines.append(
                f"- Frontier prompt hash: {short_hash(mode_config.frontier_prompt_hash)}"
            )
        if mode_config.primary_pool_fingerprint:
            lines.append(
                "- Primary pool fingerprint: "
                f"{short_hash(mode_config.primary_pool_fingerprint)}"
            )
        if mode_config.holdout_pool_fingerprint:
            lines.append(
                "- Holdout pool fingerprint: "
                f"{short_hash(mode_config.holdout_pool_fingerprint)}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def existing_or_new_manifest(*, repo_ref: str, eval_pack_path: str) -> FrontierManifest:
    path = frontier_manifest_path(eval_pack_path)
    if path.exists():
        return load_frontier_manifest(eval_pack_path)
    return FrontierManifest(
        schema_version=FRONTIER_SCHEMA_VERSION,
        repo_ref=repo_ref,
        eval_pack=str(resolve_eval_pack_path(eval_pack_path)),
        modes={},
        updated_at=timestamp_now(),
    )


def ensure_known_tasks(selected: list[str], available: list[str], *, label: str) -> None:
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown {label} task ids: {', '.join(unknown)}. "
            f"Available tasks: {', '.join(available)}"
        )


def timestamp_now() -> str:
    return datetime.now(UTC).isoformat()
