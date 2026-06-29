from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

BENCHMARKS_ROOT_ENV = "PROMPTFORGE_BENCHMARKS_ROOT"
REGISTRY_MARKER_FILENAME = "promptforge-benchmark-registry.json"
DEFAULT_BENCHMARKS_DIR = "benchmarks"
PROMPTFORGE_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkRegistry:
    root: Path
    benchmarks_dir: Path
    marker_path: Path
    schema_version: int
    registry_name: str | None


def resolve_benchmark_registry(
    explicit_root: str | None = None,
    *,
    require_exists: bool = True,
) -> BenchmarkRegistry:
    configured_root = explicit_root or os.environ.get(BENCHMARKS_ROOT_ENV)
    if configured_root:
        return load_registry_from_reference(configured_root, require_exists=require_exists)

    discovered_root = discover_registry_root()
    if discovered_root is None:
        raise FileNotFoundError(
            "Could not find a PromptForge benchmark registry. "
            f"Set {BENCHMARKS_ROOT_ENV} or add {REGISTRY_MARKER_FILENAME} to the "
            "benchmark registry repo."
        )
    return load_registry_from_reference(str(discovered_root), require_exists=require_exists)


def resolve_benchmarks_root(
    explicit_root: str | None = None,
    *,
    require_exists: bool = True,
) -> Path:
    return resolve_benchmark_registry(
        explicit_root=explicit_root,
        require_exists=require_exists,
    ).benchmarks_dir


def resolve_eval_pack_path(
    eval_pack_ref: str,
    *,
    benchmarks_root: str | None = None,
    require_exists: bool = True,
) -> Path:
    direct_path = Path(eval_pack_ref).expanduser()
    if direct_path.exists():
        return direct_path.resolve()

    if looks_like_path(eval_pack_ref):
        raise FileNotFoundError(f"Eval pack path does not exist: {direct_path}")

    pack_path = (
        resolve_benchmarks_root(benchmarks_root, require_exists=require_exists) / eval_pack_ref
    )
    if require_exists and not pack_path.exists():
        raise FileNotFoundError(
            "Could not find the eval pack in the benchmark registry: "
            f"{pack_path}. Pass a filesystem path or a pack id under the "
            f"benchmark root configured by {BENCHMARKS_ROOT_ENV}."
        )
    return pack_path.resolve()


def load_registry_from_reference(
    reference: str,
    *,
    require_exists: bool,
) -> BenchmarkRegistry:
    input_path = Path(reference).expanduser()
    candidate_root = normalize_registry_reference(input_path)
    marker_path = candidate_root / REGISTRY_MARKER_FILENAME

    if require_exists and not marker_path.exists():
        raise FileNotFoundError(
            "Benchmark registry marker not found. Expected: "
            f"{marker_path}. Add {REGISTRY_MARKER_FILENAME} to the registry repo "
            f"or set {BENCHMARKS_ROOT_ENV} to a valid registry root."
        )

    payload = read_registry_payload(marker_path)
    benchmarks_dir_name = payload.get("benchmarks_dir", DEFAULT_BENCHMARKS_DIR)
    if not isinstance(benchmarks_dir_name, str) or not benchmarks_dir_name.strip():
        raise ValueError(
            f"Invalid `benchmarks_dir` in {marker_path}. Expected a non-empty string."
        )

    benchmarks_dir = (candidate_root / benchmarks_dir_name).resolve()
    if require_exists and not benchmarks_dir.exists():
        raise FileNotFoundError(
            "Benchmark registry is missing its benchmarks directory: "
            f"{benchmarks_dir}"
        )

    schema_version = payload.get("schema_version", 1)
    if not isinstance(schema_version, int):
        raise ValueError(f"Invalid `schema_version` in {marker_path}. Expected an integer.")

    registry_name = payload.get("registry_name")
    if registry_name is not None and not isinstance(registry_name, str):
        raise ValueError(f"Invalid `registry_name` in {marker_path}. Expected a string.")

    return BenchmarkRegistry(
        root=candidate_root.resolve(),
        benchmarks_dir=benchmarks_dir,
        marker_path=marker_path.resolve(),
        schema_version=schema_version,
        registry_name=registry_name,
    )


def normalize_registry_reference(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_file():
        if expanded.name != REGISTRY_MARKER_FILENAME:
            raise ValueError(
                "Benchmark registry reference must be a registry root, benchmarks "
                f"directory, or {REGISTRY_MARKER_FILENAME}."
            )
        return expanded.parent

    if (expanded / REGISTRY_MARKER_FILENAME).exists():
        return expanded

    parent_marker = expanded.parent / REGISTRY_MARKER_FILENAME
    if expanded.name == DEFAULT_BENCHMARKS_DIR and parent_marker.exists():
        return expanded.parent

    return expanded


def discover_registry_root() -> Path | None:
    for base_dir in discovery_bases():
        found = discover_registry_under(base_dir)
        if found is not None:
            return found
    return None


def discovery_bases() -> list[Path]:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd.parent,
        PROMPTFORGE_REPO_ROOT,
        PROMPTFORGE_REPO_ROOT.parent,
    ]
    return unique_paths(candidates)


def discover_registry_under(base_dir: Path) -> Path | None:
    if not base_dir.exists() or not base_dir.is_dir():
        return None
    if (base_dir / REGISTRY_MARKER_FILENAME).exists():
        return base_dir

    try:
        children = sorted(base_dir.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return None

    for child in children:
        if child.is_dir() and (child / REGISTRY_MARKER_FILENAME).exists():
            return child
    return None


def read_registry_payload(marker_path: Path) -> dict[str, object]:
    if not marker_path.exists():
        return {"schema_version": 1, "benchmarks_dir": DEFAULT_BENCHMARKS_DIR}

    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Registry marker must contain a JSON object: {marker_path}")
    return payload


def unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        unique.append(path)
        seen.add(key)
    return unique


def looks_like_path(value: str) -> bool:
    return (
        value.startswith(".")
        or value.startswith("~")
        or value.startswith("/")
        or "/" in value
        or "\\" in value
    )
