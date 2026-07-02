from __future__ import annotations

import ast
import json
import os
import py_compile
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kata.agent_bundle import (
    AGENT_ENTRY_FILENAME,
    AGENT_MANIFEST_FILENAME,
    find_unexpected_bundle_paths,
    is_allowed_bundle_relative_path,
    load_bundle_files,
    validate_agent_manifest,
    write_agent_manifest,
)
from kata.benchmarks import ensure_active_repo_pack, resolve_eval_pack_path
from kata.challenge import (
    SN60_MINER_LANE_ID,
    SN60_MINER_MODE,
    SN60_VALIDATOR_MODEL,
    ChallengeSummary,
    current_holdout_pool_fingerprint,
    current_primary_pool_fingerprint,
    evaluate_promotion,
    load_challenge_summary,
    run_frontier_challenge,
    run_sn60_challenge,
)
from kata.config import resolve_validator_model
from kata.evaluators.sn60_bitsec import (
    DEFAULT_REPLICAS_PER_PROJECT,
    SN60_BITSEC_EVALUATOR_ID,
)
from kata.frontier import (
    load_frontier_manifest,
    promote_frontier_artifact,
    resolve_frontier_artifact_hash,
)
from kata.lane_state import (
    KING_STATE_SCHEMA_VERSION,
    LaneKingState,
    PackRegistryEntry,
    benchmark_snapshot_path,
    challenge_state_path,
    lane_king_state_path,
    load_benchmark_snapshot,
    load_challenge_state,
    load_lane_king_state,
    load_pack_registry,
    write_lane_king_state,
)
from kata.provenance import sha256_directory, short_hash
from kata.public_artifacts import (
    publish_public_king,
    resolve_artifact_path,
    resolve_kata_root,
    resolve_public_king_root,
)
from kata.screening import validate_sn60_static_screening

SUBMISSIONS_DIRNAME = "submissions"
SUBMISSION_SCHEMA_VERSION = 2
SUBMISSION_METADATA_FILENAME = "submission.json"
SUBMISSION_AGENT_FILENAME = AGENT_ENTRY_FILENAME
SUBMISSION_AGENT_MANIFEST_FILENAME = AGENT_MANIFEST_FILENAME
TOP_LEVEL_SUBMISSION_FILENAMES = {
    SUBMISSION_METADATA_FILENAME,
    SUBMISSION_AGENT_FILENAME,
    SUBMISSION_AGENT_MANIFEST_FILENAME,
}
SUPPORTED_SUBMISSION_MODES = {"contributor", "miner", "reviewer"}
DEFAULT_AGENT_PLACEHOLDER = (
    "Replace this scaffold with a real challenger agent implementation before opening a PR."
)
SUBMISSION_ID_CONVENTION = "<github-username>-YYYYMMDD-NN"
PR_ACTION_CLOSE_INVALID = "close-invalid"
PR_ACTION_EVALUATE = "evaluate"
PR_ACTION_CLOSE_LOSING = "close-losing"
PR_ACTION_RERUN_STALE = "rerun-stale"
PR_ACTION_MERGE = "merge"
MAX_SUBMISSION_BUNDLE_FILES = 16
MAX_SUBMISSION_FILE_BYTES = 64 * 1024
MAX_SUBMISSION_BUNDLE_BYTES = 128 * 1024
FORBIDDEN_ENV_REFERENCE_TOKENS = (
    "KATA_VALIDATOR_API_KEY",
    "KATA_VALIDATOR_API_BASE",
    "KATA_VALIDATOR_MODEL",
    "CHUTES_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)
FORBIDDEN_PROVIDER_SUBSTRINGS = (
    "api.openai.com",
    "openrouter.ai",
    "anthropic.com",
    "generativelanguage.googleapis.com",
    "api.groq.com",
    "api.together.xyz",
    "api.fireworks.ai",
    "api.mistral.ai",
    "api.deepseek.com",
    "deepinfra.com",
    "cohere.ai",
)
FORBIDDEN_SAMPLING_NAMES = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "top_a",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
}
REQUIRED_SOLVE_ARGS = ("repo_path", "issue", "model", "api_base", "api_key")
SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{10,}|cpk_[A-Za-z0-9]{10,})"
)


@dataclass(frozen=True)
class SubmissionMetadata:
    schema_version: int
    repo_pack: str
    mode: str
    submission_id: str
    created_at: str
    author: str | None = None
    title: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class SubmissionDescriptor:
    root: Path
    repo_pack: str
    mode: str
    submission_id: str
    agent_path: Path
    agent_manifest_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class SubmissionValidationResult:
    submission_path: str
    repo_pack: str | None
    mode: str | None
    submission_id: str | None
    agent_path: str | None
    metadata_path: str | None
    changed_paths: list[str]
    off_scope_paths: list[str]
    reasons: list[str]
    metadata: SubmissionMetadata | None
    evaluator_id: str | None = None

    @property
    def is_valid(self) -> bool:
        return not self.reasons and not self.off_scope_paths


@dataclass(frozen=True)
class SubmissionVerificationResult:
    submission_path: str
    challenge_summary_path: str
    repo_pack: str
    mode: str
    submission_id: str
    candidate_artifact_hash: str
    recorded_candidate_artifact_hash: str
    current_frontier_artifact_hash: str
    recorded_frontier_artifact_hash: str
    current_validator_model: str
    recorded_validator_model: str
    submission_matches_challenge: bool
    frontier_is_current: bool
    benchmark_is_current: bool
    promotion_ready: bool
    auto_merge_ready: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PullRequestInspectionResult:
    action: str
    submission_path: str | None
    repo_pack: str | None
    mode: str | None
    submission_id: str | None
    changed_paths: list[str]
    reasons: list[str]
    candidate_submission_dirs: list[str]


@dataclass(frozen=True)
class SubmissionDecisionResult:
    action: str
    submission_path: str
    challenge_summary_path: str
    repo_pack: str
    mode: str
    submission_id: str
    reason: str
    reasons: list[str]
    promotion_ready: bool
    auto_merge_ready: bool


def init_submission(
    *,
    repo_pack: str,
    mode: str,
    submission_id: str,
    output_root: str | None = None,
    author: str | None = None,
    title: str | None = None,
    notes: str | None = None,
) -> Path:
    validate_submission_mode(mode)
    lane_reasons = validate_submission_lane(repo_pack, mode)
    if lane_reasons:
        raise ValueError("; ".join(lane_reasons))
    effective_author = author.strip() if author and author.strip() else None
    root_base = (
        Path(output_root).expanduser().resolve()
        if output_root
        else default_submissions_root()
    )
    submission_root = root_base / repo_pack / mode / submission_id
    submission_root.mkdir(parents=True, exist_ok=False)
    metadata = SubmissionMetadata(
        schema_version=SUBMISSION_SCHEMA_VERSION,
        repo_pack=repo_pack,
        mode=mode,
        submission_id=submission_id,
        created_at=datetime.now(UTC).isoformat(),
        author=effective_author,
        title=title,
        notes=notes or default_submission_notes(mode),
    )
    write_submission_metadata(submission_root / SUBMISSION_METADATA_FILENAME, metadata)
    write_agent_manifest(submission_root / SUBMISSION_AGENT_MANIFEST_FILENAME)
    agent_path = submission_root / SUBMISSION_AGENT_FILENAME
    agent_path.write_text(default_submission_agent(mode), encoding="utf-8")
    return submission_root


def validate_submission(
    submission_path: str,
    *,
    changed_paths: list[str] | None = None,
    repo_root: str | None = None,
) -> SubmissionValidationResult:
    reasons: list[str] = []
    off_scope_paths: list[str] = []
    metadata: SubmissionMetadata | None = None

    resolved_repo_root = Path(repo_root).expanduser().resolve() if repo_root else None
    root = Path(submission_path).expanduser().resolve()
    descriptor, descriptor_errors = resolve_submission_descriptor(
        root,
        repo_root=resolved_repo_root,
    )
    reasons.extend(descriptor_errors)
    normalized_changed = normalize_changed_paths(changed_paths or [])

    if descriptor is None:
        return SubmissionValidationResult(
            submission_path=str(root),
            repo_pack=None,
            mode=None,
            submission_id=None,
            agent_path=None,
            metadata_path=None,
            changed_paths=normalized_changed,
            off_scope_paths=[],
            reasons=reasons,
            metadata=None,
        )

    symlink_paths = find_bundle_symlink_paths(descriptor.root)
    if symlink_paths:
        reasons.append(
            "Submission bundle must not contain symlinks: " + ", ".join(symlink_paths)
        )
        return SubmissionValidationResult(
            submission_path=str(descriptor.root),
            repo_pack=descriptor.repo_pack,
            mode=descriptor.mode,
            submission_id=descriptor.submission_id,
            agent_path=str(descriptor.agent_path),
            metadata_path=str(descriptor.metadata_path),
            changed_paths=normalized_changed,
            off_scope_paths=off_scope_paths,
            reasons=dedupe(reasons),
            metadata=None,
        )

    metadata_path = descriptor.metadata_path
    agent_path = descriptor.agent_path
    agent_manifest_path = descriptor.agent_manifest_path

    if normalized_changed:
        changed_scope = validate_changed_paths(descriptor, normalized_changed)
        off_scope_paths.extend(changed_scope.off_scope_paths)
        reasons.extend(changed_scope.reasons)

    if not metadata_path.exists():
        reasons.append(f"Missing required submission file: {metadata_path.name}")
    else:
        try:
            metadata = load_submission_metadata(metadata_path)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            reasons.append(str(exc))

    if not agent_path.exists():
        reasons.append(f"Missing required submission file: {agent_path.name}")
    else:
        agent_text = agent_path.read_text(encoding="utf-8").strip()
        if not agent_text:
            reasons.append("Submission agent file is empty.")
        elif DEFAULT_AGENT_PLACEHOLDER in agent_text:
            reasons.append("Submission agent still contains scaffold placeholder text.")
        if not agent_defines_required_entrypoint(agent_text, descriptor.mode):
            reasons.append(required_submission_entrypoint_reason(descriptor.mode))

    if not agent_manifest_path.exists():
        reasons.append(f"Missing required submission file: {agent_manifest_path.name}")
    else:
        reasons.extend(validate_agent_manifest(agent_manifest_path))

    if metadata is not None:
        reasons.extend(validate_submission_metadata(metadata, descriptor))
        reasons.extend(validate_submission_target(metadata))
        if agent_path.exists():
            reasons.extend(
                validate_submission_candidate(
                    metadata=metadata,
                    submission_root=descriptor.root,
                )
            )

    evaluator_entry = find_evaluator_pack_entry(descriptor.repo_pack, descriptor.mode)
    return SubmissionValidationResult(
        submission_path=str(descriptor.root),
        repo_pack=descriptor.repo_pack,
        mode=descriptor.mode,
        submission_id=descriptor.submission_id,
        agent_path=str(agent_path),
        metadata_path=str(metadata_path),
        changed_paths=normalized_changed,
        off_scope_paths=off_scope_paths,
        reasons=dedupe(reasons),
        metadata=metadata,
        evaluator_id=evaluator_entry.evaluator_id if evaluator_entry else None,
    )


def evaluate_submission(
    submission_path: str,
    *,
    agent_command: str,
    output_root: str | None = None,
    agent_timeout_seconds: int | None = None,
    checks_timeout_seconds: int | None = None,
    sn60_project_keys: list[str] | None = None,
    sn60_replicas_per_project: int | None = None,
    sn60_sandbox_root: str | None = None,
    sn60_benchmark_file: str | None = None,
    sn60_sandbox_commit: str | None = None,
) -> ChallengeSummary:
    validation = validate_submission(submission_path)
    if (
        not validation.is_valid
        or validation.metadata is None
        or validation.agent_path is None
    ):
        raise ValueError(
            "Submission is invalid. Run `kata submission validate` first. "
            + "; ".join(validation.reasons or ["unknown validation failure"])
        )

    if is_sn60_miner_metadata(validation.metadata):
        project_keys = sn60_project_keys or parse_sn60_project_keys_from_env()
        if not project_keys:
            raise ValueError(
                "SN60 miner evaluation requires at least one project key. "
                "Pass --sn60-project-key or set KATA_SN60_PROJECT_KEYS."
            )
        lane_id, frontier_artifact_path = resolve_sn60_king_artifact(validation.metadata)
        return run_sn60_challenge(
            frontier_artifact_path=frontier_artifact_path,
            candidate_artifact_path=validation.submission_path,
            project_keys=project_keys,
            candidate_submission_id=validation.metadata.submission_id,
            lane_id=lane_id,
            output_root=output_root,
            replicas_per_project=sn60_replicas_per_project or DEFAULT_REPLICAS_PER_PROJECT,
            sandbox_root=sn60_sandbox_root,
            benchmark_file=sn60_benchmark_file,
            sandbox_commit=sn60_sandbox_commit,
        )

    return run_frontier_challenge(
        eval_pack_path=validation.metadata.repo_pack,
        mode=validation.metadata.mode,
        candidate_artifact_path=validation.submission_path,
        agent_command=agent_command,
        output_root=output_root,
        agent_timeout_seconds=agent_timeout_seconds,
        checks_timeout_seconds=checks_timeout_seconds,
    )


def parse_sn60_project_keys_from_env() -> list[str]:
    configured = os.environ.get("KATA_SN60_PROJECT_KEYS", "")
    return [part.strip() for part in configured.split(",") if part.strip()]


def is_sn60_miner_metadata(metadata: SubmissionMetadata) -> bool:
    # Evaluator adapters are selected by the pack registry's evaluator id;
    # the SN60 lane id is only a fallback for pre-registry lanes.
    entry = find_evaluator_pack_entry(metadata.repo_pack, metadata.mode)
    if entry is not None:
        return entry.evaluator_id == SN60_BITSEC_EVALUATOR_ID
    return metadata.repo_pack == SN60_MINER_LANE_ID and metadata.mode == SN60_MINER_MODE


def resolve_sn60_lane_king_hash(
    lane_id: str,
    *,
    repo_pack: str,
    mode: str,
) -> str | None:
    """Resolve the current king artifact hash for a registry-backed SN60 lane."""
    if lane_king_state_path(lane_id).exists():
        king = load_lane_king_state(lane_id)
        if king.current_king_artifact_hash:
            return king.current_king_artifact_hash
    king_root = resolve_public_king_root(public_root=None, repo_pack=repo_pack, mode=mode)
    if (king_root / SUBMISSION_AGENT_FILENAME).exists():
        return hash_submission_bundle(king_root)
    return None


def sn60_lane_benchmark_is_current(lane_id: str, summary: ChallengeSummary) -> bool:
    """Freshness check against the lane's recorded benchmark snapshot and fingerprint."""
    if not benchmark_snapshot_path(lane_id).exists():
        return False
    snapshot = load_benchmark_snapshot(lane_id)
    expected_version = f"{snapshot.scorer_version}@{short_hash(snapshot.sandbox_commit_hash)}"
    if summary.evaluator_version != expected_version:
        return False
    if challenge_state_path(lane_id).exists():
        challenge_state = load_challenge_state(lane_id)
        if challenge_state.freshness_fingerprint != summary.primary_pool_fingerprint:
            return False
    return True


def resolve_sn60_king_artifact(metadata: SubmissionMetadata) -> tuple[str, str]:
    """Resolve (lane_id, king_artifact_path) for an SN60 duel.

    Registry-backed lanes use the published king under kings/<repo-pack>/<mode>/;
    pre-registry lanes fall back to the legacy frontier manifest.
    """
    entry = find_evaluator_pack_entry(metadata.repo_pack, metadata.mode)
    if entry is not None:
        king_root = resolve_public_king_root(
            public_root=None,
            repo_pack=metadata.repo_pack,
            mode=metadata.mode,
        )
        if not (king_root / SUBMISSION_AGENT_FILENAME).exists():
            raise ValueError(
                f"SN60 lane king artifact is not seeded: {king_root}. "
                "Seed the current king under kings/<repo-pack>/<mode>/ before running duels."
            )
        return entry.lane_id, str(king_root)

    manifest = load_frontier_manifest(metadata.repo_pack)
    mode_config = manifest.modes.get(metadata.mode)
    if mode_config is None:
        raise ValueError(f"Mode is not configured in frontier manifest: {metadata.mode}")
    return metadata.repo_pack, str(resolve_artifact_path(mode_config.frontier_artifact))


def inspect_pull_request(
    *,
    repo_root: str,
    changed_paths: list[str],
) -> PullRequestInspectionResult:
    resolved_repo_root = Path(repo_root).expanduser().resolve()
    normalized_changed = normalize_changed_paths(changed_paths)
    candidate_dirs = infer_submission_dirs(normalized_changed)
    reasons: list[str] = []

    if not normalized_changed:
        reasons.append("PR does not contain any changed files.")
        return PullRequestInspectionResult(
            action=PR_ACTION_CLOSE_INVALID,
            submission_path=None,
            repo_pack=None,
            mode=None,
            submission_id=None,
            changed_paths=[],
            reasons=reasons,
            candidate_submission_dirs=[],
        )

    if not candidate_dirs:
        reasons.append(
            "PR does not contain an agent submission under "
            "`submissions/<repo-pack>/<mode>/<submission-id>`."
        )
        return PullRequestInspectionResult(
            action=PR_ACTION_CLOSE_INVALID,
            submission_path=None,
            repo_pack=None,
            mode=None,
            submission_id=None,
            changed_paths=normalized_changed,
            reasons=reasons,
            candidate_submission_dirs=[],
        )

    if len(candidate_dirs) > 1:
        reasons.append("PR touches multiple submission directories. Submit exactly one challenger.")
        return PullRequestInspectionResult(
            action=PR_ACTION_CLOSE_INVALID,
            submission_path=None,
            repo_pack=None,
            mode=None,
            submission_id=None,
            changed_paths=normalized_changed,
            reasons=reasons,
            candidate_submission_dirs=candidate_dirs,
        )

    relative_dir = candidate_dirs[0]
    descriptor, descriptor_errors = resolve_submission_descriptor(
        resolved_repo_root / relative_dir,
        repo_root=resolved_repo_root,
        require_exists=False,
    )
    reasons.extend(descriptor_errors)
    if descriptor is None:
        return PullRequestInspectionResult(
            action=PR_ACTION_CLOSE_INVALID,
            submission_path=None,
            repo_pack=None,
            mode=None,
            submission_id=None,
            changed_paths=normalized_changed,
            reasons=dedupe(reasons),
            candidate_submission_dirs=candidate_dirs,
        )

    changed_scope = validate_changed_paths(descriptor, normalized_changed)
    reasons.extend(changed_scope.reasons)
    if changed_scope.off_scope_paths:
        reasons.append(
            "PR changes files outside the allowed submission directory or adds unsupported files."
        )
    reasons.extend(validate_submission_lane(descriptor.repo_pack, descriptor.mode))

    action = PR_ACTION_EVALUATE if not reasons else PR_ACTION_CLOSE_INVALID
    return PullRequestInspectionResult(
        action=action,
        submission_path=str((resolved_repo_root / relative_dir).resolve()),
        repo_pack=descriptor.repo_pack,
        mode=descriptor.mode,
        submission_id=descriptor.submission_id,
        changed_paths=normalized_changed,
        reasons=dedupe(reasons),
        candidate_submission_dirs=candidate_dirs,
    )


def verify_submission_result(
    submission_path: str,
    challenge_summary_path: str,
) -> SubmissionVerificationResult:
    validation = validate_submission(submission_path)
    if (
        not validation.is_valid
        or validation.metadata is None
        or validation.agent_path is None
    ):
        raise ValueError(
            "Submission is invalid. Run `kata submission validate` first. "
            + "; ".join(validation.reasons or ["unknown validation failure"])
        )

    summary = load_challenge_summary(challenge_summary_path)
    candidate_hash = hash_submission_bundle(Path(validation.submission_path))

    if is_sn60_miner_metadata(validation.metadata):
        evaluator_entry = find_evaluator_pack_entry(
            validation.metadata.repo_pack, validation.metadata.mode
        )
        if evaluator_entry is not None:
            current_frontier_hash = (
                resolve_sn60_lane_king_hash(
                    evaluator_entry.lane_id,
                    repo_pack=validation.metadata.repo_pack,
                    mode=validation.metadata.mode,
                )
                or ""
            )
            lane_benchmark_is_current = sn60_lane_benchmark_is_current(
                evaluator_entry.lane_id, summary
            )
        else:
            manifest = load_frontier_manifest(validation.metadata.repo_pack)
            mode_config = manifest.modes.get(validation.metadata.mode)
            if mode_config is None:
                raise ValueError(
                    f"Mode is not configured in frontier manifest: {validation.metadata.mode}"
                )
            current_frontier_hash = resolve_frontier_artifact_hash(mode_config)
            lane_benchmark_is_current = True
        submission_matches = (
            summary.mode == validation.metadata.mode
            and summary.candidate_artifact_hash == candidate_hash
        )
        frontier_is_current = summary.frontier_artifact_hash == current_frontier_hash
        benchmark_is_current = (
            summary.validator_model == SN60_VALIDATOR_MODEL and lane_benchmark_is_current
        )
        current_promotion_ready = summary.promotion_ready

        reasons: list[str] = []
        if not submission_matches:
            reasons.append("Challenge result does not match the current submission payload.")
        if not frontier_is_current:
            reasons.append("Challenge result is stale because the frontier artifact has changed.")
        if not benchmark_is_current:
            reasons.append("Challenge result is stale because the SN60 benchmark lane has changed.")
        if not current_promotion_ready:
            reasons.append(f"Challenge is not promotion-ready: {summary.promotion_reason}")

        return SubmissionVerificationResult(
            submission_path=validation.submission_path,
            challenge_summary_path=str(Path(challenge_summary_path).expanduser().resolve()),
            repo_pack=validation.metadata.repo_pack,
            mode=validation.metadata.mode,
            submission_id=validation.metadata.submission_id,
            candidate_artifact_hash=candidate_hash,
            recorded_candidate_artifact_hash=summary.candidate_artifact_hash,
            current_frontier_artifact_hash=current_frontier_hash,
            recorded_frontier_artifact_hash=summary.frontier_artifact_hash,
            current_validator_model=SN60_VALIDATOR_MODEL,
            recorded_validator_model=summary.validator_model,
            submission_matches_challenge=submission_matches,
            frontier_is_current=frontier_is_current,
            benchmark_is_current=benchmark_is_current,
            promotion_ready=current_promotion_ready,
            auto_merge_ready=submission_matches
            and frontier_is_current
            and benchmark_is_current
            and current_promotion_ready,
            reasons=reasons,
        )

    manifest = load_frontier_manifest(validation.metadata.repo_pack)
    mode_config = manifest.modes.get(validation.metadata.mode)
    if mode_config is None:
        raise ValueError(
            f"Mode is not configured in frontier manifest: {validation.metadata.mode}"
        )
    current_frontier_hash = resolve_frontier_artifact_hash(mode_config)
    current_validator_model = resolve_validator_model()
    current_primary_fingerprint = current_primary_pool_fingerprint(
        validation.metadata.repo_pack,
        mode_config,
        selected_task_ids=summary.primary.task_ids,
    )
    current_holdout_fingerprint = current_holdout_pool_fingerprint(
        validation.metadata.repo_pack,
        mode_config,
    )

    expected_manifest_path = (
        resolve_eval_pack_path(validation.metadata.repo_pack) / "frontier.json"
    ).resolve()
    submission_matches = (
        summary.mode == validation.metadata.mode
        and summary.candidate_artifact_hash == candidate_hash
        and Path(summary.manifest_path).resolve() == expected_manifest_path
    )
    frontier_is_current = summary.frontier_artifact_hash == current_frontier_hash
    benchmark_is_current = (
        summary.evaluator_version == (mode_config.evaluator_version or summary.evaluator_version)
        and summary.validator_model == current_validator_model
        and summary.primary_pool_fingerprint == current_primary_fingerprint
        and summary.holdout_pool_fingerprint == current_holdout_fingerprint
    )
    current_promotion_ready, current_promotion_reason = evaluate_promotion(
        summary.primary,
        summary.holdout,
        promotion_margin_points=mode_config.promotion_margin_points,
        holdout_promotion_margin_points=mode_config.holdout_promotion_margin_points,
    )

    reasons: list[str] = []
    if not submission_matches:
        reasons.append("Challenge result does not match the current submission payload.")
    if not frontier_is_current:
        reasons.append("Challenge result is stale because the frontier artifact has changed.")
    if summary.validator_model != current_validator_model:
        reasons.append("Challenge result is stale because the validator model has changed.")
    elif not benchmark_is_current:
        reasons.append("Challenge result is stale because the benchmark lane has changed.")
    if not current_promotion_ready:
        reasons.append(f"Challenge is not promotion-ready: {current_promotion_reason}")

    return SubmissionVerificationResult(
        submission_path=validation.submission_path,
        challenge_summary_path=str(Path(challenge_summary_path).expanduser().resolve()),
        repo_pack=validation.metadata.repo_pack,
        mode=validation.metadata.mode,
        submission_id=validation.metadata.submission_id,
        candidate_artifact_hash=candidate_hash,
        recorded_candidate_artifact_hash=summary.candidate_artifact_hash,
        current_frontier_artifact_hash=current_frontier_hash,
        recorded_frontier_artifact_hash=summary.frontier_artifact_hash,
        current_validator_model=current_validator_model,
        recorded_validator_model=summary.validator_model,
        submission_matches_challenge=submission_matches,
        frontier_is_current=frontier_is_current,
        benchmark_is_current=benchmark_is_current,
        promotion_ready=current_promotion_ready,
        auto_merge_ready=submission_matches
        and frontier_is_current
        and benchmark_is_current
        and current_promotion_ready,
        reasons=reasons,
    )


def decide_submission_action(
    submission_path: str,
    challenge_summary_path: str,
) -> SubmissionDecisionResult:
    validation = validate_submission(submission_path)
    if not validation.is_valid or validation.metadata is None:
        reasons = validation.reasons or ["Submission is invalid."]
        return SubmissionDecisionResult(
            action=PR_ACTION_CLOSE_INVALID,
            submission_path=validation.submission_path,
            challenge_summary_path=str(Path(challenge_summary_path).expanduser().resolve()),
            repo_pack=validation.repo_pack or "unknown",
            mode=validation.mode or "unknown",
            submission_id=validation.submission_id or "unknown",
            reason="Submission is invalid and should be auto-closed.",
            reasons=reasons,
            promotion_ready=False,
            auto_merge_ready=False,
        )

    verification = verify_submission_result(submission_path, challenge_summary_path)
    if verification.auto_merge_ready:
        return SubmissionDecisionResult(
            action=PR_ACTION_MERGE,
            submission_path=verification.submission_path,
            challenge_summary_path=verification.challenge_summary_path,
            repo_pack=verification.repo_pack,
            mode=verification.mode,
            submission_id=verification.submission_id,
            reason="Submission beat the current frontier and is safe to auto-merge.",
            reasons=[],
            promotion_ready=verification.promotion_ready,
            auto_merge_ready=verification.auto_merge_ready,
        )

    stale_reasons = [
        reason
        for reason in verification.reasons
        if "stale" in reason or "does not match" in reason
    ]
    if stale_reasons:
        return SubmissionDecisionResult(
            action=PR_ACTION_RERUN_STALE,
            submission_path=verification.submission_path,
            challenge_summary_path=verification.challenge_summary_path,
            repo_pack=verification.repo_pack,
            mode=verification.mode,
            submission_id=verification.submission_id,
            reason="Submission result is stale and must be rerun against the current frontier.",
            reasons=stale_reasons,
            promotion_ready=verification.promotion_ready,
            auto_merge_ready=False,
        )

    losing_reasons = verification.reasons or [
        "Submission did not satisfy the promotion rule against the current frontier."
    ]
    return SubmissionDecisionResult(
        action=PR_ACTION_CLOSE_LOSING,
        submission_path=verification.submission_path,
        challenge_summary_path=verification.challenge_summary_path,
        repo_pack=verification.repo_pack,
        mode=verification.mode,
        submission_id=verification.submission_id,
        reason="Submission lost to the current frontier and should be auto-closed.",
        reasons=losing_reasons,
        promotion_ready=verification.promotion_ready,
        auto_merge_ready=False,
    )


def promote_submission_result(
    submission_path: str,
    challenge_summary_path: str,
    *,
    public_root: str | None = None,
):
    verification = verify_submission_result(submission_path, challenge_summary_path)
    if not verification.auto_merge_ready:
        raise ValueError(
            "Submission is not safe to promote. "
            + "; ".join(
                verification.reasons
                or ["submission result is not auto-merge ready"]
            )
        )

    summary = load_challenge_summary(challenge_summary_path)
    evaluator_entry = find_evaluator_pack_entry(verification.repo_pack, verification.mode)
    if evaluator_entry is not None:
        return promote_lane_king(
            entry=evaluator_entry,
            verification=verification,
            summary=summary,
            public_root=public_root,
        )

    eval_pack_path = (
        verification.repo_pack
        if verification.repo_pack == SN60_MINER_LANE_ID
        and verification.mode == SN60_MINER_MODE
        else Path(summary.manifest_path).parent.as_posix()
    )
    manifest = promote_frontier_artifact(
        eval_pack_path=eval_pack_path,
        mode=summary.mode,
        candidate_artifact_path=verification.submission_path,
        source=summary.run_id,
        evaluator_version=summary.evaluator_version,
    )
    if public_root is not None:
        publish_public_king(
            public_root=public_root,
            repo_pack=verification.repo_pack,
            mode=verification.mode,
            submission_id=verification.submission_id,
            challenge_run_id=summary.run_id,
            candidate_artifact_path=verification.submission_path,
            frontier_artifact_hash=manifest.modes[verification.mode].frontier_artifact_hash or "",
            candidate_artifact_hash=summary.candidate_artifact_hash,
        )
    return manifest


@dataclass(frozen=True)
class LanePromotionResult:
    lane_id: str
    king_root: str
    king: LaneKingState


def promote_lane_king(
    *,
    entry: PackRegistryEntry,
    verification: SubmissionVerificationResult,
    summary: ChallengeSummary,
    public_root: str | None = None,
):
    king_root = publish_public_king(
        public_root=str(resolve_kata_root(public_root)),
        repo_pack=verification.repo_pack,
        mode=verification.mode,
        submission_id=verification.submission_id,
        challenge_run_id=summary.run_id,
        candidate_artifact_path=verification.submission_path,
        frontier_artifact_hash=verification.candidate_artifact_hash,
        candidate_artifact_hash=verification.candidate_artifact_hash,
    )
    now = datetime.now(UTC).isoformat()
    king = LaneKingState(
        schema_version=KING_STATE_SCHEMA_VERSION,
        current_king_submission_id=verification.submission_id,
        current_king_artifact_hash=verification.candidate_artifact_hash,
        promotion_source_pr=None,
        promotion_timestamp=now,
        updated_at=now,
    )
    write_lane_king_state(entry.lane_id, king, public_root=public_root)
    return LanePromotionResult(
        lane_id=entry.lane_id,
        king_root=str(king_root),
        king=king,
    )


def render_submission_validation(result: SubmissionValidationResult) -> str:
    lines: list[str] = []
    lines.append(f"Submission: {result.submission_path}")
    if result.repo_pack:
        lines.append(f"Repo pack: {result.repo_pack}")
    if result.mode:
        lines.append(f"Mode: {result.mode}")
    if result.submission_id:
        lines.append(f"Submission id: {result.submission_id}")
    if result.agent_path:
        lines.append(f"Agent file: {result.agent_path}")
    lines.append(f"Status: {'valid' if result.is_valid else 'invalid'}")
    if result.changed_paths:
        lines.append("Changed paths:")
        lines.extend(f"- {path}" for path in result.changed_paths)
    if result.off_scope_paths:
        lines.append("Off-scope paths:")
        lines.extend(f"- {path}" for path in result.off_scope_paths)
    if result.reasons:
        lines.append("Reasons:")
        lines.extend(f"- {reason}" for reason in result.reasons)
    return "\n".join(lines)


def render_pull_request_inspection(result: PullRequestInspectionResult) -> str:
    lines = [
        f"Action: {result.action}",
        f"Changed paths: {len(result.changed_paths)}",
    ]
    if result.submission_path:
        lines.append(f"Submission path: {result.submission_path}")
    if result.candidate_submission_dirs:
        lines.append("Candidate submission dirs:")
        lines.extend(f"- {path}" for path in result.candidate_submission_dirs)
    if result.reasons:
        lines.append("Reasons:")
        lines.extend(f"- {reason}" for reason in result.reasons)
    return "\n".join(lines)


def render_submission_verification(result: SubmissionVerificationResult) -> str:
    lines: list[str] = []
    lines.append(f"Submission: {result.submission_path}")
    lines.append(f"Challenge summary: {result.challenge_summary_path}")
    lines.append(f"Repo pack: {result.repo_pack}")
    lines.append(f"Mode: {result.mode}")
    lines.append(f"Submission id: {result.submission_id}")
    lines.append(
        "Submission matches challenge: "
        + ("yes" if result.submission_matches_challenge else "no")
    )
    lines.append(f"Frontier is current: {'yes' if result.frontier_is_current else 'no'}")
    lines.append(f"Benchmark lane is current: {'yes' if result.benchmark_is_current else 'no'}")
    lines.append(f"Promotion ready: {'yes' if result.promotion_ready else 'no'}")
    lines.append(f"Auto-merge ready: {'yes' if result.auto_merge_ready else 'no'}")
    if result.reasons:
        lines.append("Reasons:")
        lines.extend(f"- {reason}" for reason in result.reasons)
    return "\n".join(lines)


def render_submission_decision(result: SubmissionDecisionResult) -> str:
    lines = [
        f"Action: {result.action}",
        f"Submission: {result.submission_path}",
        f"Challenge summary: {result.challenge_summary_path}",
        f"Reason: {result.reason}",
        f"Promotion ready: {'yes' if result.promotion_ready else 'no'}",
        f"Auto-merge ready: {'yes' if result.auto_merge_ready else 'no'}",
    ]
    if result.reasons:
        lines.append("Reasons:")
        lines.extend(f"- {reason}" for reason in result.reasons)
    return "\n".join(lines)


def render_pr_comment(action: str, title: str, reasons: list[str]) -> str:
    lines = [f"### {title}", ""]
    if reasons:
        lines.append("Reasons:")
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("No additional reasons recorded.")
    lines.append("")
    lines.append(f"Action: `{action}`")
    return "\n".join(lines)


def render_submission_json(
    value: SubmissionValidationResult
    | SubmissionVerificationResult
    | PullRequestInspectionResult
    | SubmissionDecisionResult,
) -> str:
    payload = asdict(value)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        payload["metadata"] = metadata
    return json.dumps(payload, indent=2) + "\n"


@dataclass(frozen=True)
class ChangedPathValidation:
    off_scope_paths: list[str]
    reasons: list[str]


def validate_changed_paths(
    descriptor: SubmissionDescriptor,
    changed_paths: list[str],
) -> ChangedPathValidation:
    expected_prefix = (
        Path(SUBMISSIONS_DIRNAME)
        / descriptor.repo_pack
        / descriptor.mode
        / descriptor.submission_id
    ).as_posix() + "/"
    off_scope_paths: list[str] = []
    reasons: list[str] = []
    touched_bundle_file = False

    for changed_path in changed_paths:
        normalized = changed_path.strip("/")
        if not normalized.startswith(expected_prefix):
            off_scope_paths.append(normalized)
            continue
        relative_name = normalized.removeprefix(expected_prefix)
        if (
            "/" not in relative_name
            and relative_name in TOP_LEVEL_SUBMISSION_FILENAMES
        ) or is_allowed_bundle_relative_path(relative_name):
            if is_allowed_bundle_relative_path(relative_name):
                touched_bundle_file = True
            continue
        else:
            off_scope_paths.append(normalized)

    if off_scope_paths:
        reasons.append("Submission PR touches paths outside the allowed submission scope.")
    if not touched_bundle_file:
        reasons.append("Submission PR must modify at least one agent bundle file.")

    return ChangedPathValidation(
        off_scope_paths=off_scope_paths,
        reasons=reasons,
    )


def validate_submission_metadata(
    metadata: SubmissionMetadata,
    descriptor: SubmissionDescriptor,
) -> list[str]:
    reasons: list[str] = []
    if metadata.schema_version != SUBMISSION_SCHEMA_VERSION:
        reasons.append(
            "Unsupported submission schema version: "
            f"{metadata.schema_version}. Expected {SUBMISSION_SCHEMA_VERSION}."
        )
    if metadata.repo_pack != descriptor.repo_pack:
        reasons.append(
            "submission.json repo_pack does not match the submission path."
        )
    if metadata.mode != descriptor.mode:
        reasons.append("submission.json mode does not match the submission path.")
    if metadata.submission_id != descriptor.submission_id:
        reasons.append(
            "submission.json submission_id does not match the submission path."
        )
    return reasons


def validate_submission_target(metadata: SubmissionMetadata) -> list[str]:
    return validate_submission_lane(metadata.repo_pack, metadata.mode)


def validate_submission_candidate(
    *,
    metadata: SubmissionMetadata,
    submission_root: Path,
) -> list[str]:
    reasons: list[str] = []
    unexpected_paths = find_unexpected_bundle_paths(submission_root)
    if unexpected_paths:
        reasons.append(
            "Submission bundle contains unsupported files: " + ", ".join(unexpected_paths)
        )

    symlink_paths = find_bundle_symlink_paths(submission_root)
    if symlink_paths:
        reasons.append(
            "Submission bundle must not contain symlinks: " + ", ".join(symlink_paths)
        )

    bundle_paths = find_bundle_relative_paths(submission_root)
    if len(bundle_paths) > MAX_SUBMISSION_BUNDLE_FILES:
        reasons.append(
            "Submission bundle is too large. "
            f"Found {len(bundle_paths)} files; limit is {MAX_SUBMISSION_BUNDLE_FILES}."
        )

    total_bytes = 0
    for relative_path in bundle_paths:
        file_path = submission_root / relative_path
        file_bytes = file_path.stat().st_size
        total_bytes += file_bytes
        if file_bytes > MAX_SUBMISSION_FILE_BYTES:
            reasons.append(
                f"Submission bundle file is too large: {relative_path} "
                f"({file_bytes} bytes; limit is {MAX_SUBMISSION_FILE_BYTES})."
            )
    if total_bytes > MAX_SUBMISSION_BUNDLE_BYTES:
        reasons.append(
            "Submission bundle total size is too large. "
            f"Found {total_bytes} bytes; limit is {MAX_SUBMISSION_BUNDLE_BYTES}."
        )

    bundle_files = load_bundle_files(submission_root)
    if metadata.mode == "miner":
        reasons.extend(validate_sn60_static_screening(submission_root))

    reasons.extend(validate_bundle_python_sources(bundle_files, mode=metadata.mode))
    reasons.extend(validate_bundle_static_policy(bundle_files, mode=metadata.mode))
    reasons.extend(
        validate_submission_not_copycat(
            metadata=metadata,
            submission_root=submission_root,
            bundle_files=bundle_files,
        )
    )
    return dedupe(reasons)


def find_evaluator_pack_entry(repo_pack: str, mode: str) -> PackRegistryEntry | None:
    try:
        registry = load_pack_registry()
    except ValueError:
        return None
    for pack in registry.packs:
        if pack.repo_pack == repo_pack and pack.mode == mode:
            return pack
    return None


def validate_submission_lane(repo_pack: str, mode: str) -> list[str]:
    # Evaluator-backed subnet packs are validated against the central pack
    # registry and must not depend on eval-pack or frontier-manifest state.
    evaluator_entry = find_evaluator_pack_entry(repo_pack, mode)
    if evaluator_entry is not None:
        if not evaluator_entry.active:
            return [
                "Evaluator-backed lane is not active in the pack registry: "
                f"{evaluator_entry.lane_id}"
            ]
        return []

    reasons: list[str] = []
    try:
        ensure_active_repo_pack(repo_pack)
    except (FileNotFoundError, ValueError) as exc:
        reasons.append(str(exc))
        return reasons

    try:
        resolve_eval_pack_path(repo_pack)
    except FileNotFoundError as exc:
        reasons.append(str(exc))
        return reasons

    try:
        manifest = load_frontier_manifest(repo_pack)
    except FileNotFoundError:
        reasons.append(
            "Frontier manifest does not exist for the target repo pack. "
            "Initialize the frontier before accepting PR submissions."
        )
        return reasons
    if mode not in manifest.modes:
        reasons.append(
            f"Mode is not configured in the frontier manifest: {mode}"
        )
    return reasons


def resolve_submission_descriptor(
    submission_root: Path,
    *,
    repo_root: Path | None,
    require_exists: bool = True,
) -> tuple[SubmissionDescriptor | None, list[str]]:
    reasons: list[str] = []
    root = submission_root.resolve()
    if require_exists:
        if not root.exists():
            return None, [f"Submission path does not exist: {submission_root}"]
        if not root.is_dir():
            return None, [f"Submission path must be a directory: {submission_root}"]

    if repo_root is not None:
        try:
            relative = root.relative_to(repo_root)
        except ValueError:
            return None, ["Submission path must live under the Kata repo root."]
        parts = relative.parts
    else:
        parts = root.parts
        if SUBMISSIONS_DIRNAME in parts:
            parts = parts[parts.index(SUBMISSIONS_DIRNAME) :]

    if len(parts) < 4 or parts[0] != SUBMISSIONS_DIRNAME:
        reasons.append(
            "Submission path must match "
            "`submissions/<repo-pack>/<mode>/<submission-id>`."
        )
        return None, reasons

    repo_pack = parts[1]
    mode = parts[2]
    submission_id = parts[3]
    if mode not in SUPPORTED_SUBMISSION_MODES:
        reasons.append(
            "Submission mode must be one of: "
            + ", ".join(sorted(SUPPORTED_SUBMISSION_MODES))
        )
    return (
        SubmissionDescriptor(
            root=root,
            repo_pack=repo_pack,
            mode=mode,
            submission_id=submission_id,
            agent_path=root / SUBMISSION_AGENT_FILENAME,
            agent_manifest_path=root / SUBMISSION_AGENT_MANIFEST_FILENAME,
            metadata_path=root / SUBMISSION_METADATA_FILENAME,
        ),
        reasons,
    )


def load_submission_metadata(path: Path) -> SubmissionMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Submission metadata must contain a JSON object: {path}")
    try:
        return SubmissionMetadata(
            schema_version=int(payload["schema_version"]),
            repo_pack=str(payload["repo_pack"]),
            mode=str(payload["mode"]),
            submission_id=str(payload["submission_id"]),
            created_at=str(payload["created_at"]),
            author=str(payload["author"]) if payload.get("author") is not None else None,
            title=str(payload["title"]) if payload.get("title") is not None else None,
            notes=str(payload["notes"]) if payload.get("notes") is not None else None,
        )
    except KeyError as exc:
        raise ValueError(
            f"Submission metadata is missing required field: {exc.args[0]}"
        ) from exc


def write_submission_metadata(path: Path, metadata: SubmissionMetadata) -> None:
    path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")


def validate_submission_mode(mode: str) -> None:
    if mode not in SUPPORTED_SUBMISSION_MODES:
        raise ValueError(
            "Submission mode must be one of: "
            + ", ".join(sorted(SUPPORTED_SUBMISSION_MODES))
        )


def default_submissions_root() -> Path:
    return Path.cwd().resolve() / SUBMISSIONS_DIRNAME


def default_submission_agent(mode: str) -> str:
    if mode == "miner":
        return (
            "from __future__ import annotations\n\n"
            f'\"\"\"Kata submission scaffold for the {mode} lane.\"\"\"\n\n'
            "def agent_main(\n"
            "    project_dir: str | None = None,\n"
            "    inference_api: str | None = None,\n"
            ") -> dict:\n"
            f"    # {DEFAULT_AGENT_PLACEHOLDER}\n"
            "    return {\n"
            "        \"vulnerabilities\": [],\n"
            "    }\n"
        )
    return (
        "from __future__ import annotations\n\n"
        f'\"\"\"Kata submission scaffold for the {mode} lane.\"\"\"\n\n'
        "def solve(repo_path: str, issue: str, model: str, api_base: str, api_key: str) -> dict:\n"
        f"    # {DEFAULT_AGENT_PLACEHOLDER}\n"
        "    return {\n"
        "        \"success\": False,\n"
        "        \"message\": \"scaffold agent - replace before submitting\",\n"
        "        \"diff\": \"\",\n"
        "    }\n"
    )


def default_submission_notes(mode: str) -> str:
    lines = [
        "Recommended conventions:",
        "- author: your GitHub username",
        f"- submission_id: {SUBMISSION_ID_CONVENTION}",
        "- implement a real agent in agent.py before opening the PR",
    ]
    if mode == "miner":
        lines.append("- SN60 miner submissions in V1 must stay self-contained in agent.py")
    else:
        lines.append("- optional helper modules may live under helpers/*.py")
    return "\n".join(lines) + "\n"


def submission_entrypoint_name(mode: str) -> str:
    if mode == "miner":
        return "agent_main"
    return "solve"


def required_submission_entrypoint_reason(mode: str) -> str:
    if mode == "miner":
        return "Submission agent must define agent_main(...)."
    return "Submission agent must define solve(...)."


def agent_defines_required_entrypoint(agent_source: str, mode: str) -> bool:
    pattern = re.compile(
        rf"(?m)^(?:async\s+)?def\s+{re.escape(submission_entrypoint_name(mode))}\s*\("
    )
    return pattern.search(agent_source) is not None


def infer_submission_dirs(changed_paths: list[str]) -> list[str]:
    candidate_dirs: list[str] = []
    for changed_path in changed_paths:
        parts = Path(changed_path).parts
        if len(parts) < 5 or parts[0] != SUBMISSIONS_DIRNAME:
            continue
        candidate_dir = Path(*parts[:4]).as_posix()
        if candidate_dir not in candidate_dirs:
            candidate_dirs.append(candidate_dir)
    return candidate_dirs


def read_changed_paths_file(path: str) -> list[str]:
    file_path = Path(path).expanduser().resolve()
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_changed_paths(changed_paths: list[str]) -> list[str]:
    normalized: list[str] = []
    for changed_path in changed_paths:
        value = changed_path.strip()
        if not value:
            continue
        normalized.append(value.strip("/"))
    return normalized


def dedupe(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return unique


def hash_submission_bundle(root: Path) -> str:
    bundle_root = root.expanduser().resolve()
    relative_paths = sorted(
        path for path in find_bundle_relative_paths(bundle_root)
    )
    return sha256_directory(bundle_root, include=relative_paths)


def find_bundle_relative_paths(root: Path) -> list[str]:
    relative_paths = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if not path.is_symlink()
        and path.is_file()
        and is_allowed_bundle_relative_path(path.relative_to(root).as_posix())
    ]
    return relative_paths


def find_bundle_symlink_paths(root: Path) -> list[str]:
    return [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_symlink()
    ]


def validate_bundle_python_sources(
    bundle_files: dict[str, str],
    *,
    mode: str,
) -> list[str]:
    reasons: list[str] = []
    for relative_path, content in sorted(bundle_files.items()):
        try:
            ast.parse(content, filename=relative_path)
        except SyntaxError as exc:
            line_number = exc.lineno or 1
            reasons.append(
                "Submission bundle contains invalid Python syntax in "
                f"{relative_path}:{line_number}."
            )
            continue
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".py",
                encoding="utf-8",
                delete=False,
            ) as handle:
                handle.write(content)
                temp_path = Path(handle.name)
            py_compile.compile(str(temp_path), doraise=True)
        except py_compile.PyCompileError:
            reasons.append(
                "Submission bundle failed Python compile smoke check in "
                f"{relative_path}."
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    agent_source = bundle_files.get(AGENT_ENTRY_FILENAME, "")
    if agent_source and not agent_defines_required_entrypoint(agent_source, mode):
        reasons.append(required_submission_entrypoint_reason(mode))
    return reasons


def validate_bundle_static_policy(
    bundle_files: dict[str, str],
    *,
    mode: str,
) -> list[str]:
    reasons: list[str] = []
    parsed_trees: dict[str, ast.AST] = {}
    for relative_path, content in sorted(bundle_files.items()):
        try:
            parsed_trees[relative_path] = ast.parse(content, filename=relative_path)
        except SyntaxError:
            continue
        for token in FORBIDDEN_ENV_REFERENCE_TOKENS:
            if token in content:
                reasons.append(
                    f"Submission bundle must not read validator/provider secret env vars "
                    f"directly: {relative_path} references `{token}`."
                )
        lowered = content.lower()
        for token in FORBIDDEN_PROVIDER_SUBSTRINGS:
            if token in lowered:
                reasons.append(
                    f"Submission bundle must not hardcode provider endpoints directly: "
                    f"{relative_path} references `{token}`."
                )
        if SECRET_PATTERN.search(content):
            reasons.append(
                f"Submission bundle appears to contain a hardcoded secret token: {relative_path}."
            )
    reasons.extend(validate_bundle_entrypoint_contract(parsed_trees, mode=mode))
    reasons.extend(validate_bundle_sampling_policy(parsed_trees, mode=mode))
    return reasons


def validate_bundle_entrypoint_contract(
    parsed_trees: dict[str, ast.AST],
    *,
    mode: str,
) -> list[str]:
    if mode == "miner":
        return validate_bundle_miner_contract(parsed_trees)
    return validate_bundle_solver_contract(parsed_trees)


def validate_bundle_solver_contract(parsed_trees: dict[str, ast.AST]) -> list[str]:
    agent_tree = parsed_trees.get(AGENT_ENTRY_FILENAME)
    if agent_tree is None:
        return []
    solve_fn = find_module_function_def(agent_tree, "solve")
    if solve_fn is None:
        return [required_submission_entrypoint_reason("contributor")]
    if len(solve_fn.args.args) != len(REQUIRED_SOLVE_ARGS):
        return [
            "Submission agent must keep the validator solve signature: "
            "solve(repo_path, issue, model, api_base, api_key)."
        ]
    arg_names = [arg.arg for arg in solve_fn.args.args[: len(REQUIRED_SOLVE_ARGS)]]
    if tuple(arg_names) != REQUIRED_SOLVE_ARGS:
        return [
            "Submission agent must keep the validator solve signature: "
            "solve(repo_path, issue, model, api_base, api_key)."
        ]
    if solve_fn.args.vararg is not None or solve_fn.args.kwarg is not None:
        return [
            "Submission agent must not use *args or **kwargs in solve(...)."
        ]
    return []


def validate_bundle_miner_contract(parsed_trees: dict[str, ast.AST]) -> list[str]:
    agent_tree = parsed_trees.get(AGENT_ENTRY_FILENAME)
    if agent_tree is None:
        return []
    agent_main_fn = find_module_function_def(agent_tree, "agent_main")
    if agent_main_fn is None:
        if find_module_async_function_def(agent_tree, "agent_main") is not None:
            return [
                "Submission agent_main must be a synchronous function; the SN60 "
                "sandbox runner calls agent_main() directly and does not await "
                "coroutines."
            ]
        return [required_submission_entrypoint_reason("miner")]

    positional_args = [*agent_main_fn.args.posonlyargs, *agent_main_fn.args.args]
    required_positional_args = len(positional_args) - len(agent_main_fn.args.defaults)
    if required_positional_args > 0:
        return ["Submission agent must support no-argument invocation: agent_main()."]

    required_keyword_only_args = [
        arg.arg
        for arg, default in zip(agent_main_fn.args.kwonlyargs, agent_main_fn.args.kw_defaults)
        if default is None
    ]
    if required_keyword_only_args:
        return ["Submission agent must support no-argument invocation: agent_main()."]

    for return_node in iter_non_nested_function_returns(agent_main_fn):
        if return_node.value is None or not isinstance(return_node.value, ast.Dict):
            continue
        if not dict_contains_string_key(return_node.value, "vulnerabilities"):
            return [
                "Submission agent must return a Bitsec-compatible report with "
                "top-level `vulnerabilities`."
            ]
    return []


def validate_bundle_sampling_policy(
    parsed_trees: dict[str, ast.AST],
    *,
    mode: str,
) -> list[str]:
    reasons: list[str] = []
    for relative_path, tree in sorted(parsed_trees.items()):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg in FORBIDDEN_SAMPLING_NAMES:
                    reasons.append(
                        "Submission bundle must not control model sampling parameters "
                        f"directly: {relative_path} uses `{keyword.arg}`."
                    )
        if relative_path != AGENT_ENTRY_FILENAME:
            continue
        solve_fn = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "solve"
            ),
            None,
        )
        if mode == "miner" or solve_fn is None:
            continue
        for node in ast.walk(solve_fn):
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                for target_name in targets:
                    if target_name in {"model", "api_base", "api_key"}:
                        reasons.append(
                            "Submission agent must not override validator-provided routing "
                            f"parameters inside solve(...): `{target_name}`."
                        )
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in {"model", "api_base", "api_key"}:
                    reasons.append(
                        "Submission agent must not override validator-provided routing "
                        f"parameters inside solve(...): `{node.target.id}`."
                    )
    return dedupe(reasons)


def iter_non_nested_function_returns(function_node: ast.FunctionDef):
    stack: list[ast.AST] = list(reversed(function_node.body))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            yield node
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def dict_contains_string_key(node: ast.Dict, key_name: str) -> bool:
    for key in node.keys:
        if isinstance(key, ast.Constant) and key.value == key_name:
            return True
    return False


def find_module_function_def(
    module_tree: ast.AST,
    function_name: str,
) -> ast.FunctionDef | None:
    if not isinstance(module_tree, ast.Module):
        return None
    for node in module_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    return None


def find_module_async_function_def(
    module_tree: ast.AST,
    function_name: str,
) -> ast.AsyncFunctionDef | None:
    if not isinstance(module_tree, ast.Module):
        return None
    for node in module_tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name:
            return node
    return None


def validate_submission_not_copycat(
    *,
    metadata: SubmissionMetadata,
    submission_root: Path,
    bundle_files: dict[str, str],
) -> list[str]:
    evaluator_entry = find_evaluator_pack_entry(metadata.repo_pack, metadata.mode)
    if evaluator_entry is not None:
        return validate_submission_not_copycat_of_lane_king(
            lane_id=evaluator_entry.lane_id,
            submission_root=submission_root,
        )

    try:
        manifest = load_frontier_manifest(metadata.repo_pack)
    except FileNotFoundError:
        return []
    mode_config = manifest.modes.get(metadata.mode)
    if mode_config is None:
        return []

    reasons: list[str] = []
    candidate_hash = hash_submission_bundle(submission_root)
    frontier_hash = resolve_frontier_artifact_hash(mode_config)
    if candidate_hash == frontier_hash:
        reasons.append("Submission bundle is an exact copy of the current frontier artifact.")

    candidate_agent = bundle_files.get(AGENT_ENTRY_FILENAME)
    if candidate_agent is None:
        return reasons

    frontier_agent_path = (
        resolve_artifact_path(mode_config.frontier_artifact) / AGENT_ENTRY_FILENAME
    )
    if frontier_agent_path.exists() and python_sources_equivalent(
        candidate_agent,
        frontier_agent_path.read_text(encoding="utf-8"),
    ):
        reasons.append(
            "Submission agent duplicates the current frontier agent implementation."
        )
    return reasons


def validate_submission_not_copycat_of_lane_king(
    *,
    lane_id: str,
    submission_root: Path,
) -> list[str]:
    if not lane_king_state_path(lane_id).exists():
        return []
    king = load_lane_king_state(lane_id)
    if king.current_king_artifact_hash is None:
        return []
    candidate_hash = hash_submission_bundle(submission_root)
    if candidate_hash == king.current_king_artifact_hash:
        return [
            "Submission bundle is an exact copy of the current lane king artifact."
        ]
    return []


def python_sources_equivalent(left: str, right: str) -> bool:
    try:
        left_tree = ast.parse(left)
        right_tree = ast.parse(right)
    except SyntaxError:
        return left == right
    return ast.dump(left_tree, include_attributes=False) == ast.dump(
        right_tree,
        include_attributes=False,
    )
