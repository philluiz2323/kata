"""Taopedia-specific first king agent for the Kata contributor lane."""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SEED_INSTRUCTIONS = """# Kata Contributor Seed Instructions: Taopedia Articles

Repo: `taopedia-articles`
GitHub: `e35ventura/taopedia-articles`

This agent is optimized for Kata benchmark tasks in the Taopedia article repo.

## Repo Rules
- Articles live under `content/pages/<slug>/index.mdx`.
- Preserve existing front matter, heading structure, citations, and formatting unless asked.
- Keep edits narrow. Do not reformat unrelated prose.
- Sources are required for factual and technical claims.
- Prefer official docs, source code, release notes, or specifications over generic summaries.
- When docs and code disagree, implementation code is the source of truth.
- Every added section should add a new fact, distinction, caveat, source, or operational detail.

## Benchmark Strategy
- Read the task literally and edit only the target article unless another file is named.
- If the task says "fix", replace or remove the wrong statement instead of adding duplicate text.
- If the task says "improve", make the smallest complete source-backed improvement.
- If asked for a distinction, add a concise `## Distinction from ...` section.
- Return only a unified diff that applies cleanly with `git apply`.
"""

LANE_MODE = "contributor"
AGENT_LABEL = "frontier"
MAX_TARGET_BYTES = 24000
MAX_RELATED_BYTES = 8000
MAX_TOTAL_CONTEXT_BYTES = 85000
MAX_RELATED_FILES = 8
REQUEST_TIMEOUT_SECONDS = 180

PATH_PATTERN = re.compile(
    r"`([^`]+?\.(?:mdx|md|json|ya?ml|toml|txt|ts|tsx|js|jsx|py))`"
)
HEADING_PATTERN = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class TaskSpec:
    title: str
    target_paths: tuple[str, ...]
    tokens: frozenset[str]
    wants_distinction: bool


def solve(repo_path: str, issue: str, model: str, api_base: str, api_key: str) -> dict:
    if not model:
        return {"success": False, "message": "validator did not provide a model", "diff": ""}
    if not api_base:
        return {
            "success": False,
            "message": "validator did not provide an api_base",
            "diff": "",
        }

    repo_root = Path(repo_path).resolve()
    task = parse_task(issue)
    repo_context = build_repo_context(repo_root, task)
    response_text = request_diff(
        model=model,
        api_base=api_base,
        api_key=api_key,
        issue=issue,
        repo_context=repo_context,
    )
    diff_text = normalize_diff(response_text)
    if not diff_text:
        return {
            "success": False,
            "message": "model did not return a unified diff",
            "diff": "",
        }

    check_result = git_apply_check(repo_root, diff_text)
    if not check_result.ok:
        repaired = request_diff(
            model=model,
            api_base=api_base,
            api_key=api_key,
            issue=issue,
            repo_context=repo_context,
            previous_diff=diff_text,
            apply_error=check_result.error,
        )
        repaired_diff = normalize_diff(repaired)
        repaired_check = (
            git_apply_check(repo_root, repaired_diff) if repaired_diff else check_result
        )
        if repaired_diff and repaired_check.ok:
            diff_text = repaired_diff
        else:
            return {
                "success": False,
                "message": (
                    "model returned a diff that failed git apply --check: "
                    f"{check_result.error}"
                ),
                "diff": "",
            }

    return {
        "success": True,
        "message": f"{AGENT_LABEL} Taopedia king produced an applyable diff",
        "diff": diff_text,
    }


@dataclass(frozen=True)
class ApplyCheck:
    ok: bool
    error: str = ""


def parse_task(issue: str) -> TaskSpec:
    title = first_nonempty_line(issue)
    paths = tuple(
        dict.fromkeys(
            path for path in PATH_PATTERN.findall(issue) if not path.startswith(".")
        )
    )
    content_paths = tuple(path for path in paths if path.startswith("content/pages/"))
    target_paths = content_paths or paths
    raw_tokens = set(TOKEN_PATTERN.findall(issue.lower()))
    noisy = {
        "task",
        "title",
        "goal",
        "update",
        "fix",
        "improve",
        "content",
        "pages",
        "index",
        "mdx",
    }
    tokens = frozenset(token for token in raw_tokens if len(token) > 2 and token not in noisy)
    wants_distinction = "distinction" in raw_tokens or "distinguish" in raw_tokens
    return TaskSpec(
        title=title,
        target_paths=target_paths,
        tokens=tokens,
        wants_distinction=wants_distinction,
    )


def first_nonempty_line(value: str) -> str:
    for line in value.splitlines():
        stripped = line.strip("#: \t")
        if stripped:
            return stripped[:180]
    return "Kata benchmark task"


def build_repo_context(repo_root: Path, task: TaskSpec) -> str:
    sections: list[str] = []
    budget = MAX_TOTAL_CONTEXT_BYTES

    sections.append(
        "## Parsed Task\n"
        f"Title: {task.title}\n"
        f"Target paths: {', '.join(task.target_paths) if task.target_paths else '(none parsed)'}\n"
        f"Important tokens: {', '.join(sorted(task.tokens)) or '(none)'}"
    )

    for relative_path in baseline_context_paths(repo_root):
        chunk = file_section(repo_root, relative_path, MAX_RELATED_BYTES)
        if chunk and fits_budget(chunk, budget):
            sections.append(chunk)
            budget -= byte_len(chunk)

    for relative_path in task.target_paths:
        chunk = file_section(repo_root, relative_path, MAX_TARGET_BYTES)
        if chunk and fits_budget(chunk, budget):
            sections.append(chunk)
            budget -= byte_len(chunk)

    for relative_path in related_article_paths(repo_root, task):
        if relative_path in task.target_paths:
            continue
        chunk = file_section(repo_root, relative_path, MAX_RELATED_BYTES)
        if chunk and fits_budget(chunk, budget):
            sections.append(chunk)
            budget -= byte_len(chunk)

    content_index = content_pages_index(repo_root, task.target_paths)
    if content_index:
        sections.append(content_index)

    return "\n\n".join(sections)


def baseline_context_paths(repo_root: Path) -> list[str]:
    candidates = ["CONTRIBUTING.md", "README.md", "package.json"]
    return [path for path in candidates if (repo_root / path).is_file()]


def related_article_paths(repo_root: Path, task: TaskSpec) -> list[str]:
    content_root = repo_root / "content" / "pages"
    if not content_root.is_dir():
        return []

    scored: list[tuple[int, str]] = []
    for path in content_root.glob("*/index.mdx"):
        relative_path = path.relative_to(repo_root).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        haystack = f"{relative_path}\n{extract_headings(content)}".lower()
        score = sum(1 for token in task.tokens if token in haystack)
        if task.wants_distinction and "## distinction from " in content.lower():
            score += 4
        if score > 0:
            scored.append((score, relative_path))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [relative_path for _, relative_path in scored[:MAX_RELATED_FILES]]


def content_pages_index(repo_root: Path, target_paths: tuple[str, ...]) -> str:
    content_root = repo_root / "content" / "pages"
    if not content_root.is_dir():
        return ""
    names = sorted(path.parent.name for path in content_root.glob("*/index.mdx"))
    target_names = {
        Path(path).parent.name
        for path in target_paths
        if path.startswith("content/pages/")
    }
    nearby = [name for name in names if name in target_names]
    remaining = [name for name in names if name not in target_names]
    selected = nearby + remaining[:160]
    if len(names) > len(selected):
        selected.append(f"... {len(names) - len(selected)} more")
    return "## Article Slug Index\n" + "\n".join(selected)


def extract_headings(content: str) -> str:
    return "\n".join(match.group(1) for match in HEADING_PATTERN.finditer(content))


def file_section(repo_root: Path, relative_path: str, max_bytes: int) -> str:
    path = repo_root / relative_path
    if not path.is_file():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    encoded = content.encode("utf-8")
    truncated = ""
    if len(encoded) > max_bytes:
        content = encoded[:max_bytes].decode("utf-8", errors="ignore")
        truncated = "\n...[truncated]"
    return f"### FILE: {relative_path}\n```\n{content.rstrip()}{truncated}\n```"


def fits_budget(value: str, remaining: int) -> bool:
    return byte_len(value) <= remaining


def byte_len(value: str) -> int:
    return len(value.encode("utf-8"))


def request_diff(
    *,
    model: str,
    api_base: str,
    api_key: str,
    issue: str,
    repo_context: str,
    previous_diff: str = "",
    apply_error: str = "",
) -> str:
    system_prompt = (
        "You are the current first king agent for Kata's Taopedia contributor lane.\n"
        "Solve the task from the task description and repository context only. "
        "Do not rely on hidden oracle files, test fixtures, or external private metadata.\n"
        "Return only a unified diff that can be applied with git apply. "
        "Do not return prose, markdown fences, or explanations.\n\n"
        "Repo-specific instructions:\n"
        f"{SEED_INSTRUCTIONS}"
    )
    repair_context = ""
    if previous_diff or apply_error:
        repair_context = (
            "\n\nThe previous diff failed `git apply --check`.\n"
            f"Apply error:\n{apply_error.strip()}\n\n"
            f"Previous diff:\n{previous_diff.strip()}\n\n"
            "Return a corrected unified diff only."
        )
    user_prompt = (
        f"Lane mode: {LANE_MODE}\n\n"
        "Task:\n"
        f"{issue.strip()}\n\n"
        f"{repo_context}"
        f"{repair_context}\n\n"
        "Output requirement: return only the final unified diff."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 4000,
    }
    request = urllib.request.Request(
        build_chat_completions_url(api_base),
        data=json.dumps(payload).encode("utf-8"),
        headers=build_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat completion request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"chat completion request failed: {exc.reason}") from exc
    return extract_message_content(response_payload)


def build_chat_completions_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def build_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_message_content(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def normalize_diff(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    diff_start = text.find("diff --git ")
    patch_start = text.find("--- ")
    starts = [index for index in (diff_start, patch_start) if index >= 0]
    if starts:
        text = text[min(starts) :].strip()
    if text.startswith("diff --git") or text.startswith("--- "):
        return text + "\n"
    return ""


def git_apply_check(repo_root: Path, diff_text: str) -> ApplyCheck:
    if not diff_text:
        return ApplyCheck(ok=False, error="empty diff")
    completed = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=str(repo_root),
        input=diff_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return ApplyCheck(ok=True)
    error = (completed.stderr or completed.stdout or "unknown git apply error").strip()
    return ApplyCheck(ok=False, error=error[:2000])
