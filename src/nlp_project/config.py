"""Typed YAML configuration loader for the NewsDigest summarisation project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil
import warnings

import yaml


DEFAULT_CONFIG_PATH = Path("config/project_config.yaml")
EXAMPLE_CONFIG_PATH = Path("config/project_config.example.yaml")


class ConfigError(ValueError):
    """Raised when project configuration is missing or invalid."""


class PlaceholderConfigWarning(UserWarning):
    """Warning emitted when configurable placeholders remain in the project config."""


@dataclass(frozen=True)
class ProjectConfig:
    title: str
    task: str
    domain: str
    application_area: str
    github_repo_url: str
    random_seed: int


@dataclass(frozen=True)
class ProposalConfig:
    motivation: str
    problem_statement: str
    expected_product: str
    process_improvement: str
    research_questions: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    path: Path
    text_column: str
    summary_column: str
    id_column: str | None
    source_column: str | None
    source: str
    provenance: str
    domain: str
    challenges: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    baseline: str
    extractive: str
    abstractive: str
    max_input_tokens: int
    min_summary_tokens: int
    max_summary_tokens: int
    num_sentences_extractive: int
    artifact_dir: Path
    use_abstractive: bool


@dataclass(frozen=True)
class EvaluationConfig:
    rouge_variants: tuple[str, ...]
    max_examples: int


@dataclass(frozen=True)
class ScrapeConfig:
    user_agent: str
    request_timeout_seconds: int


@dataclass(frozen=True)
class TeamMember:
    name: str
    email: str
    responsibilities: tuple[str, ...]


@dataclass(frozen=True)
class TeamConfig:
    members: tuple[TeamMember, ...]


@dataclass(frozen=True)
class AppConfig:
    repo_root: Path
    config_path: Path
    project: ProjectConfig
    proposal: ProposalConfig
    data: DataConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    scrape: ScrapeConfig
    team: TeamConfig
    reports_dir: Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / "AGENTS.md").exists():
            return candidate
    if current.name == "config":
        return current.parent
    return current


def ensure_default_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.exists():
        return path

    root = find_repo_root(path.parent)
    example = root / EXAMPLE_CONFIG_PATH
    if not example.exists():
        raise ConfigError(f"Config file not found: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(example, path)
    return path


def load_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    warn_placeholders: bool = True,
) -> AppConfig:
    path = ensure_default_config(config_path)
    repo_root = find_repo_root(path.parent)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a YAML mapping.")

    required = ("project", "proposal", "data", "model", "team")
    missing = [section for section in required if section not in raw]
    if missing:
        raise ConfigError(
            f"Missing required top-level config section(s): {', '.join(missing)}"
        )

    project_raw = _mapping(raw["project"], "project")
    proposal_raw = _mapping(raw["proposal"], "proposal")
    data_raw = _mapping(raw["data"], "data")
    model_raw = _mapping(raw["model"], "model")
    eval_raw = _mapping(raw.get("evaluation", {}), "evaluation")
    scrape_raw = _mapping(raw.get("scrape", {}), "scrape")
    team_raw = _mapping(raw["team"], "team")
    reports_raw = _mapping(raw.get("reports", {}), "reports")

    project = ProjectConfig(
        title=_required_str(project_raw, "title", "project"),
        task=_required_str(project_raw, "task", "project"),
        domain=_required_str(project_raw, "domain", "project"),
        application_area=_required_str(project_raw, "application_area", "project"),
        github_repo_url=_required_str(project_raw, "github_repo_url", "project"),
        random_seed=_required_int(project_raw, "random_seed", "project"),
    )

    proposal = ProposalConfig(
        motivation=_required_str(proposal_raw, "motivation", "proposal"),
        problem_statement=_required_str(proposal_raw, "problem_statement", "proposal"),
        expected_product=_required_str(proposal_raw, "expected_product", "proposal"),
        process_improvement=_required_str(
            proposal_raw, "process_improvement", "proposal"
        ),
        research_questions=_required_str_tuple(
            proposal_raw, "research_questions", "proposal"
        ),
    )

    data = DataConfig(
        path=_normalize_path(_required_str(data_raw, "path", "data"), repo_root),
        text_column=_required_str(data_raw, "text_column", "data"),
        summary_column=_required_str(data_raw, "summary_column", "data"),
        id_column=_optional_str(data_raw.get("id_column")),
        source_column=_optional_str(data_raw.get("source_column")),
        source=_required_str(data_raw, "source", "data"),
        provenance=_required_str(data_raw, "provenance", "data"),
        domain=_required_str(data_raw, "domain", "data"),
        challenges=_required_str_tuple(data_raw, "challenges", "data"),
    )

    model = ModelConfig(
        baseline=_required_str(model_raw, "baseline", "model"),
        extractive=_required_str(model_raw, "extractive", "model"),
        abstractive=_required_str(model_raw, "abstractive", "model"),
        max_input_tokens=_required_int(model_raw, "max_input_tokens", "model"),
        min_summary_tokens=_required_int(model_raw, "min_summary_tokens", "model"),
        max_summary_tokens=_required_int(model_raw, "max_summary_tokens", "model"),
        num_sentences_extractive=_required_int(
            model_raw, "num_sentences_extractive", "model"
        ),
        artifact_dir=_normalize_path(model_raw.get("artifact_dir", "artifacts"), repo_root),
        use_abstractive=bool(model_raw.get("use_abstractive", False)),
    )

    evaluation = EvaluationConfig(
        rouge_variants=_required_str_tuple(
            eval_raw or {"rouge_variants": ["rouge1", "rouge2", "rougeL"]},
            "rouge_variants",
            "evaluation",
        )
        if eval_raw.get("rouge_variants")
        else ("rouge1", "rouge2", "rougeL", "rougeLsum"),
        max_examples=int(eval_raw.get("max_examples", 50)),
    )

    scrape = ScrapeConfig(
        user_agent=str(scrape_raw.get("user_agent", "NewsDigest/0.1")),
        request_timeout_seconds=int(scrape_raw.get("request_timeout_seconds", 15)),
    )

    team = TeamConfig(members=_parse_team_members(team_raw))
    reports_dir = _normalize_path(reports_raw.get("directory", "reports"), repo_root)

    app_config = AppConfig(
        repo_root=repo_root,
        config_path=path,
        project=project,
        proposal=proposal,
        data=data,
        model=model,
        evaluation=evaluation,
        scrape=scrape,
        team=team,
        reports_dir=reports_dir,
    )

    if warn_placeholders:
        _warn_for_placeholders(raw)

    return app_config


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be a mapping.")
    return value


def _required_str(raw: dict[str, Any], key: str, section: str) -> str:
    value = raw.get(key)
    if value is None or str(value).strip() == "":
        raise ConfigError(f"Missing required config value: {section}.{key}")
    return str(value).strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_int(raw: dict[str, Any], key: str, section: str) -> int:
    value = raw.get(key)
    if value is None:
        raise ConfigError(f"Missing required config value: {section}.{key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{section}.{key} must be an integer.") from exc


def _required_str_tuple(raw: dict[str, Any], key: str, section: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{section}.{key} must be a non-empty list.")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if not result:
        raise ConfigError(f"{section}.{key} must contain at least one non-empty value.")
    return result


def _parse_team_members(team_raw: dict[str, Any]) -> tuple[TeamMember, ...]:
    members_raw = team_raw.get("members")
    if not isinstance(members_raw, list) or not members_raw:
        raise ConfigError("team.members must be a non-empty list.")

    members: list[TeamMember] = []
    for index, member_raw in enumerate(members_raw, start=1):
        member = _mapping(member_raw, f"team.members[{index}]")
        members.append(
            TeamMember(
                name=_required_str(member, "name", f"team.members[{index}]"),
                email=str(member.get("email", "")).strip(),
                responsibilities=_required_str_tuple(
                    member, "responsibilities", f"team.members[{index}]"
                ),
            )
        )
    return tuple(members)


def _normalize_path(value: Any, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _warn_for_placeholders(raw: dict[str, Any]) -> None:
    placeholders = list(_iter_placeholder_paths(raw))
    if not placeholders:
        return
    preview = ", ".join(placeholders[:8])
    extra = "" if len(placeholders) <= 8 else f", and {len(placeholders) - 8} more"
    warnings.warn(
        "Project configuration still contains placeholder values: "
        f"{preview}{extra}. Replace them before final submission.",
        PlaceholderConfigWarning,
        stacklevel=3,
    )


def _iter_placeholder_paths(value: Any, prefix: str = "") -> list[str]:
    markers = (
        "replace-with",
        "TODO",
        "PLACEHOLDER",
        "<your-",
    )
    matches: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            matches.extend(_iter_placeholder_paths(item, child_prefix))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            matches.extend(_iter_placeholder_paths(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and any(marker in value for marker in markers):
        matches.append(prefix)
    return matches
