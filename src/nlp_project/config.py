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
    label_column: str
    id_column: str | None
    source: str
    provenance: str
    domain: str
    challenges: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    type: str
    test_size: float
    max_features: int
    ngram_range: tuple[int, int]
    class_weight: str | None
    artifact_path: Path


@dataclass(frozen=True)
class TeamMember:
    name: str
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
    team: TeamConfig
    reports_dir: Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root using common project markers."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / "AGENTS.md").exists():
            return candidate
    if current.name == "config":
        return current.parent
    return current


def ensure_default_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
    """Create config/project_config.yaml from the example when it is absent."""

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
    """Load and validate project configuration."""

    path = ensure_default_config(config_path)
    repo_root = find_repo_root(path.parent)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a YAML mapping.")

    missing_sections = [
        section for section in ("project", "proposal", "data", "model", "team") if section not in raw
    ]
    if missing_sections:
        joined = ", ".join(missing_sections)
        raise ConfigError(f"Missing required top-level config section(s): {joined}")

    project_raw = _mapping(raw["project"], "project")
    proposal_raw = _mapping(raw["proposal"], "proposal")
    data_raw = _mapping(raw["data"], "data")
    model_raw = _mapping(raw["model"], "model")
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
        label_column=_required_str(data_raw, "label_column", "data"),
        id_column=_optional_str(data_raw.get("id_column")),
        source=_required_str(data_raw, "source", "data"),
        provenance=_required_str(data_raw, "provenance", "data"),
        domain=_required_str(data_raw, "domain", "data"),
        challenges=_required_str_tuple(data_raw, "challenges", "data"),
    )

    ngram_range = _required_int_tuple(model_raw, "ngram_range", "model", length=2)
    if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
        raise ConfigError("model.ngram_range must be a valid [min_n, max_n] pair.")

    test_size = _required_float(model_raw, "test_size", "model")
    if not 0 < test_size < 1:
        raise ConfigError("model.test_size must be between 0 and 1.")

    class_weight = _optional_str(model_raw.get("class_weight"))
    if class_weight is not None and class_weight.lower() in {"none", "null"}:
        class_weight = None

    model = ModelConfig(
        type=_required_str(model_raw, "type", "model"),
        test_size=test_size,
        max_features=_required_int(model_raw, "max_features", "model"),
        ngram_range=ngram_range,
        class_weight=class_weight,
        artifact_path=_normalize_path(model_raw.get("artifact_path", "artifacts/model.joblib"), repo_root),
    )
    if model.max_features <= 0:
        raise ConfigError("model.max_features must be positive.")

    team = TeamConfig(members=_parse_team_members(team_raw))
    reports_dir = _normalize_path(reports_raw.get("directory", "reports"), repo_root)

    app_config = AppConfig(
        repo_root=repo_root,
        config_path=path,
        project=project,
        proposal=proposal,
        data=data,
        model=model,
        team=team,
        reports_dir=reports_dir,
    )

    if warn_placeholders:
        _warn_for_placeholders(raw)

    return app_config


def _mapping(value: Any, section: str) -> dict[str, Any]:
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


def _required_float(raw: dict[str, Any], key: str, section: str) -> float:
    value = raw.get(key)
    if value is None:
        raise ConfigError(f"Missing required config value: {section}.{key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{section}.{key} must be a number.") from exc


def _required_str_tuple(raw: dict[str, Any], key: str, section: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{section}.{key} must be a non-empty list.")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if not result:
        raise ConfigError(f"{section}.{key} must contain at least one non-empty value.")
    return result


def _required_int_tuple(
    raw: dict[str, Any], key: str, section: str, *, length: int
) -> tuple[int, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ConfigError(f"{section}.{key} must be a list with {length} integers.")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{section}.{key} must contain integers.") from exc


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
        "Team Member",
        "Explain ",
        "Describe ",
        "RQ1: Replace",
        "RQ2: Replace",
        "not a final research dataset",
        "sample dataset",
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
