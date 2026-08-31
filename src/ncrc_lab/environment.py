from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import path_for
from .constants import EvidenceStatus
from .experiment.parsers import parse_yaml_subset
from .source.scanner import sha256_file


def _remove_reward_sections(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_reward_sections(child)
            for key, child in value.items()
            if str(key).lower() not in {"reward", "rewards"}
        }
    if isinstance(value, list):
        return [_remove_reward_sections(item) for item in value]
    return value


def snapshot_background_defaults(env_path: Path, robot: str) -> dict[str, Any]:
    """Preserve an env snapshot and make every non-reward field the default.

    The raw YAML remains immutable provenance. A JSON projection excludes only
    mappings named reward/rewards; no values are synthesized.
    """
    if robot not in {"H1", "Go2"}:
        raise ValueError("robot must be H1 or Go2")
    env_path = env_path.resolve()
    data = parse_yaml_subset(env_path)
    background = _remove_reward_sections(data)
    digest = sha256_file(env_path)
    preset_dir = path_for(f"presets/user/{robot}")
    preset_dir.mkdir(parents=True, exist_ok=True)
    raw_target = preset_dir / "server_env_background_default.yaml"
    shutil.copy2(env_path, raw_target)
    payload = {
        "robot": robot,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": EvidenceStatus.VERIFIED_SERVER_ENV,
        "source_path": str(env_path),
        "source_sha256": digest,
        "raw_snapshot": str(raw_target),
        "reward_sections_excluded": True,
        "default_policy": "ALL_NON_REWARD_VALUES_FROM_SERVER_ENV_SNAPSHOT",
        "background_defaults": background,
    }
    output = path_for(f"generated/background_defaults_{robot.lower()}.json")
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": str(EvidenceStatus.VERIFIED_SERVER_ENV), "raw_snapshot": str(raw_target), "defaults": str(output), "source_sha256": digest}


def load_background_defaults(robot: str) -> dict[str, Any]:
    path = path_for(f"generated/background_defaults_{robot.lower()}.json")
    if not path.exists():
        return {"status": "NOT_AVAILABLE", "background_defaults": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    # A portable build must never depend on the original development-drive
    # absolute path recorded as provenance. Prefer the immutable snapshot
    # shipped beside the executable while preserving source_path separately.
    portable_snapshot = path_for(f"presets/user/{robot}/server_env_background_default.yaml")
    if portable_snapshot.exists():
        payload["raw_snapshot"] = str(portable_snapshot.resolve())
    return payload


def _path_differences(expected: Any, actual: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Return every exact structural/value difference without numeric tolerances."""
    differences: list[dict[str, Any]] = []
    if type(expected) is not type(actual):
        return [{"path": prefix or "$", "expected": expected, "actual": actual, "kind": "TYPE"}]
    if isinstance(expected, dict):
        for key in sorted(expected.keys() | actual.keys(), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append({"path": path, "expected": "<MISSING>", "actual": actual[key], "kind": "ADDED"})
            elif key not in actual:
                differences.append({"path": path, "expected": expected[key], "actual": "<MISSING>", "kind": "REMOVED"})
            else:
                differences.extend(_path_differences(expected[key], actual[key], path))
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            differences.append({"path": prefix or "$", "expected_length": len(expected), "actual_length": len(actual), "kind": "LENGTH"})
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(_path_differences(left, right, f"{prefix}[{index}]"))
    elif expected != actual:
        differences.append({"path": prefix or "$", "expected": expected, "actual": actual, "kind": "VALUE"})
    return differences


def _reward_mapping(data: dict[str, Any]) -> dict[str, Any]:
    rewards = data.get("rewards")
    if not isinstance(rewards, dict):
        raise ValueError("env.yaml must contain a top-level rewards mapping")
    return rewards


def _validate_numeric_tree(value: Any, path: str) -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_validate_numeric_tree(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_validate_numeric_tree(child, f"{path}[{index}]"))
    elif value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
        errors.append(path)
    return errors


def _validate_reward_params(before: Any, after: Any, path: str) -> list[dict[str, Any]]:
    """Permit finite numeric leaf changes while locking every structural/non-numeric leaf."""
    if type(before) is not type(after):
        # int/float are a single numeric class for editable parameter purposes.
        if isinstance(before, (int, float)) and not isinstance(before, bool) and isinstance(after, (int, float)) and not isinstance(after, bool):
            return [] if math.isfinite(float(after)) else [{"path": path, "kind": "NON_FINITE"}]
        return [{"path": path, "kind": "PARAMETER_TYPE_CHANGED"}]
    if isinstance(before, dict):
        if set(before) != set(after):
            return [{"path": path, "kind": "PARAMETER_SCHEMA_CHANGED"}]
        errors: list[dict[str, Any]] = []
        for key in before:
            errors.extend(_validate_reward_params(before[key], after[key], f"{path}.{key}"))
        return errors
    if isinstance(before, list):
        if len(before) != len(after):
            return [{"path": path, "kind": "PARAMETER_SCHEMA_CHANGED"}]
        errors = []
        for index, (left, right) in enumerate(zip(before, after)):
            errors.extend(_validate_reward_params(left, right, f"{path}[{index}]"))
        return errors
    if isinstance(before, (int, float)) and not isinstance(before, bool):
        return [] if math.isfinite(float(after)) else [{"path": path, "kind": "NON_FINITE"}]
    return [] if before == after else [{"path": path, "kind": "NON_NUMERIC_PARAMETER_CHANGED"}]


def validate_reward_only_env(env_path: Path, robot: str = "H1") -> dict[str, Any]:
    """Validate that an input changes only existing reward weights/parameters.

    All non-reward values are compared exactly with the immutable first-server
    snapshot. Reward names and function bindings are also immutable. Numeric
    values below ``weight`` and existing ``params`` leaves are editable.
    """
    snapshot = load_background_defaults(robot)
    raw = snapshot.get("raw_snapshot")
    if not raw or not Path(raw).exists():
        raise FileNotFoundError(f"{robot} background snapshot is unavailable")
    baseline = parse_yaml_subset(Path(raw))
    candidate = parse_yaml_subset(env_path.resolve())
    background_differences = _path_differences(
        _remove_reward_sections(baseline), _remove_reward_sections(candidate)
    )
    baseline_rewards = _reward_mapping(baseline)
    candidate_rewards = _reward_mapping(candidate)
    reward_errors: list[dict[str, Any]] = []
    baseline_names, candidate_names = set(baseline_rewards), set(candidate_rewards)
    for name in sorted(candidate_names - baseline_names):
        reward_errors.append({"path": f"rewards.{name}", "kind": "NEW_REWARD_NAME"})
    for name in sorted(baseline_names - candidate_names):
        reward_errors.append({"path": f"rewards.{name}", "kind": "REMOVED_REWARD_NAME"})
    editable: list[str] = []
    for name in sorted(baseline_names & candidate_names):
        before, after = baseline_rewards[name], candidate_rewards[name]
        if before is None:
            if after is not None:
                reward_errors.append({
                    "path": f"rewards.{name}", "kind": "INACTIVE_TERM_HAS_NO_SERIALIZED_FUNCTION",
                    "detail": "activation requires the exact NCRC base config; no function is guessed",
                })
            continue
        if not isinstance(before, dict) or not isinstance(after, dict):
            reward_errors.append({"path": f"rewards.{name}", "kind": "TERM_STRUCTURE_CHANGED"})
            continue
        if before.get("func") != after.get("func"):
            reward_errors.append({"path": f"rewards.{name}.func", "kind": "FUNCTION_CHANGED"})
        allowed_keys = {"weight", "params"}
        immutable_before = {key: val for key, val in before.items() if key not in allowed_keys}
        immutable_after = {key: val for key, val in after.items() if key not in allowed_keys}
        reward_errors.extend(_path_differences(immutable_before, immutable_after, f"rewards.{name}"))
        for bad_path in _validate_numeric_tree(after.get("weight"), f"rewards.{name}.weight"):
            reward_errors.append({"path": bad_path, "kind": "NON_FINITE_OR_NON_NUMERIC"})
        reward_errors.extend(_validate_reward_params(before.get("params", {}), after.get("params", {}), f"rewards.{name}.params"))
        editable.append(name)
    valid = not background_differences and not reward_errors
    return {
        "valid": valid,
        "policy": "FIRST_PROVIDED_ENV_BACKGROUND_LOCKED_REWARD_VALUES_ONLY",
        "baseline_path": str(raw),
        "baseline_sha256": snapshot.get("source_sha256"),
        "candidate_path": str(env_path.resolve()),
        "background_locked": not background_differences,
        "background_differences": background_differences,
        "reward_schema_valid": not reward_errors,
        "reward_errors": reward_errors,
        "editable_reward_terms": editable,
    }
