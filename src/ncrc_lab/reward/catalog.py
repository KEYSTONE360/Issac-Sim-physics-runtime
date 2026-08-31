from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ..config import path_for
from ..constants import EvidenceStatus
from ..experiment.parsers import parse_env


def _assigned_rewards(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return items
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        value = node.value
        name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
        if not name or "reward" not in name.lower():
            continue
        if isinstance(value, ast.Dict):
            for key_node, value_node in zip(value.keys, value.values):
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    literal = value_node.value if isinstance(value_node, ast.Constant) else None
                    items.append({"name": key_node.value, "default_weight": literal if isinstance(literal, (int, float)) else None, "active": literal is not None, "source_symbol": name})
    return items


def build_catalog(robot: str, source_root: Path | None = None, env_files: list[Path] | None = None) -> dict[str, Any]:
    source_root = source_root or path_for("ncrc_source")
    if env_files is None:
        default_env = path_for(f"presets/user/{robot}/server_env_background_default.yaml")
        env_files = [default_env] if default_env.exists() else []
    found: dict[str, dict[str, Any]] = {}
    for path in source_root.rglob("*.py") if source_root.exists() else []:
        for item in _assigned_rewards(path):
            item.update({
                "function": "UNKNOWN", "namespace": "UNKNOWN",
                "classification": "NCRC_AVAILABLE" if item["active"] else "NCRC_INACTIVE",
                "current_weight": item["default_weight"], "parameters": {}, "editable": True,
                "server_verified": False, "required_state": "UNKNOWN",
                "source_file": str(path), "source_version": "UNKNOWN",
                "description": "Source-discovered reward", "evidence_summary": [],
                "tested_values": [], "evidence_status": EvidenceStatus.VERIFIED_NCRC_SOURCE,
            })
            found[item["name"]] = item
    for env_path in env_files:
        parsed = parse_env(env_path)
        for name, term in parsed.get("reward_terms", {}).items():
            if not term.get("active"):
                continue
            weight = term["weight"]
            item = found.setdefault(name, {
                "name": name, "function": term.get("function", "UNKNOWN"), "namespace": str(term.get("function", "UNKNOWN")).rsplit(":", 1)[0],
                "default_weight": None, "parameters": term.get("parameters", {}), "editable": True,
                "required_state": "UNKNOWN", "source_file": str(env_path),
                "source_symbol": "UNKNOWN", "source_version": "UNKNOWN",
                "description": "Server environment-discovered reward", "evidence_summary": [], "tested_values": [],
            })
            item.update({"classification": "NCRC_ACTIVE", "current_weight": weight, "active": True, "server_verified": True, "evidence_status": EvidenceStatus.VERIFIED_SERVER_ENV})
        for name in parsed["inactive_rewards"]:
            term = parsed.get("reward_terms", {}).get(name, {})
            item = found.setdefault(name, {
                "name": name, "function": term.get("function", "UNKNOWN"),
                "namespace": str(term.get("function", "UNKNOWN")).rsplit(":", 1)[0],
                "default_weight": None, "parameters": term.get("parameters", {}),
                "editable": True, "required_state": "UNKNOWN", "source_file": str(env_path),
                "source_symbol": name, "source_version": "UNKNOWN", "description": "Inactive term defined in server env",
                "evidence_summary": [], "tested_values": [],
            })
            item.update({"classification": "NCRC_INACTIVE", "current_weight": None, "active": False, "server_verified": True, "evidence_status": EvidenceStatus.VERIFIED_SERVER_ENV})
    statuses = {str(item.get("evidence_status")) for item in found.values()}
    status = EvidenceStatus.VERIFIED_NCRC_SOURCE if str(EvidenceStatus.VERIFIED_NCRC_SOURCE) in statuses else EvidenceStatus.VERIFIED_SERVER_ENV if found else EvidenceStatus.UNKNOWN
    reference_path = path_for("generated/isaaclab_reward_functions.json")
    reference_functions = []
    if reference_path.exists():
        try:
            reference_functions = json.loads(reference_path.read_text(encoding="utf-8")).get("functions", [])
        except (OSError, json.JSONDecodeError):
            reference_functions = []
    term_functions = {str(item.get("function")) for item in found.values()}
    for index, function in enumerate(reference_functions):
        if function.get("qualified_name") in term_functions:
            function = dict(function)
            function["weight_editable"] = True
            function["reason"] = "Matched to a term defined in the server env snapshot."
        function["ncrc_version_match"] = "UNKNOWN"
        reference_functions[index] = function
    return {"robot": robot, "status": status, "rewards": [found[name] for name in sorted(found)], "isaaclab_reference_functions": reference_functions}


def write_catalog(robot: str, env_files: list[Path] | None = None) -> Path:
    result = build_catalog(robot, env_files=env_files)
    destination = path_for(f"generated/reward_catalog_{robot.lower()}.json")
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination
