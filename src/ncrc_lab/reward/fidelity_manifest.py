from __future__ import annotations

import ast
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..config import path_for
from ..environment import load_background_defaults
from ..experiment.parsers import parse_env
from ..source.scanner import sha256_file
from .catalog import build_catalog


def _symbol_source(path: Path, name: str) -> tuple[str | None, str | None]:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            if segment:
                return segment, hashlib.sha256(segment.encode("utf-8")).hexdigest()
    return None, None


def build_isaac_fidelity_manifest(robot: str, env_path: Path | None = None) -> dict[str, Any]:
    reference_path = path_for("generated/isaaclab_reward_functions.json")
    if not reference_path.exists():
        raise FileNotFoundError("Isaac Lab reference is not synced")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    source_dir = Path(reference["source_dir"])
    functions = {item["qualified_name"]: item for item in reference["functions"]}
    input_env_hash = None
    if env_path is not None:
        env_path = env_path.resolve()
        parsed = parse_env(env_path)
        if parsed.get("status") != "PARSED":
            raise ValueError("env.yaml could not be parsed")
        reward_items = []
        for name, term in parsed.get("reward_terms", {}).items():
            reward_items.append({
                "name": name, "function": term.get("function", "UNKNOWN"),
                "current_weight": term.get("weight"), "active": term.get("active", False),
                "parameters": term.get("parameters", {}),
                "evidence_status": "VERIFIED_SERVER_ENV",
            })
        environment_data = parsed["data"]
        input_env_hash = sha256_file(env_path)
    else:
        catalog = build_catalog(robot)
        reward_items = catalog["rewards"]
        background = load_background_defaults(robot)
        environment_data = background.get("background_defaults", {})
    terms = []
    matched = 0
    active_count = 0
    active_matched = 0
    for term in reward_items:
        qualified = str(term.get("function", "UNKNOWN"))
        ref = functions.get(qualified)
        entry = {
            "term_name": term["name"], "function": qualified,
            "weight": term.get("current_weight"), "active": term.get("active", False),
            "parameters": term.get("parameters", {}),
            "server_env_status": term.get("evidence_status"),
            "source_match": bool(ref), "source_commit": reference["commit"],
            "source_ref": reference.get("ref_requested"),
        }
        if ref:
            matched += 1
            if term.get("active"):
                active_matched += 1
            local_source = source_dir / ref["source_file"]
            segment, segment_hash = _symbol_source(local_source, ref["name"])
            entry.update({
                "source_file": ref["source_file"], "source_file_sha256": next(
                    (f["sha256"] for f in reference["files"] if f["path"] == ref["source_file"]), None
                ),
                "formula_source_sha256": segment_hash,
                "formula_source_available": segment is not None,
                "formula_source": segment,
                "verification_status": "VERIFIED_NVIDIA_COMPATIBLE_REFERENCE",
                "ncrc_exact_source_match": "UNKNOWN",
            })
        else:
            entry.update({"verification_status": "UNKNOWN", "ncrc_exact_source_match": "UNKNOWN"})
        if term.get("active"):
            active_count += 1
        terms.append(entry)
    sim = environment_data.get("sim", {})
    sim_dt = Decimal(str(sim.get("dt"))) if sim.get("dt") is not None else None
    decimation_value = environment_data.get("decimation")
    decimation = Decimal(str(decimation_value)) if decimation_value is not None else None
    env_dt = sim_dt * decimation if sim_dt is not None and decimation is not None else None
    manager_relative = "source/isaaclab/isaaclab/managers/reward_manager.py"
    manager_file = next((f for f in reference["files"] if f["path"] == manager_relative), None)
    return {
        "robot": robot, "calculation_target": "NCRC_ISAAC_SIM_5_1",
        "input_env_path": str(env_path) if env_path else None,
        "input_env_sha256": input_env_hash,
        "input_env_format": "ISAAC_LAB_MANAGER_BASED_ENV_YAML",
        "isaac_sim_version": "5.1.0", "isaac_lab_compatible_ref": reference.get("ref_requested"),
        "isaac_lab_commit": reference["commit"], "ncrc_exact_isaac_lab_commit": "UNKNOWN",
        "reward_manager_rule": "value = func(env, **params) * weight * dt; ordered float32 accumulation",
        "reward_manager_source": manager_relative,
        "reward_manager_sha256": manager_file.get("sha256") if manager_file else None,
        "sim_dt_literal": str(sim_dt) if sim_dt is not None else None,
        "decimation_literal": str(decimation) if decimation is not None else None,
        "environment_dt_literal": str(env_dt) if env_dt is not None else None,
        "term_count": len(terms), "matched_term_count": matched,
        "active_term_count": active_count, "active_matched_term_count": active_matched,
        "source_coverage": matched / len(terms) if terms else 0.0,
        "active_source_coverage": active_matched / active_count if active_count else 0.0,
        "terms": terms,
        "reward_weights": {term["term_name"]: term["weight"] for term in terms if term.get("active")},
        "full_formula_replay_ready": bool(active_count and active_matched == active_count and env_dt is not None and manager_file),
    }


def write_isaac_fidelity_manifest(robot: str, env_path: Path | None = None) -> Path:
    payload = build_isaac_fidelity_manifest(robot, env_path)
    suffix = f"_{payload['input_env_sha256'][:12]}" if payload.get("input_env_sha256") else ""
    destination = path_for(f"generated/isaac_fidelity_{robot.lower()}{suffix}.json")
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination
