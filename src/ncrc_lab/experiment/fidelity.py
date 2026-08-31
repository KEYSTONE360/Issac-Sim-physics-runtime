from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import path_for
from ..environment import _remove_reward_sections, load_background_defaults
from ..source.scanner import sha256_file


CALCULATION_POLICY_VERSION = 1


def canonical_background(value: Any) -> bytes:
    """Canonical representation without rewards; no numeric rounding or coercion."""
    background = _remove_reward_sections(value)
    return json.dumps(
        background, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def background_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_background(value)).hexdigest()


def calculation_policy(robot: str) -> dict[str, Any]:
    snapshot = load_background_defaults(robot)
    background = snapshot.get("background_defaults", {})
    raw_path_text = snapshot.get("raw_snapshot")
    raw_path = Path(raw_path_text) if raw_path_text else None
    raw_exists = bool(raw_path and raw_path.exists())
    recorded_hash = snapshot.get("source_sha256")
    actual_hash = sha256_file(raw_path) if raw_exists and raw_path else None
    return {
        "version": CALCULATION_POLICY_VERSION,
        "mode": "FULL_FIDELITY",
        "reward_formula_execution": "OFFICIAL_SERVER_NATIVE_IMPLEMENTATION",
        "reward_formula_modification_allowed": False,
        "numeric_weight_override_only": True,
        "formula_simplification_allowed": False,
        "term_omission_allowed": False,
        "proxy_metric_substitution_allowed": False,
        "analysis_rounding_allowed": False,
        "missing_state_policy": "UNKNOWN_NOT_APPROXIMATED",
        "training_precision": "OFFICIAL_SERVER_NATIVE_PRECISION",
        "analysis_precision": "SOURCE_PRECISION_PRESERVED_FLOAT64_OR_DECIMAL",
        "eta_calibration_mutates_experiment": False,
        "background_snapshot_status": snapshot.get("status", "NOT_AVAILABLE"),
        "background_source_sha256": recorded_hash,
        "background_actual_sha256": actual_hash,
        "background_hash_valid": bool(recorded_hash and actual_hash == recorded_hash),
        "background_fingerprint": background_fingerprint(background) if background else None,
        "server_formula_version_match": "UNKNOWN",
        "server_ready": bool(recorded_hash and actual_hash == recorded_hash and background),
    }


def write_calculation_policy(robot: str) -> Path:
    payload = calculation_policy(robot)
    destination = path_for(f"generated/calculation_policy_{robot.lower()}.json")
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination

