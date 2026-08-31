from __future__ import annotations

import json
import math
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import path_for
from ..database import Database, utc_now
from ..recommendation.engine import repeat_plan
from ..reward.catalog import build_catalog
from ..submission.patch import reward_weights_snippet
from .analysis import confounding_classification
from .fidelity import calculation_policy, write_calculation_policy


def parse_numeric_literal(text: str) -> tuple[str, float]:
    literal = text.strip()
    try:
        decimal = Decimal(literal)
    except InvalidOperation as exc:
        raise ValueError(f"invalid numeric value: {text}") from exc
    if not decimal.is_finite():
        raise ValueError("reward weight must be finite")
    value = float(decimal)
    if not math.isfinite(value):
        raise ValueError("reward weight exceeds server numeric range")
    return literal, value


def create_experiment_plan(
    parent_id: str,
    changes: dict[str, str],
    hypothesis: str,
    success_criteria: str,
    failure_criteria: str,
    reason: str = "",
    stage: str = "exploration",
    database: Database | None = None,
) -> dict[str, Any]:
    db = database or Database(); db.initialize()
    parent = db.get_experiment(parent_id)
    if not parent:
        raise KeyError(f"parent experiment not found: {parent_id}")
    robot = parent["robot"]
    catalog = build_catalog(robot)
    allowed = {item["name"]: item for item in catalog["rewards"] if item.get("editable")}
    baseline = dict(parent.get("reward_weights") or {})
    candidate = dict(baseline)
    literals = {name: repr(value) for name, value in baseline.items()}
    effective_changes: dict[str, dict[str, Any]] = {}
    for name, text in changes.items():
        if name not in allowed:
            raise ValueError(f"reward is not defined/editable in the current environment: {name}")
        literal, value = parse_numeric_literal(text)
        old = baseline.get(name)
        if old == value:
            continue
        candidate[name] = value
        literals[name] = literal
        effective_changes[name] = {
            "from": old, "to": value, "to_literal": literal,
            "classification": allowed[name].get("classification"),
            "function": allowed[name].get("function", "UNKNOWN"),
        }
    if not effective_changes:
        raise ValueError("at least one reward value must actually change")

    fidelity = calculation_policy(robot)
    if not fidelity["server_ready"]:
        raise ValueError("FULL_FIDELITY background snapshot is unavailable or its hash changed")
    confounding = confounding_classification(len(effective_changes))
    repeats = repeat_plan(stage, confounded=len(effective_changes) > 1)
    experiment_id = db.next_experiment_id(robot)
    plan_dir = path_for(f"runs/{robot}/{experiment_id}")
    plan_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "experiment_id": experiment_id, "robot": robot, "parent_experiment": parent_id,
        "run_type": "PLANNED", "source_location": str(plan_dir),
        "official_server": 0, "submission_eligible": 0,
        "reward_weights": candidate, "applied_rewards": {}, "skipped_rewards": [],
        "metrics": {}, "verdict": "PLANNED", "confidence": "HYPOTHESIS",
        "notes": reason, "software_version": __version__,
    }
    try:
        db.insert_experiment(record)
        with db.connect() as connection:
            connection.execute(
                "INSERT INTO lineage_edges(child_id,parent_id,reward_changes,reason,hypothesis) VALUES(?,?,?,?,?)",
                (experiment_id, parent_id, json.dumps(effective_changes, ensure_ascii=False), reason, hypothesis),
            )
            connection.execute(
                "INSERT INTO experiment_plans(experiment_id,status,changes,reward_literals,hypothesis,success_criteria,failure_criteria,calculation_policy,background_fingerprint,background_source_sha256,planned_repeats,recommended_repeats,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (experiment_id, "PLANNED", json.dumps(effective_changes, ensure_ascii=False),
                 json.dumps(literals, ensure_ascii=False), hypothesis, success_criteria,
                 failure_criteria, json.dumps(fidelity, ensure_ascii=False),
                 fidelity["background_fingerprint"], fidelity["background_source_sha256"],
                 repeats.planned, repeats.recommended, utc_now()),
            )
    except Exception:
        # Leave no false planned record if the DB transaction failed. The
        # directory contains only generated plan files at this point.
        raise

    snippet_path = path_for(f"generated/{experiment_id}_REWARD_WEIGHTS.py")
    snippet_path.write_text(reward_weights_snippet(candidate, literals=literals) + "\n", encoding="utf-8")
    write_calculation_policy(robot)
    payload = {
        "experiment_id": experiment_id, "status": "PLANNED", "robot": robot,
        "parent": parent_id, "changes": effective_changes, "reward_weights": candidate,
        "reward_literals": literals, "hypothesis": hypothesis,
        "success_criteria": success_criteria, "failure_criteria": failure_criteria,
        "reason": reason, "confounding": confounding,
        "repeat_plan": asdict(repeats), "calculation_policy": fidelity,
        "background_locked": True, "server_cost": "UNKNOWN",
        "server_eta": "UNKNOWN — no matching official-server history",
        "reward_weights_file": str(snippet_path),
    }
    (plan_dir / "experiment_plan.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (plan_dir / "metadata.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (plan_dir / "verdict.md").write_text("# PLANNED\n\nNo result has been observed.\n", encoding="utf-8")
    return payload


def list_plans(database: Database | None = None, status: str | None = "PLANNED") -> list[dict[str, Any]]:
    db = database or Database(); db.initialize()
    query = "SELECT * FROM experiment_plans"
    params: tuple[Any, ...] = ()
    if status:
        query += " WHERE status=?"; params = (status,)
    query += " ORDER BY created_at DESC"
    with db.connect() as connection:
        rows = [dict(row) for row in connection.execute(query, params)]
    for row in rows:
        for field in ("changes", "reward_literals", "calculation_policy"):
            row[field] = json.loads(row[field])
    return rows
