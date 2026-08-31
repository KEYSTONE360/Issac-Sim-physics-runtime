from __future__ import annotations

from typing import Any

from ..database import Database


def reward_changes(a: dict[str, Any], b: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    left = a.get("applied_rewards") or a.get("reward_weights") or {}
    right = b.get("applied_rewards") or b.get("reward_weights") or {}
    return {name: {"from": left.get(name), "to": right.get(name)} for name in sorted(set(left) | set(right)) if left.get(name) != right.get(name)}


def confounding_classification(change_count: int, designed_interaction: bool = False) -> dict[str, str]:
    if designed_interaction and change_count >= 2:
        return {"classification": "DESIGNED_INTERACTION_TEST", "severity": "DECLARED"}
    if change_count == 0:
        return {"classification": "REPLICATION", "severity": "NONE"}
    if change_count == 1:
        return {"classification": "CLEAN_SINGLE_VARIABLE", "severity": "NONE"}
    severity = "MEDIUM" if change_count == 2 else "HIGH" if change_count <= 4 else "VERY_HIGH"
    return {"classification": "MULTI_VARIABLE_CONFOUNDED", "severity": severity}


def experiment_diff(a_id: str, b_id: str, database: Database | None = None) -> dict[str, Any]:
    db = database or Database()
    a, b = db.get_experiment(a_id), db.get_experiment(b_id)
    if not a or not b:
        raise KeyError("experiment not found")
    changes = reward_changes(a, b)
    am, bm = a.get("metrics", {}), b.get("metrics", {})
    metric_diff = {}
    for name in sorted(set(am) | set(bm)):
        av, bv = am.get(name), bm.get(name)
        metric_diff[name] = {"from": av, "to": bv, "delta": bv - av if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else None}
    return {
        "a": a_id, "b": b_id, "config_diff": changes, "metric_diff": metric_diff,
        "artifact_diff": {}, "evaluation_diff": {},
        "confounding": confounding_classification(len(changes)),
        "causal_claim_allowed": len(changes) == 1,
    }


def lineage(experiment_id: str, database: Database | None = None) -> list[str]:
    db = database or Database()
    result, seen, current = [], set(), experiment_id
    while current:
        if current in seen:
            result.append("CYCLE_DETECTED")
            break
        seen.add(current)
        result.append(current)
        item = db.get_experiment(current)
        current = item.get("parent_experiment") if item else None
    return result

