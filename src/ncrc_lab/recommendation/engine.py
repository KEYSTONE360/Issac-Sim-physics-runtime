from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from typing import Any

from ..config import path_for
from ..database import Database, utc_now


@dataclass(frozen=True)
class RepeatPlan:
    stage: str
    planned: int
    recommended: int
    minimum_useful: int
    maximum_justified: int
    reason: str


def repeat_plan(stage: str = "exploration", confounded: bool = False,
                dramatically_worse: bool = False, budget_runs: int | None = None) -> RepeatPlan:
    stage = stage.lower()
    if confounded:
        plan = RepeatPlan("ISOLATION_REQUIRED", 1, 1, 1, 1, "Confounded result; isolate variables before replication.")
    elif dramatically_worse:
        plan = RepeatPlan("SCREENING_FAIL", 0, 0, 0, 1, "Do not spend more server time before a new hypothesis.")
    elif stage == "promising":
        plan = RepeatPlan("PROMISING", 1, 3, 2, 5, "Confirm gross improvement after the screening run.")
    elif stage == "finalist":
        plan = RepeatPlan("FINALIST", 3, 5, 3, 10, "Estimate stability before submission promotion.")
    elif stage == "research":
        plan = RepeatPlan("RESEARCH", 5, 10, 5, 20, "Stronger statistical claim requires more independent runs.")
    else:
        plan = RepeatPlan("EXPLORATION", 1, 1, 1, 1, "One screening run maximizes information efficiency.")
    if budget_runs is not None and plan.planned > budget_runs:
        return RepeatPlan(plan.stage, max(0, budget_runs), min(plan.recommended, max(0, budget_runs)), min(plan.minimum_useful, max(0, budget_runs)), min(plan.maximum_justified, max(0, budget_runs)), plan.reason + " Reduced by available budget.")
    return plan


def pareto_front(experiments: list[dict[str, Any]], objectives: dict[str, str]) -> list[dict[str, Any]]:
    eligible = [e for e in experiments if all(isinstance((e.get("metrics") or {}).get(k), (int, float)) for k in objectives)]
    result = []
    for candidate in eligible:
        cm = candidate["metrics"]
        dominated = False
        for other in eligible:
            if other is candidate:
                continue
            om = other["metrics"]
            weak = all(om[k] >= cm[k] if direction == "max" else om[k] <= cm[k] for k, direction in objectives.items())
            strict = any(om[k] > cm[k] if direction == "max" else om[k] < cm[k] for k, direction in objectives.items())
            if weak and strict:
                dominated = True; break
        if not dominated:
            result.append(candidate)
    return result


def current_best(robot: str, database: Database | None = None) -> dict[str, Any]:
    db = database or Database()
    experiments = [db.get_experiment(row["experiment_id"]) for row in db.list_experiments(robot)]
    experiments = [x for x in experiments if x]
    front = pareto_front(experiments, {"fall": "min", "terrain": "max"})
    if not front:
        return {"status": "INCONCLUSIVE", "robot": robot, "reason": "No run has both fall and terrain metrics.", "candidates": []}
    return {"status": "PARETO_SET", "robot": robot, "reason": "No single metric is treated as the submission objective.", "candidates": [x["experiment_id"] for x in front]}


def recommend(robot: str, mode: str = "BALANCED", database: Database | None = None) -> dict[str, Any]:
    db = database or Database(); db.initialize()
    best = current_best(robot, db)
    catalog_path = path_for(f"generated/reward_catalog_{robot.lower()}.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {"rewards": []}
    if not best["candidates"] or not catalog["rewards"]:
        return {
            "status": "INSUFFICIENT_EVIDENCE", "robot": robot, "mode": mode.upper(),
            "candidate": None, "confidence": "LOW", "expected_information_gain": "UNKNOWN",
            "reason": "A verified baseline with comparable metrics and a source/env-derived reward catalog are required.",
            "repeat_plan": repeat_plan().__dict__,
        }
    baseline = db.get_experiment(best["candidates"][0])
    tested = set((baseline or {}).get("applied_rewards", {}))
    selectable = [r for r in catalog["rewards"] if r.get("active") and r.get("name") in tested]
    return {
        "status": "HYPOTHESIS_REQUIRED", "robot": robot, "mode": mode.upper(),
        "parent": baseline["experiment_id"] if baseline else None, "candidate": None,
        "selectable_rewards": [r["name"] for r in selectable], "confidence": "LOW",
        "expected_information_gain": "UNKNOWN",
        "reason": "No evidence-backed numeric direction exists; the engine will not invent an optimum.",
        "repeat_plan": repeat_plan().__dict__,
    }

