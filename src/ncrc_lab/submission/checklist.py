from __future__ import annotations

from typing import Any

from ..database import Database
from ..recommendation.engine import repeat_plan


def checklist(experiment_id: str, database: Database | None = None) -> dict[str, Any]:
    db = database or Database()
    item = db.get_experiment(experiment_id)
    if not item:
        raise KeyError(experiment_id)
    unknown = sorted(set(item["reward_weights"]) - set(item["applied_rewards"]) - set(item["skipped_rewards"]))
    plan = repeat_plan("finalist" if item.get("verdict") == "FINALIST" else "exploration")
    return {
        "title": "SERVER RUN CHECKLIST", "robot": item["robot"], "experiment_id": experiment_id,
        "official_server": bool(item["official_server"]),
        "submission_eligible": bool(item["submission_eligible"]),
        "candidate_reward_list": item["reward_weights"], "expected_ok": sorted(item["applied_rewards"]),
        "known_warn_skip": item["skipped_rewards"], "unknown_rewards": unknown,
        "diff_count": None, "runs_now": plan.planned,
        "recommended_if_promising": plan.recommended,
        "server_eta": "UNKNOWN — no matching official server history",
        "cloud_budget_impact": "UNKNOWN",
        "checks": [
            "Confirm latest official NCRC rules and source version",
            "Confirm only REWARD_WEIGHTS numeric values changed",
            "Verify every training-log [OK] and record every [WARN] skip",
            "Run normal play and push play separately",
            "Preserve env/report/model/policy/video artifacts",
        ],
    }

