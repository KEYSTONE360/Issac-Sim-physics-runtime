from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .config import path_for
from .database import Database


def record_ncrc_runtime_profile() -> dict[str, Any]:
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "isaac_sim": {
            "version": "5.1.0",
            "status": "VERIFIED_SERVER_RUN",
            "provenance": "USER_CONFIRMATION",
        },
        "isaac_lab": {
            "compatible_reference": "v2.3.2",
            "compatible_reference_status": "VERIFIED_NVIDIA",
            "exact_ncrc_version": "UNKNOWN",
            "exact_ncrc_commit": "UNKNOWN",
        },
        "physx": {"backend": "PhysX", "exact_version": "UNKNOWN"},
        "rsl_rl": {"exact_version": "UNKNOWN"},
        "python": {
            "isaac_sim_5_1_required": "3.11",
            "ncrc_server_actual": "UNKNOWN",
        },
        "calculation_target": "NCRC_ISAAC_SIM_5_1",
        "forbidden_targets": ["ISAAC_SIM_6_X"],
    }
    destination = path_for("generated/ncrc_runtime_profile.json")
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    db = Database(); db.initialize()
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO evidence(subject,payload,status) VALUES(?,?,?)",
            ("ncrc_runtime/isaac_sim", json.dumps(payload, ensure_ascii=False), "VERIFIED_SERVER_RUN"),
        )
    return {**payload, "path": str(destination)}
