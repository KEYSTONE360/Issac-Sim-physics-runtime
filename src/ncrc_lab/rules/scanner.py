from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import path_for
from ..constants import EvidenceStatus
from ..database import Database
from ..source.scanner import sha256_file


def scan_rules(root: Path | None = None) -> dict:
    root = root or path_for("rules")
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "files": files,
        "latest_rule_available": bool(files),
        "status": EvidenceStatus.VERIFIED_NCRC_RULE if files else EvidenceStatus.UNKNOWN,
        "score_model": "UNKNOWN", "training_budget": "UNKNOWN",
    }


def write_rules_manifest() -> Path:
    result = scan_rules()
    destination = path_for("generated/rules_manifest.json")
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    db = Database(); db.initialize()
    with db.connect() as connection:
        for item in result["files"]:
            connection.execute(
                "INSERT INTO rules_versions(version,effective_date,sha256,path,status,imported_at) VALUES(?,?,?,?,?,?)",
                (None, None, item["sha256"], item["path"], result["status"], result["generated_at"]),
            )
    return destination

