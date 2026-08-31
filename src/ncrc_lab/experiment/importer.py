from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

from .. import __version__
from ..config import path_for
from ..database import Database, utc_now
from ..source.scanner import sha256_file
from .parsers import parse_env, parse_report, parse_reward_application_log


ARTIFACT_PATTERNS = {
    "report": ("report.html",), "env": ("env.yaml", "env.yml"),
    "model_best": ("model_best.pt", "model_best.pth"),
    "policy_pt": ("policy.pt",), "policy_onnx": ("policy.onnx",),
    "normal_video": ("normal_play.mp4", "play_video.mp4", "play.mp4"),
    "push_video": ("push_play.mp4", "push_video.mp4"),
    "training_log": ("training.log", "train.log", "output.log"),
}


def discover_artifacts(folder: Path) -> dict[str, Path]:
    files = {p.name.lower(): p for p in folder.iterdir() if p.is_file()}
    result: dict[str, Path] = {}
    for kind, names in ARTIFACT_PATTERNS.items():
        for name in names:
            if name in files:
                result[kind] = files[name]
                break
    # Official downloads commonly add " (N)" or a timestamp suffix. Prefer a
    # timestamped original when both names contain identical content.
    candidates = sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: ("_20" not in p.stem, p.name.lower()))
    for path in candidates:
        stem, suffix = path.stem.lower(), path.suffix.lower()
        normalized = re.sub(r"(?:\s*\(\d+\)|_\d{8}(?:_?\d{6})?)$", "", stem)
        kind = None
        if normalized == "env" and suffix in {".yaml", ".yml"}: kind = "env"
        elif normalized == "report" and suffix in {".html", ".htm"}: kind = "report"
        elif normalized == "model_best" and suffix in {".pt", ".pth"}: kind = "model_best"
        elif normalized == "policy" and suffix == ".pt": kind = "policy_pt"
        elif normalized == "policy" and suffix == ".onnx": kind = "policy_onnx"
        elif normalized in {"normal_play", "play_video", "play"} and suffix == ".mp4": kind = "normal_video"
        elif normalized in {"push_play", "push_video"} and suffix == ".mp4": kind = "push_video"
        if kind and kind not in result:
            result[kind] = path
    return result


def completeness(found: dict[str, Path]) -> dict[str, str]:
    required = ("report", "env", "model_best", "policy_pt", "policy_onnx", "normal_video")
    result = {kind: ("FOUND" if kind in found else "MISSING") for kind in required}
    result["push_video"] = "FOUND" if "push_video" in found else "OPTIONAL"
    result["push_status"] = "AVAILABLE" if "push_video" in found else "NOT_TESTED"
    return result


def _combined_hash(hashes: dict[str, str]) -> str:
    import hashlib
    digest = hashlib.sha256()
    for kind, value in sorted(hashes.items()):
        digest.update(kind.encode())
        digest.update(value.encode())
    return digest.hexdigest()


def import_run(folder: Path, robot: str, parent: str | None = None,
               official_server: bool = False, database: Database | None = None) -> dict[str, Any]:
    if robot not in {"H1", "Go2"}:
        raise ValueError("robot must be H1 or Go2")
    folder = folder.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    db = database or Database()
    db.initialize()
    found = discover_artifacts(folder)
    if not found:
        raise ValueError("no recognized artifacts")
    hashes = {kind: sha256_file(path) for kind, path in found.items()}
    fingerprint = _combined_hash(hashes)
    with db.connect() as connection:
        duplicate = connection.execute("SELECT experiment_id FROM experiments WHERE source_hash=?", (fingerprint,)).fetchone()
    if duplicate:
        return {"status": "DUPLICATE_BACKUP", "experiment_id": duplicate["experiment_id"], "completeness": completeness(found)}

    experiment_id = db.next_experiment_id(robot)
    run_dir = path_for(f"runs/{robot}/{experiment_id}")
    run_dir.mkdir(parents=True, exist_ok=False)
    stored: dict[str, Path] = {}
    for kind, source in found.items():
        target = run_dir / source.name
        shutil.copy2(source, target)
        stored[kind] = target

    env_result = parse_env(stored["env"]) if "env" in stored else {"reward_weights": {}, "inactive_rewards": []}
    report_result = parse_report(stored["report"]) if "report" in stored else {"metrics": {}}
    application = parse_reward_application_log(stored["training_log"]) if "training_log" in stored else {"applied_rewards": {}, "skipped_rewards": []}
    metric_values = {name: item["value"] for name, item in report_result.get("metrics", {}).items()}
    requested = env_result.get("reward_weights", {})
    # env.yaml proves configuration, not successful runtime application. Only
    # explicit server [OK] records enter applied_rewards.
    applied = application["applied_rewards"] if "training_log" in stored else {}
    skipped = application["skipped_rewards"]
    for name in skipped:
        applied.pop(name, None)
    record = {
        "experiment_id": experiment_id, "robot": robot,
        "timestamp": datetime.fromtimestamp(folder.stat().st_mtime, timezone.utc).isoformat(),
        "parent_experiment": parent, "source_location": str(folder),
        "official_server": int(official_server),
        "submission_eligible": int(official_server),
        "reward_weights": requested, "applied_rewards": applied, "skipped_rewards": skipped,
        "report_path": str(stored.get("report", "")), "env_path": str(stored.get("env", "")),
        "model_best_path": str(stored.get("model_best", "")), "policy_pt_path": str(stored.get("policy_pt", "")),
        "policy_onnx_path": str(stored.get("policy_onnx", "")),
        "normal_video_path": str(stored.get("normal_video", "")), "push_video_path": str(stored.get("push_video", "")),
        "metrics": metric_values, "source_hash": fingerprint, "software_version": __version__,
    }
    db.insert_experiment(record)
    with db.connect() as connection:
        for kind, target in stored.items():
            connection.execute(
                "INSERT INTO artifacts(experiment_id,kind,original_name,stored_path,sha256,size_bytes,status,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (experiment_id, kind, found[kind].name, str(target), hashes[kind], target.stat().st_size, "FOUND", utc_now()),
            )
        for name, item in report_result.get("metrics", {}).items():
            connection.execute(
                "INSERT OR REPLACE INTO metrics(experiment_id,name,value,unit,source,evidence_status) VALUES(?,?,?,?,?,?)",
                (experiment_id, name, item["value"], item["unit"], item["source"], item["status"]),
            )
        if parent:
            connection.execute(
                "INSERT INTO lineage_edges(child_id,parent_id,reward_changes,reason,hypothesis) VALUES(?,?,?,?,?)",
                (experiment_id, parent, "{}", "imported with parent", None),
            )
    metadata = {**record, "completeness": completeness(found), "parser": {"env": env_result.get("status"), "report": report_result.get("status")}}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "IMPORTED", "experiment_id": experiment_id, "run_dir": str(run_dir), "completeness": completeness(found)}
