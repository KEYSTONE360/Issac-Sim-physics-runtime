from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import path_for


SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    robot TEXT NOT NULL CHECK(robot IN ('H1','Go2')),
    timestamp TEXT NOT NULL,
    parent_experiment TEXT REFERENCES experiments(experiment_id),
    run_type TEXT NOT NULL DEFAULT 'IMPORTED',
    source_location TEXT,
    official_server INTEGER NOT NULL DEFAULT 0,
    submission_eligible INTEGER NOT NULL DEFAULT 0,
    reward_weights TEXT NOT NULL DEFAULT '{}',
    applied_rewards TEXT NOT NULL DEFAULT '{}',
    skipped_rewards TEXT NOT NULL DEFAULT '[]',
    iterations INTEGER,
    seed INTEGER,
    determinism_status TEXT NOT NULL DEFAULT 'UNKNOWN',
    report_path TEXT, env_path TEXT, model_best_path TEXT,
    policy_pt_path TEXT, policy_onnx_path TEXT,
    normal_video_path TEXT, push_video_path TEXT,
    training_start TEXT, training_end TEXT, wall_time REAL,
    metrics TEXT NOT NULL DEFAULT '{}', annotations TEXT NOT NULL DEFAULT '[]',
    evidence TEXT NOT NULL DEFAULT '[]', notes TEXT,
    verdict TEXT NOT NULL DEFAULT 'INCONCLUSIVE',
    confidence TEXT NOT NULL DEFAULT 'LOW',
    replication_group_id TEXT,
    source_hash TEXT,
    software_version TEXT,
    locked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    kind TEXT NOT NULL, original_name TEXT NOT NULL, stored_path TEXT NOT NULL,
    sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'FOUND', created_at TEXT NOT NULL,
    UNIQUE(experiment_id, kind, sha256)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_sha ON artifacts(sha256);
CREATE TABLE IF NOT EXISTS reward_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT REFERENCES experiments(experiment_id),
    robot TEXT NOT NULL, reward_name TEXT NOT NULL, value REAL,
    value_status TEXT NOT NULL DEFAULT 'UNTESTED_CANDIDATE', evidence_status TEXT NOT NULL,
    UNIQUE(experiment_id, reward_name)
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    name TEXT NOT NULL, value REAL, unit TEXT, source TEXT, evidence_status TEXT NOT NULL,
    UNIQUE(experiment_id, name)
);
CREATE TABLE IF NOT EXISTS annotations (id INTEGER PRIMARY KEY, experiment_id TEXT, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hypotheses (id INTEGER PRIMARY KEY, experiment_id TEXT, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence (id INTEGER PRIMARY KEY, subject TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS recommendations (id INTEGER PRIMARY KEY, robot TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS benchmarks (id INTEGER PRIMARY KEY, task_type TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eta_predictions (id INTEGER PRIMARY KEY, task_type TEXT NOT NULL, predicted_low REAL, predicted_high REAL, actual REAL, confidence TEXT, payload TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS budgets (id INTEGER PRIMARY KEY, scope TEXT UNIQUE NOT NULL, total REAL, used REAL NOT NULL DEFAULT 0, reserved REAL NOT NULL DEFAULT 0, source_status TEXT NOT NULL DEFAULT 'UNKNOWN');
CREATE TABLE IF NOT EXISTS rules_versions (id INTEGER PRIMARY KEY, version TEXT, effective_date TEXT, sha256 TEXT NOT NULL, path TEXT NOT NULL, status TEXT NOT NULL, imported_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS submission_candidates (id INTEGER PRIMARY KEY, experiment_id TEXT UNIQUE NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lineage_edges (child_id TEXT PRIMARY KEY REFERENCES experiments(experiment_id), parent_id TEXT NOT NULL REFERENCES experiments(experiment_id), reward_changes TEXT NOT NULL, reason TEXT, hypothesis TEXT);
CREATE TABLE IF NOT EXISTS experiment_plans (
    experiment_id TEXT PRIMARY KEY REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'PLANNED',
    changes TEXT NOT NULL,
    reward_literals TEXT NOT NULL,
    hypothesis TEXT,
    success_criteria TEXT,
    failure_criteria TEXT,
    calculation_policy TEXT NOT NULL,
    background_fingerprint TEXT NOT NULL,
    background_source_sha256 TEXT NOT NULL,
    planned_repeats INTEGER NOT NULL,
    recommended_repeats INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or path_for("database/ncrc_lab.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    def backup(self) -> Path:
        self.initialize()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target = path_for(f"backups/ncrc_lab_{stamp}.sqlite")
        target.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as source, closing(sqlite3.connect(target)) as destination:
            source.backup(destination)
        return target

    def next_experiment_id(self, robot: str) -> str:
        prefix = f"EXP-{robot}-"
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT experiment_id FROM experiments WHERE robot=?", (robot,)
            ).fetchall()
        numbers = []
        for row in rows:
            try:
                numbers.append(int(row["experiment_id"].removeprefix(prefix)))
            except ValueError:
                continue
        return f"{prefix}{max(numbers, default=0) + 1:04d}"

    def insert_experiment(self, record: dict[str, Any]) -> None:
        json_fields = {"reward_weights", "applied_rewards", "skipped_rewards", "metrics", "annotations", "evidence"}
        fields = [
            "experiment_id", "robot", "timestamp", "parent_experiment", "run_type",
            "source_location", "official_server", "submission_eligible", "reward_weights",
            "applied_rewards", "skipped_rewards", "iterations", "seed", "determinism_status",
            "report_path", "env_path", "model_best_path", "policy_pt_path", "policy_onnx_path",
            "normal_video_path", "push_video_path", "training_start", "training_end", "wall_time",
            "metrics", "annotations", "evidence", "notes", "verdict", "confidence",
            "replication_group_id", "source_hash", "software_version", "locked", "created_at",
        ]
        values = []
        defaults: dict[str, Any] = {
            "timestamp": utc_now(), "run_type": "IMPORTED", "official_server": 0,
            "submission_eligible": 0, "reward_weights": {}, "applied_rewards": {},
            "skipped_rewards": [], "determinism_status": "UNKNOWN", "metrics": {},
            "annotations": [], "evidence": [], "verdict": "INCONCLUSIVE",
            "confidence": "LOW", "locked": 0, "created_at": utc_now(),
        }
        for field in fields:
            value = record.get(field, defaults.get(field))
            if field in json_fields and not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            values.append(value)
        placeholders = ",".join("?" for _ in fields)
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO experiments ({','.join(fields)}) VALUES ({placeholders})", values
            )

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        for field in ("reward_weights", "applied_rewards", "skipped_rewards", "metrics", "annotations", "evidence"):
            try:
                result[field] = json.loads(result[field])
            except (TypeError, json.JSONDecodeError):
                pass
        return result

    def list_experiments(self, robot: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM experiments"
        params: tuple[Any, ...] = ()
        if robot:
            query += " WHERE robot=?"
            params = (robot,)
        query += " ORDER BY timestamp DESC"
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def export_json(self, destination: Path) -> None:
        tables = (
            "experiments", "artifacts", "reward_values", "metrics", "annotations",
            "hypotheses", "evidence", "recommendations", "benchmarks",
            "eta_predictions", "budgets", "rules_versions", "submission_candidates",
        )
        payload: dict[str, list[dict[str, Any]]] = {}
        with self.connect() as connection:
            for table in tables:
                payload[table] = [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
