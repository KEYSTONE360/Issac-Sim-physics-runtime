from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def path_for(name: str) -> Path:
    return project_root() / name


def ensure_layout() -> None:
    for relative in (
        "ncrc_source", "rules", "incoming", "runs/H1", "runs/Go2",
        "database", "evidence/papers", "evidence/official",
        "presets/ncrc_default", "presets/user", "generated",
        "benchmark_profiles", "backups", "logs",
    ):
        path_for(relative).mkdir(parents=True, exist_ok=True)


def settings_path() -> Path:
    return path_for("generated/settings.json")


def load_settings() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "active_robot": "H1",
        "low_disk_warning_gb": 10,
        "memory_soft_gb": 8.0,
        "memory_strong_gb": 9.5,
        "memory_hard_gb": 10.5,
    }
    path = settings_path()
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                defaults.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    return defaults


def save_settings(settings: dict[str, Any]) -> None:
    settings_path().write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
