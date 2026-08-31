from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import path_for
from ..constants import EvidenceStatus, HASH_CHUNK_BYTES

SUPPORTED = {".py", ".yaml", ".yml", ".json", ".toml", ".usd", ".usda", ".usdc", ".urdf", ".xml", ".mjcf", ".pt", ".pth", ".onnx", ".md", ".txt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_python(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"symbols": [], "imports": [], "classes": []}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError) as exc:
        result["parse_error"] = str(exc)
        return result
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result["symbols"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            result["classes"].append({"name": node.name, "bases": [ast.unparse(x) for x in node.bases]})
        elif isinstance(node, ast.Import):
            result["imports"].extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result["imports"].append(node.module or "")
    return result


def scan_source(root: Path | None = None) -> dict[str, Any]:
    root = root or path_for("ncrc_source")
    files: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.suffix.lower() not in SUPPORTED and path.name.lower() not in {"readme", "requirements"}:
                continue
            item: dict[str, Any] = {
                "path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path), "status": EvidenceStatus.VERIFIED_NCRC_SOURCE,
            }
            if path.suffix.lower() == ".py":
                item["python"] = inspect_python(path)
            files.append(item)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root), "source_available": bool(files),
        "status": EvidenceStatus.VERIFIED_NCRC_SOURCE if files else EvidenceStatus.UNKNOWN,
        "ncrc_default": {"status": "AVAILABLE" if files else "NOT_AVAILABLE"},
        "files": files,
        "rules": [], "robot": {}, "physics": {}, "actions": {}, "observations": {},
        "rewards": {}, "terminations": {}, "reset": {}, "commands": {},
        "randomization": {}, "training": {},
    }


def write_manifest(destination: Path | None = None) -> Path:
    destination = destination or path_for("generated/ncrc_manifest.json")
    destination.write_text(json.dumps(scan_source(), indent=2, ensure_ascii=False), encoding="utf-8")
    return destination

