from __future__ import annotations

import ast
import difflib
import json
import re
import shutil
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ..config import path_for


NUMERIC = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _numeric_literal(value: Any) -> str:
    literal = value if isinstance(value, str) else repr(value)
    try:
        number = Decimal(literal)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid numeric reward weight: {value}") from exc
    if not number.is_finite():
        raise ValueError("reward weights must be finite")
    return literal


def reward_weights_snippet(weights: dict[str, float], literals: dict[str, str] | None = None) -> str:
    """Render numeric literals without rounding or float reformatting."""
    lines = ["REWARD_WEIGHTS = {"]
    for key, value in sorted(weights.items()):
        literal = _numeric_literal((literals or {}).get(key, value))
        lines.append(f"    {json.dumps(str(key), ensure_ascii=False)}: {literal},")
    lines.append("}")
    return "\n".join(lines)


def patch_existing_values(original: str, weights: dict[str, float]) -> tuple[str, list[str]]:
    modified = original
    missing: list[str] = []
    for key, value in weights.items():
        pattern = re.compile(rf"(?P<prefix>(['\"]){re.escape(key)}\2\s*:\s*)(?P<number>{NUMERIC})")
        matches = list(pattern.finditer(modified))
        if len(matches) != 1:
            missing.append(key)
            continue
        modified = pattern.sub(lambda m: m.group("prefix") + repr(float(value)), modified, count=1)
    return modified, missing


def _reward_numeric_spans(source: str) -> list[tuple[int, int]]:
    lines = source.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total); total += len(line)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "REWARD_WEIGHTS" for t in targets):
            continue
        if isinstance(node.value, ast.Dict):
            for value in node.value.values:
                unary_number = isinstance(value, ast.UnaryOp) and isinstance(value.operand, ast.Constant) and isinstance(value.operand.value, (int, float))
                plain_number = isinstance(value, ast.Constant) and isinstance(value.value, (int, float)) and not isinstance(value.value, bool)
                if plain_number or unary_number:
                    spans.append((offsets[value.lineno - 1] + value.col_offset, offsets[value.end_lineno - 1] + value.end_col_offset))
    return spans


def validate_numeric_only(original: str, modified: str) -> dict[str, Any]:
    if original == modified:
        return {"ONLY_ALLOWED_VALUES_CHANGED": True, "changed_spans": 0, "reason": "NO_CHANGES"}
    old_spans, new_spans = _reward_numeric_spans(original), _reward_numeric_spans(modified)
    if not old_spans or len(old_spans) != len(new_spans):
        return {"ONLY_ALLOWED_VALUES_CHANGED": False, "changed_spans": 0, "reason": "REWARD_WEIGHTS_STRUCTURE_CHANGED_OR_MISSING"}
    old_masked, new_masked = original, modified
    for start, end in reversed(old_spans): old_masked = old_masked[:start] + "<NUMBER>" + old_masked[end:]
    for start, end in reversed(new_spans): new_masked = new_masked[:start] + "<NUMBER>" + new_masked[end:]
    allowed = old_masked == new_masked
    changes = sum(1 for a, b in zip([original[s:e] for s, e in old_spans], [modified[s:e] for s, e in new_spans]) if a != b)
    return {"ONLY_ALLOWED_VALUES_CHANGED": allowed, "changed_spans": changes, "reason": "NUMERIC_VALUES_ONLY" if allowed else "NON_NUMERIC_BYTES_CHANGED"}


def generate_full_patch(original_path: Path, weights: dict[str, float]) -> dict[str, Any]:
    original = original_path.read_text(encoding="utf-8")
    backup = path_for(f"backups/rewards_original_{datetime.now():%Y%m%d_%H%M%S_%f}{original_path.suffix}")
    shutil.copy2(original_path, backup)
    modified, missing = patch_existing_values(original, weights)
    validation = validate_numeric_only(original, modified)
    output = path_for(f"generated/{original_path.stem}_server_patch{original_path.suffix}")
    if missing or not validation["ONLY_ALLOWED_VALUES_CHANGED"]:
        return {"server_ready": False, "missing_keys": missing, "validation": validation, "backup": str(backup), "output": None}
    output.write_text(modified, encoding="utf-8", newline="")
    return {"server_ready": True, "missing_keys": [], "validation": validation, "backup": str(backup), "output": str(output)}


def rollback(backup: Path, destination: Path) -> None:
    if path_for("backups").resolve() not in backup.resolve().parents:
        raise ValueError("rollback source must be under project backups")
    shutil.copy2(backup, destination)
