from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.parts.append(value)


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC_ALIASES = {
    "fall": ("fall", "fall_rate", "fall rate"),
    "terrain": ("terrain", "terrain_level", "terrain level"),
    "exploration_std": ("exploration_std", "exploration std", "std"),
    "reward": ("mean_reward", "mean reward", "reward"),
}


def parse_report(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"metrics": {}, "status": "PARSED", "warnings": []}
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        parser = _TextExtractor()
        parser.feed(raw)
        text = "\n".join(parser.parts)
    except Exception as exc:
        return {"metrics": {}, "status": "CORRUPT", "warnings": [str(exc)]}
    # NCRC HTML reports render metric values before their labels in stat cards.
    for match in re.finditer(
        rf"(?is)<div\s+class=[\"']?sv[\"']?[^>]*>\s*({NUMBER})\s*(%)?\s*</div>\s*"
        r"<div\s+class=[\"']?sk[\"']?[^>]*>\s*(.*?)\s*</div>", raw
    ):
        label = re.sub(r"<[^>]+>", " ", match.group(3)).strip().lower()
        canonical = None
        if "낙상" in label or "fall" in label: canonical = "fall"
        elif "지형" in label or "terrain" in label: canonical = "terrain"
        elif "탐색" in label and "std" in label or "exploration" in label and "std" in label: canonical = "exploration_std"
        elif "보상" in label or "reward" in label: canonical = "reward"
        if canonical:
            result["metrics"][canonical] = {
                "value": float(match.group(1)), "unit": "%" if match.group(2) else None,
                "source": str(path), "status": "EXPERIMENT_OBSERVED",
            }
    for canonical, aliases in METRIC_ALIASES.items():
        if canonical in result["metrics"]:
            continue
        for alias in aliases:
            match = re.search(rf"(?i)(?<![\w]){re.escape(alias)}(?![\w])\s*[:=]?\s*({NUMBER})\s*(%)?", text)
            if match:
                result["metrics"][canonical] = {
                    "value": float(match.group(1)),
                    "unit": "%" if match.group(2) else None,
                    "source": str(path), "status": "EXPERIMENT_OBSERVED",
                }
                break
    if not result["metrics"]:
        result["warnings"].append("NO_RECOGNIZED_METRICS")
    return result


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value or value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return float(value) if any(c in value for c in ".eE") else int(value)
    except ValueError:
        if value.startswith("[") or value.startswith("{"):
            try:
                return json.loads(value.replace("'", '"'))
            except json.JSONDecodeError:
                pass
        return value.split(" #", 1)[0].strip()


def parse_yaml_subset(path: Path) -> dict[str, Any]:
    try:
        import yaml
        class IsaacEnvSafeLoader(yaml.SafeLoader):
            """Safe loader for non-executable Isaac Lab serialization tags."""

        IsaacEnvSafeLoader.add_constructor(
            "tag:yaml.org,2002:python/tuple",
            lambda loader, node: loader.construct_sequence(node, deep=True),
        )
        IsaacEnvSafeLoader.add_constructor(
            "tag:yaml.org,2002:python/object/apply:builtins.slice",
            lambda loader, node: {"__python_slice__": loader.construct_sequence(node, deep=True)},
        )
        loaded = yaml.load(
            path.read_text(encoding="utf-8", errors="strict"), Loader=IsaacEnvSafeLoader
        )
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError("YAML root must be a mapping")
        return loaded
    except ImportError:
        pass
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#") or raw.lstrip().startswith("-"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = raw.strip()
        if ":" not in content:
            continue
        key, value = content.split(":", 1)
        key = key.strip().strip("'\"")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"invalid indentation at line {number}")
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _scalar(value)
    return root


def _find_reward_maps(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    found: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    if isinstance(value, dict):
        if path and "reward" in path[-1].lower():
            found.append((path, value))
        for key, child in value.items():
            found.extend(_find_reward_maps(child, path + (str(key),)))
    return found


def parse_env(path: Path) -> dict[str, Any]:
    try:
        data = parse_yaml_subset(path)
    except Exception as exc:
        return {"data": {}, "reward_weights": {}, "inactive_rewards": [], "status": "CORRUPT", "warnings": [str(exc)]}
    rewards: dict[str, float] = {}
    reward_terms: dict[str, dict[str, Any]] = {}
    inactive: list[str] = []
    for _, mapping in _find_reward_maps(data):
        for name, value in mapping.items():
            if value is None:
                inactive.append(name)
                reward_terms[name] = {"name": name, "weight": None, "function": "UNKNOWN", "parameters": {}, "active": False}
            elif isinstance(value, (int, float)):
                rewards[name] = float(value)
                reward_terms[name] = {"name": name, "weight": float(value), "function": "UNKNOWN", "parameters": {}, "active": True}
            elif isinstance(value, dict):
                weight = value.get("weight")
                function = value.get("func", "UNKNOWN")
                parameters = value.get("params", {}) if isinstance(value.get("params", {}), dict) else {}
                if weight is None and "weight" in value:
                    inactive.append(name)
                    reward_terms[name] = {"name": name, "weight": None, "function": function, "parameters": parameters, "active": False}
                elif isinstance(weight, (int, float)):
                    rewards[name] = float(weight)
                    reward_terms[name] = {"name": name, "weight": float(weight), "function": function, "parameters": parameters, "active": True}
    return {"data": data, "reward_weights": rewards, "reward_terms": reward_terms, "inactive_rewards": sorted(set(inactive)), "status": "PARSED", "warnings": []}


def parse_reward_application_log(path: Path) -> dict[str, Any]:
    applied: dict[str, float | None] = {}
    skipped: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        ok = re.search(r"\[OK\].*?([A-Za-z_][\w.]*)\s*(?:[:=]\s*(%s))?" % NUMBER, line, re.I)
        warn = re.search(r"\[WARN\].*?skip(?:ped)?\s*[:=-]?\s*([A-Za-z_][\w.]*)", line, re.I)
        if ok:
            applied[ok.group(1)] = float(ok.group(2)) if ok.group(2) else None
        if warn:
            skipped.append(warn.group(1))
    for name in skipped:
        applied.pop(name, None)
    return {"applied_rewards": applied, "skipped_rewards": sorted(set(skipped))}
