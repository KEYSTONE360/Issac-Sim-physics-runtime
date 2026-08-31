from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import path_for


REPOSITORY = "isaac-sim/IsaacLab"


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "NCRC-Reward-Lab/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "NCRC-Reward-Lab/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _symbols(source: str) -> list[dict[str, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    values = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            values.append({"name": node.name, "kind": "function", "doc": (ast.get_docstring(node) or "").split("\n", 1)[0]})
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            values.append({"name": node.name, "kind": "callable_class", "doc": (ast.get_docstring(node) or "").split("\n", 1)[0]})
    return values


def sync_nvidia_reward_reference(ref: str = "v2.3.2", progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    progress = progress or (lambda _: None)
    tree_api = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{ref}?recursive=1"
    tree = _request_json(tree_api)
    commit = str(tree["sha"])
    paths = sorted(
        item["path"] for item in tree.get("tree", [])
        if item.get("type") == "blob" and item["path"].startswith("source/") and item["path"].endswith("rewards.py")
    )
    manager_paths = [
        "source/isaaclab/isaaclab/managers/reward_manager.py",
        "source/isaaclab/isaaclab/managers/manager_term_cfg.py",
    ]
    paths = sorted(set(paths + manager_paths))
    destination = path_for(f"evidence/official/isaaclab_reference/{commit}")
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(relative: str) -> tuple[str, bytes]:
        cached = destination / relative
        if cached.exists():
            return relative, cached.read_bytes()
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{relative}"
        return relative, _download(url)

    downloaded: list[tuple[str, bytes]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, value in enumerate(pool.map(fetch, paths), 1):
            downloaded.append(value)
            progress(f"[{index}/{len(paths)}] {value[0]}")

    entries, files = [], []
    for relative, content in downloaded:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        module_parts = relative.removeprefix("source/").removesuffix(".py").split("/")
        if len(module_parts) >= 2 and module_parts[0] == module_parts[1]:
            module_parts.pop(0)
        module = ".".join(module_parts)
        symbols = _symbols(content.decode("utf-8", errors="replace")) if relative.endswith("rewards.py") else []
        files.append({"path": relative, "sha256": digest, "symbol_count": len(symbols)})
        for symbol in symbols:
            entries.append({
                **symbol, "qualified_name": f"{module}:{symbol['name']}",
                "classification": "ISAACLAB_REFERENCE", "evidence_status": "VERIFIED_NVIDIA",
                "source_file": relative, "source_commit": commit,
                "weight_editable": False,
                "reason": "A callable has no weight until the NCRC environment defines a RewardTermCfg term.",
            })
    payload = {
        "repository": REPOSITORY, "ref_requested": ref, "commit": commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "ISAACLAB_REFERENCE", "isaac_sim_compatibility": "5.1.0" if ref.startswith("v2.3") else "UNKNOWN", "ncrc_version_match": "UNKNOWN",
        "source_dir": str(destination), "files": files, "functions": entries,
    }
    manifest = path_for("generated/isaaclab_reward_functions.json")
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ref": ref, "commit": commit, "file_count": len(files), "function_count": len(entries), "manifest": str(manifest), "source_dir": str(destination)}
