from __future__ import annotations

import csv
import json
import math
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..config import path_for


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _trace_rows(path: Path) -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        stream = path.open("r", encoding="utf-8-sig", newline="")
        reader = csv.DictReader(stream)
        return (iter(reader), {"schema": "CSV_RAW_TERMS", "stream": stream})
    if suffix == ".jsonl":
        stream = path.open("r", encoding="utf-8")
        def rows() -> Iterator[dict[str, Any]]:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("each JSONL row must be an object")
                    yield value
        return rows(), {"schema": "JSONL_RAW_TERMS", "stream": stream}
    if suffix == ".json":
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("large traces must use streaming CSV or JSONL")
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps = payload.get("steps") if isinstance(payload, dict) else None
        if not isinstance(steps, list):
            raise ValueError("JSON trace requires a steps list")
        return iter(steps), {"schema": payload.get("schema", "JSON_RAW_TERMS"), "dt": payload.get("dt")}
    raise ValueError("trace must be .csv, .jsonl, or .json")


def replay_raw_trace(
    trace_path: Path,
    weights: dict[str, float],
    dt: float,
    term_order: list[str],
) -> dict[str, Any]:
    """Emulate Isaac Lab RewardManager ordered float32 accumulation.

    Each trace field must be the unweighted value returned by the official
    reward function for one environment and one step.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    rows, metadata = _trace_rows(trace_path)
    totals = {name: _f32(0.0) for name in term_order}
    episode_total = _f32(0.0)
    step_count = 0
    minimum, maximum = math.inf, -math.inf
    try:
        for row in rows:
            step_total = _f32(0.0)
            for name in term_order:
                weight = float(weights.get(name, 0.0))
                if weight == 0.0:
                    value = _f32(0.0)
                else:
                    if name not in row or row[name] in (None, ""):
                        raise ValueError(f"trace row {step_count + 1} is missing active term: {name}")
                    raw = _f32(float(row[name]))
                    value = _f32(_f32(raw * _f32(weight)) * _f32(dt))
                step_total = _f32(step_total + value)
                totals[name] = _f32(totals[name] + value)
            episode_total = _f32(episode_total + step_total)
            minimum = min(minimum, step_total); maximum = max(maximum, step_total)
            step_count += 1
    finally:
        stream = metadata.get("stream")
        if stream:
            stream.close()
    if step_count == 0:
        raise ValueError("trace has no steps")
    return {
        "mode": "ISAAC_FORMULA_REPLAY", "precision": "ORDERED_IEEE754_FLOAT32_EMULATION",
        "trace_schema": metadata["schema"], "trace_path": str(trace_path),
        "environment_dt": dt, "step_count": step_count, "term_order": term_order,
        "term_episode_sums": totals, "episode_total": episode_total,
        "mean_step_reward": episode_total / step_count,
        "min_step_reward": minimum, "max_step_reward": maximum,
        "formula_simplified": False, "physics_recomputed": False,
        "limitations": "Raw term values must come from the target NCRC/Isaac environment; this replay does not train or simulate physics.",
    }


def replay_with_manifest(trace_path: Path, weights: dict[str, float] | None, robot: str, manifest_path: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path or path_for(f"generated/isaac_fidelity_{robot.lower()}.json")
    if not manifest_path.exists():
        raise FileNotFoundError("generate Isaac fidelity manifest first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("full_formula_replay_ready"):
        raise ValueError("Isaac formula/source coverage is incomplete")
    order = [term["term_name"] for term in manifest["terms"] if term.get("active")]
    selected_weights = weights or manifest["reward_weights"]
    result = replay_raw_trace(trace_path, selected_weights, float(manifest["environment_dt_literal"]), order)
    result["isaac_fidelity_manifest"] = str(manifest_path)
    result["isaac_lab_commit"] = manifest["isaac_lab_commit"]
    result["ncrc_exact_isaac_lab_commit"] = manifest["ncrc_exact_isaac_lab_commit"]
    return result
