from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass(frozen=True)
class EtaEstimate:
    low_seconds: float | None
    high_seconds: float | None
    best_seconds: float | None
    confidence: str
    reason: str

    def __post_init__(self) -> None:
        for value in (self.low_seconds, self.high_seconds, self.best_seconds):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("ETA values must be finite and non-negative")
        if self.confidence not in {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}:
            raise ValueError("invalid ETA confidence")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def expected_finish(self, now: datetime | None = None) -> tuple[datetime, datetime] | None:
        if self.low_seconds is None or self.high_seconds is None:
            return None
        now = now or datetime.now().astimezone()
        return now + timedelta(seconds=self.low_seconds), now + timedelta(seconds=self.high_seconds)


def estimate(
    work_units: float | None,
    throughput_history: list[float] | None = None,
    startup_seconds: float = 0,
    finalization_seconds: float = 0,
) -> EtaEstimate:
    if work_units is None or work_units < 0:
        return EtaEstimate(None, None, None, "UNKNOWN", "workload is unavailable")
    samples = [x for x in (throughput_history or []) if math.isfinite(x) and x > 0]
    if not samples:
        return EtaEstimate(5, max(15, work_units), None, "LOW", "no historical benchmark")
    samples.sort()
    slow = samples[max(0, int(len(samples) * 0.1) - 1)]
    fast = samples[min(len(samples) - 1, int(len(samples) * 0.9))]
    overhead = startup_seconds + finalization_seconds
    low = overhead + work_units / fast
    high = overhead + work_units / slow
    median = samples[len(samples) // 2]
    confidence = "HIGH" if len(samples) >= 5 else "MEDIUM"
    return EtaEstimate(low, high, overhead + work_units / median, confidence, f"{len(samples)} historical samples")


def precompute_plan(task: str, operations: list[str], estimate_value: EtaEstimate,
                    planned_repetitions: str = "1", recommended_repetitions: str = "1",
                    peak_ram: str = "<1 GB", cpu_threads: int = 4) -> str:
    finish = estimate_value.expected_finish()
    eta_text = "UNKNOWN" if estimate_value.low_seconds is None else f"{estimate_value.low_seconds:.0f}s - {estimate_value.high_seconds:.0f}s"
    finish_text = "UNKNOWN" if finish is None else f"{finish[0]:%H:%M:%S} - {finish[1]:%H:%M:%S}"
    return "\n".join([
        "=" * 60, "PRE-COMPUTE PLAN", "=" * 60,
        f"Task: {task}", "Planned operations:", *[f"- {x}" for x in operations],
        f"Planned repetitions: {planned_repetitions}",
        f"Recommended repetitions: {recommended_repetitions}",
        f"ETA: {eta_text}", f"Expected finish: {finish_text}",
        f"ETA confidence: {estimate_value.confidence}",
        f"Estimated peak RAM: {peak_ram}", f"CPU threads: {cpu_threads}",
        "=" * 60,
    ])

