from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import path_for
from .constants import EvidenceStatus


def _powershell_json(script: str) -> Any:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return json.loads(result.stdout.lstrip("\ufeff").strip())
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def collect_system_profile() -> dict[str, Any]:
    cpu = _powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,"
        "NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress"
    ) or {}
    memory = _powershell_json(
        "$c=Get-CimInstance Win32_ComputerSystem;$o=Get-CimInstance Win32_OperatingSystem;"
        "[pscustomobject]@{TotalBytes=[int64]$c.TotalPhysicalMemory;"
        "AvailableBytes=[int64]($o.FreePhysicalMemory*1KB)}|ConvertTo-Json -Compress"
    ) or {}
    gpu = _powershell_json(
        "@(Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion)"
        " | ConvertTo-Json -Compress"
    ) or []
    power = _powershell_json(
        "$b=Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue|Select-Object -First 1 BatteryStatus,EstimatedChargeRemaining;"
        "if($b){$b|ConvertTo-Json -Compress}else{'null'}"
    )
    disk = shutil.disk_usage(path_for("."))
    torch_version = _package_version("torch")
    cuda_available: bool | None = None
    if torch_version:
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = None
    logical = int(cpu.get("NumberOfLogicalProcessors", os.cpu_count() or 1))
    physical = cpu.get("NumberOfCores")
    profile = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": EvidenceStatus.DERIVED,
        "cpu": {
            "model": cpu.get("Name", platform.processor()) or "UNKNOWN",
            "physical_cores": physical,
            "logical_threads": logical,
            "max_clock_mhz_reported": cpu.get("MaxClockSpeed"),
            "hybrid_topology": "P/E topology present for i5-1335U; exact affinity mapping UNKNOWN"
            if "i5-1335U" in str(cpu.get("Name")) else "UNKNOWN",
        },
        "memory": {
            "total_bytes": memory.get("TotalBytes"),
            "available_bytes": memory.get("AvailableBytes"),
            "pagefile_counted_as_ram": False,
        },
        "os": {"platform": platform.platform(), "version": platform.version(), "architecture": platform.machine()},
        "python": {"version": sys.version, "executable": sys.executable, "architecture": platform.architecture()[0]},
        "pytorch": {"version": torch_version or "NOT_INSTALLED", "cuda_available": cuda_available},
        "onnx_runtime": _package_version("onnxruntime") or "NOT_INSTALLED",
        "directml": _package_version("torch-directml") or "NOT_INSTALLED",
        "nvidia_smi": shutil.which("nvidia-smi") or "NOT_FOUND",
        "gpus": gpu if isinstance(gpu, list) else [gpu],
        "storage": {"total_bytes": disk.total, "free_bytes": disk.free},
        "power": power if power else {"state": "UNKNOWN"},
        "filesystem": {"project_root": str(path_for(".")), "separator": os.sep},
    }
    return profile


def write_system_profile(destination: Path | None = None) -> Path:
    destination = destination or path_for("generated/system_profile.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(collect_system_profile(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination

