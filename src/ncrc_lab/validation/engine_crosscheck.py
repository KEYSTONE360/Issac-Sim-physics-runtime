from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from ..config import path_for


def _native_path() -> Path:
    for candidate in (path_for("native/ncrc_physics.exe"), path_for("native_runtime/build/Release/ncrc_physics.exe")):
        if candidate.exists(): return candidate
    raise FileNotFoundError("native PhysX executable not found")


def _run_native() -> dict[str, Any]:
    proc = subprocess.run([str(_native_path()), "test"], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(proc.stdout)


def _mujoco_box(dt: float, steps: int = 200) -> dict[str, float]:
    import mujoco
    xml = f"""<mujoco><option timestep='{dt}' gravity='0 0 -9.81' integrator='Euler'/>
    <default><geom friction='1.0 1.0 0.0' solref='0.02 1'/></default>
    <worldbody><geom name='terrain' type='plane' size='100 100 0.05'/>
    <body name='free_fall_probe' pos='0 0 2'><freejoint/><geom type='box' size='0.1 0.1 0.1' mass='1'/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    free_z = free_vz = 0.0; contacts = 0
    for step in range(steps):
        mujoco.mj_step(model, data)
        contacts += int(data.ncon)
        if step == 99: free_z, free_vz = float(data.qpos[2]), float(data.qvel[2])
    return {"free_fall_z": free_z, "free_fall_vz": free_vz, "final_z": float(data.qpos[2]), "final_vz": float(data.qvel[2]), "contact_points_observed": contacts}


def _metric(reference: float, actual: float) -> dict[str, float]:
    absolute = abs(actual - reference)
    relative = absolute / max(abs(reference), 1.0e-12)
    return {"reference": reference, "actual": actual, "absolute_error": absolute, "relative_error": relative, "rmse": absolute, "max_error": absolute, "mean_error": absolute}


def cross_validate_engines() -> dict[str, Any]:
    first = _run_native(); second = _run_native()
    dt = float(first["physics_dt"]); count = int(first["free_fall_step"]); gravity = -9.81
    # Both PhysX and the configured reference use semi-implicit Euler here.
    analytic_v = gravity * dt * count
    analytic_z = float(first["initial_z"]) + gravity * dt * dt * count * (count + 1) / 2.0
    mujoco_result = _mujoco_box(dt)
    comparisons = {
        "physx_vs_semi_implicit_free_fall_z": _metric(analytic_z, float(first["free_fall_z"])),
        "physx_vs_semi_implicit_free_fall_vz": _metric(analytic_v, float(first["free_fall_vz"])),
        "physx_vs_mujoco_free_fall_z": _metric(mujoco_result["free_fall_z"], float(first["free_fall_z"])),
        "physx_vs_mujoco_free_fall_vz": _metric(mujoco_result["free_fall_vz"], float(first["free_fall_vz"])),
        "physx_vs_mujoco_resting_height": _metric(mujoco_result["final_z"], float(first["final_z"])),
        "physx_repeat_free_fall_z": _metric(float(first["free_fall_z"]), float(second["free_fall_z"])),
        "physx_repeat_joint_position": _metric(float(first["joint_position"]), float(second["joint_position"])),
        "physx_repeat_joint_velocity": _metric(float(first["joint_velocity"]), float(second["joint_velocity"])),
        "physx_repeat_joint_force": _metric(float(first["joint_force"]), float(second["joint_force"])),
    }
    checks = {
        "analytic_free_fall": comparisons["physx_vs_semi_implicit_free_fall_z"]["absolute_error"] < 1e-5 and comparisons["physx_vs_semi_implicit_free_fall_vz"]["absolute_error"] < 1e-5,
        "cross_engine_free_fall": comparisons["physx_vs_mujoco_free_fall_z"]["absolute_error"] < 1e-5 and comparisons["physx_vs_mujoco_free_fall_vz"]["absolute_error"] < 1e-5,
        "cross_engine_ground_rest": comparisons["physx_vs_mujoco_resting_height"]["absolute_error"] < 5e-3,
        "physx_deterministic_repeat": all(comparisons[name]["absolute_error"] < 1e-7 for name in ("physx_repeat_free_fall_z", "physx_repeat_joint_position", "physx_repeat_joint_velocity", "physx_repeat_joint_force")),
        "contact_observed_both": int(first["contact_points_observed"]) > 0 and int(mujoco_result["contact_points_observed"]) > 0,
    }
    report = {
        "title": "NCRC ENGINE CROSS-VALIDATION REPORT",
        "target": "Isaac Sim 5.1 observable physics path",
        "primary_engine": "NVIDIA PhysX 5.6.1 CPU, Omniverse PhysX 107.3",
        "cross_engine": "MuJoCo 3.12.0 CPU",
        "renderer": False, "cuda_required": False,
        "tests": checks, "passed": all(checks.values()), "comparisons": comparisons,
        "raw": {"physx_run_1": first, "physx_run_2": second, "mujoco": mujoco_result, "analytic": {"z": analytic_z, "vz": analytic_v}},
        "not_yet_cross_validated": {
            "full_h1_articulation": "NOT_IMPLEMENTED_IN_NATIVE_PHYSX",
            "contact_force_rmse": "REFERENCE_TRACE_REQUIRED",
            "observation_rmse": "ISAAC_SIM_TRACE_REQUIRED",
            "reward_rmse": "ISAAC_SIM_TRACE_REQUIRED",
        },
    }
    destination = path_for(f"generated/engine_cross_validation_{int(time.time())}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_file"] = str(destination)
    return report
