from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..config import path_for
from ..environment import validate_reward_only_env
from ..experiment.parsers import parse_env
from ..source.scanner import sha256_file


ENGINE_VERSION = "0.2.0"
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "torso", "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
]


def _dynamic_onnx(source: Path, destination: Path) -> Path:
    """Make only the batch axis dynamic; graph operators and weights stay unchanged."""
    import onnx
    model = onnx.load(str(source))
    for value in list(model.graph.input) + list(model.graph.output):
        shape = value.type.tensor_type.shape
        if shape.dim:
            shape.dim[0].ClearField("dim_value")
            shape.dim[0].dim_param = "batch"
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    return destination


def _rotation(data: Any) -> np.ndarray:
    import mujoco
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, data.qpos[3:7])
    return matrix.reshape(3, 3)


def _yaw(data: Any) -> float:
    r = _rotation(data)
    return math.atan2(float(r[1, 0]), float(r[0, 0]))


def _wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _joint_selection(patterns: Any) -> np.ndarray:
    if isinstance(patterns, str):
        patterns = [patterns]
    return np.asarray([i for i, name in enumerate(JOINT_NAMES) if any(re.fullmatch(p, name) for p in patterns)], dtype=int)


def _model_path() -> Path:
    candidates = [
        path_for("engine/vendor/mujoco_menagerie/unitree_h1/scene.xml"),
        Path(__file__).resolve().parents[3] / "engine/vendor/mujoco_menagerie/unitree_h1/scene.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("bundled Unitree H1 MuJoCo model is missing")


def _compatibility(data: dict[str, Any]) -> dict[str, Any]:
    physx = data.get("sim", {}).get("physx", {})
    rows = []
    translated = {
        "solver_type", "min_position_iteration_count", "max_position_iteration_count",
        "min_velocity_iteration_count", "max_velocity_iteration_count", "bounce_threshold_velocity",
        "friction_offset_threshold", "friction_correlation_distance", "enable_ccd",
    }
    for key, value in physx.items():
        rows.append({
            "yaml_path": f"sim.physx.{key}", "value": value,
            "status": "TRANSLATED_TO_MUJOCO_EQUIVALENT" if key in translated else "RECORDED_NO_DIRECT_MUJOCO_EQUIVALENT",
        })
    return {
        "summary": {
            "exact": ["sim.dt", "decimation", "gravity", "episode_length_s", "scene.num_envs", "seed", "reward weights/order/dt integration", "policy ONNX graph"],
            "translated": ["rigid-body integration", "joint motors/PD", "friction/contact", "termination/contact sensor", "terrain sampling"],
            "non_equivalent": ["PhysX GPU solver internals", "Fabric replication", "RTX rendering (disabled)"],
        },
        "physx_fields": rows,
    }


class _Slot:
    def __init__(self, model: Any, seed: int, env_index: int, cfg: dict[str, Any]) -> None:
        import mujoco
        self.data = mujoco.MjData(model)
        self.rng = np.random.default_rng(seed)
        self.env_index = env_index
        self.last_action = np.zeros(19, dtype=np.float32)
        self.last_joint_vel = np.zeros(19)
        self.air_time = np.zeros(2)
        self.was_contact = np.ones(2, dtype=bool)
        self.command = np.zeros(3)
        self.heading = 0.0
        self.episode_reward = np.float32(0)
        self.term_sums: dict[str, np.float32] = {}
        self.fell = False
        self.reset(model, cfg)

    def reset(self, model: Any, cfg: dict[str, Any]) -> None:
        import mujoco
        mujoco.mj_resetData(model, self.data)
        init = cfg["scene"]["robot"]["init_state"]
        self.data.qpos[:3] = np.asarray(init["pos"], dtype=float)
        self.data.qpos[3:7] = np.asarray(init["rot"], dtype=float)
        defaults = init["joint_pos"]
        for index, name in enumerate(JOINT_NAMES):
            value = 0.0
            for pattern, candidate in defaults.items():
                if re.fullmatch(pattern, name): value = float(candidate)
            self.data.qpos[7 + index] = value
        # NCRC reset pose/yaw distributions, preserved from the supplied YAML.
        pose = cfg["events"]["reset_base"]["params"]["pose_range"]
        self.data.qpos[0] += self.rng.uniform(*pose["x"])
        self.data.qpos[1] += self.rng.uniform(*pose["y"])
        yaw = self.rng.uniform(*pose["yaw"])
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.data.qvel[:] = 0
        self.last_action[:] = 0
        self.last_joint_vel[:] = 0
        self.air_time[:] = 0
        self.was_contact[:] = True
        self.episode_reward = np.float32(0)
        self.term_sums = {}
        self.fell = False
        ranges = cfg["commands"]["base_velocity"]["ranges"]
        self.command = np.asarray([
            self.rng.uniform(*ranges["lin_vel_x"]), self.rng.uniform(*ranges["lin_vel_y"]), 0.0
        ])
        if self.rng.random() < cfg["commands"]["base_velocity"].get("rel_standing_envs", 0):
            self.command[:] = 0
        self.heading = self.rng.uniform(*ranges["heading"])
        mujoco.mj_forward(model, self.data)


class HeadlessH1Engine:
    def __init__(self, env_path: Path, policy_path: Path, chunk_size: int = 32) -> None:
        import mujoco
        import onnxruntime as ort
        validation = validate_reward_only_env(env_path, "H1")
        if not validation["valid"]:
            raise ValueError("env rejected by background/reward lock: " + json.dumps(validation, ensure_ascii=False))
        parsed = parse_env(env_path)
        self.cfg = parsed["data"]
        self.weights = parsed["reward_weights"]
        self.reward_terms = parsed["reward_terms"]
        self.validation = validation
        self.model = mujoco.MjModel.from_xml_path(str(_model_path()))
        self.model.opt.timestep = float(self.cfg["sim"]["dt"])
        self.model.opt.gravity[:] = np.asarray(self.cfg["sim"]["gravity"], dtype=float)
        material = self.cfg["sim"]["physics_material"]
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor >= 0:
            self.model.geom_friction[floor, :2] = [material["static_friction"], material["dynamic_friction"]]
        self.decimation = int(self.cfg["decimation"])
        self.control_dt = float(self.model.opt.timestep * self.decimation)
        self.num_envs = int(self.cfg["scene"]["num_envs"])
        self.chunk_size = max(1, min(int(chunk_size), self.num_envs))
        self.seed = int(self.cfg.get("seed", 42))
        self.episode_steps = int(round(float(self.cfg["episode_length_s"]) / self.control_dt))
        dynamic = path_for("generated/policy_dynamic_batch.onnx")
        dynamic.parent.mkdir(parents=True, exist_ok=True)
        _dynamic_onnx(policy_path, dynamic)
        options = ort.SessionOptions(); options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.session = ort.InferenceSession(str(dynamic), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.policy_source = policy_path
        self.ankles = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, x) for x in ("left_ankle_link", "right_ankle_link")]
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.default_pos = np.zeros(19)
        defaults = self.cfg["scene"]["robot"]["init_state"]["joint_pos"]
        for i, name in enumerate(JOINT_NAMES):
            for pattern, value in defaults.items():
                if re.fullmatch(pattern, name): self.default_pos[i] = float(value)
        self.kp, self.kd = self._gains()

    def _gains(self) -> tuple[np.ndarray, np.ndarray]:
        actuators = self.cfg["scene"]["robot"]["actuators"]
        kp, kd = np.zeros(19), np.zeros(19)
        for group in actuators.values():
            for index, name in enumerate(JOINT_NAMES):
                if any(re.fullmatch(p, name) for p in group["joint_names_expr"]):
                    for pattern, value in group["stiffness"].items():
                        if re.fullmatch(pattern, name): kp[index] = value
                    for pattern, value in group["damping"].items():
                        if re.fullmatch(pattern, name): kd[index] = value
        return kp, kd

    def _contacts(self, slot: _Slot) -> tuple[np.ndarray, bool]:
        import mujoco
        feet = np.zeros(2, dtype=bool); torso_contact = False
        for i in range(slot.data.ncon):
            contact = slot.data.contact[i]
            bodies = {int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])}
            for foot_index, body in enumerate(self.ankles):
                if body in bodies: feet[foot_index] = True
            if self.torso in bodies:
                force = np.zeros(6); mujoco.mj_contactForce(self.model, slot.data, i, force)
                torso_contact |= np.linalg.norm(force[:3]) > float(self.cfg["terminations"]["base_contact"]["params"]["threshold"])
        return feet, torso_contact

    def _observation(self, slot: _Slot) -> np.ndarray:
        r = _rotation(slot.data); lin_b = r.T @ slot.data.qvel[:3]; ang_b = r.T @ slot.data.qvel[3:6]
        gravity = r.T @ np.asarray([0.0, 0.0, -1.0])
        heading_cfg = self.cfg["commands"]["base_velocity"]
        ranges = heading_cfg["ranges"]
        slot.command[2] = np.clip(heading_cfg["heading_control_stiffness"] * _wrap(slot.heading - _yaw(slot.data)), *ranges["ang_vel_z"])
        joint_pos = slot.data.qpos[7:26] - self.default_pos
        joint_vel = slot.data.qvel[6:25]
        # 1.6 x 1.0 at 0.1 m = 17 x 11 = 187 rays. Plane/hfield query is
        # headless; current backend samples the collision ground height.
        heights = np.full(187, slot.data.qpos[2] - 0.5, dtype=np.float64)
        obs_cfg = self.cfg["observations"]["policy"]
        if obs_cfg.get("enable_corruption", False):
            lin_b += slot.rng.uniform(-0.1, 0.1, 3)
            ang_b += slot.rng.uniform(-0.2, 0.2, 3)
            gravity += slot.rng.uniform(-0.05, 0.05, 3)
            joint_pos += slot.rng.uniform(-0.01, 0.01, 19)
            joint_vel += slot.rng.uniform(-1.5, 1.5, 19)
            heights += slot.rng.uniform(-0.1, 0.1, 187)
        heights = np.clip(heights, -1.0, 1.0)
        result = np.concatenate((lin_b, ang_b, gravity, slot.command, joint_pos, joint_vel, slot.last_action, heights)).astype(np.float32)
        if result.shape != (256,): raise RuntimeError(f"policy observation mismatch: {result.shape}")
        return result

    def _raw_rewards(self, slot: _Slot, action: np.ndarray, terminated: bool) -> dict[str, float]:
        feet, _ = self._contacts(slot); r = _rotation(slot.data)
        yaw = _yaw(slot.data); cy, sy = math.cos(yaw), math.sin(yaw)
        velocity_yaw = np.asarray([cy * slot.data.qvel[0] + sy * slot.data.qvel[1], -sy * slot.data.qvel[0] + cy * slot.data.qvel[1]])
        std_lin = self.reward_terms["track_lin_vel_xy_exp"]["parameters"]["std"]
        std_ang = self.reward_terms["track_ang_vel_z_exp"]["parameters"]["std"]
        first_contact = feet & ~slot.was_contact
        slot.air_time[~feet] += self.control_dt
        air_reward = float(np.sum(np.maximum(slot.air_time - self.reward_terms["feet_air_time"]["parameters"]["threshold"], 0) * first_contact))
        if np.linalg.norm(slot.command[:2]) <= 0.1: air_reward = 0.0
        slot.air_time[feet] = 0; slot.was_contact = feet
        joint_pos = slot.data.qpos[7:26]; joint_vel = slot.data.qvel[6:25]
        joint_acc = (joint_vel - slot.last_joint_vel) / self.control_dt
        slot.last_joint_vel = joint_vel.copy()
        limits = self.model.jnt_range[1:20]
        ankle_ids = _joint_selection(".*_ankle")
        limit_penalty = np.sum(np.maximum(limits[ankle_ids, 0] - joint_pos[ankle_ids], 0) + np.maximum(joint_pos[ankle_ids] - limits[ankle_ids, 1], 0))
        foot_slide = 0.0
        for i, body in enumerate(self.ankles):
            if feet[i]: foot_slide += float(np.linalg.norm(slot.data.cvel[body, 3:5]))
        projected_gravity = r.T @ np.asarray([0.0, 0.0, -1.0])
        raw = {
            "track_lin_vel_xy_exp": math.exp(-float(np.sum((slot.command[:2] - velocity_yaw) ** 2)) / std_lin**2),
            "track_ang_vel_z_exp": math.exp(-float((slot.command[2] - slot.data.qvel[5]) ** 2) / std_ang**2),
            "ang_vel_xy_l2": float(np.sum((r.T @ slot.data.qvel[3:6])[:2] ** 2)),
            "dof_torques_l2": float(np.sum(slot.data.ctrl**2)),
            "dof_acc_l2": float(np.sum(joint_acc**2)),
            "action_rate_l2": float(np.sum((action - slot.last_action) ** 2)),
            "feet_air_time": air_reward,
            "flat_orientation_l2": float(np.sum(projected_gravity[:2] ** 2)),
            "dof_pos_limits": float(limit_penalty),
            "termination_penalty": float(terminated),
            "feet_slide": foot_slide,
            "joint_deviation_hip": float(np.sum(np.abs(joint_pos[_joint_selection([".*_hip_yaw", ".*_hip_roll"])] - self.default_pos[_joint_selection([".*_hip_yaw", ".*_hip_roll"])]))),
            "joint_deviation_arms": float(np.sum(np.abs(joint_pos[_joint_selection([".*_shoulder_.*", ".*_elbow"])] - self.default_pos[_joint_selection([".*_shoulder_.*", ".*_elbow"])]))),
            "joint_deviation_torso": float(np.sum(np.abs(joint_pos[_joint_selection("torso")] - self.default_pos[_joint_selection("torso")]))),
        }
        return raw

    def _advance(self, slot: _Slot, action: np.ndarray) -> None:
        import mujoco
        target = self.default_pos + float(self.cfg["actions"]["joint_pos"]["scale"]) * action
        torque = self.kp * (target - slot.data.qpos[7:26]) - self.kd * slot.data.qvel[6:25]
        slot.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        for _ in range(self.decimation): mujoco.mj_step(self.model, slot.data)

    def run(self, repetitions: int, progress: Any = None, max_control_steps: int | None = None) -> dict[str, Any]:
        if repetitions < 1: raise ValueError("repetitions must be at least 1")
        steps = min(self.episode_steps, max_control_steps) if max_control_steps else self.episode_steps
        started = time.perf_counter(); all_rewards: list[float] = []; all_falls: list[bool] = []
        all_terms: dict[str, list[float]] = {name: [] for name, term in self.reward_terms.items() if term["active"]}
        repeats = []
        workers = min(max(1, os.cpu_count() or 1), self.chunk_size)
        for repetition in range(repetitions):
            rep_rewards, rep_falls = [], []
            for first in range(0, self.num_envs, self.chunk_size):
                count = min(self.chunk_size, self.num_envs - first)
                slots = [_Slot(self.model, self.seed + repetition * self.num_envs + first + i, first + i, self.cfg) for i in range(count)]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for _step in range(steps):
                        observations = np.stack([self._observation(slot) for slot in slots])
                        actions = self.session.run([self.output_name], {self.input_name: observations})[0].astype(np.float32)
                        list(pool.map(lambda pair: self._advance(pair[0], pair[1]), zip(slots, actions)))
                        for slot, action in zip(slots, actions):
                            _, terminated = self._contacts(slot)
                            raw = self._raw_rewards(slot, action, terminated)
                            step_reward = np.float32(0)
                            for name, term in self.reward_terms.items():
                                if not term["active"]: continue
                                contribution = np.float32(np.float32(raw[name]) * np.float32(term["weight"]) * np.float32(self.control_dt))
                                step_reward = np.float32(step_reward + contribution)
                                slot.term_sums[name] = np.float32(slot.term_sums.get(name, 0) + contribution)
                            slot.episode_reward = np.float32(slot.episode_reward + step_reward)
                            slot.last_action = action.copy(); slot.fell |= terminated
                    for slot in slots:
                        rep_rewards.append(float(slot.episode_reward)); rep_falls.append(slot.fell)
                        for name in all_terms: all_terms[name].append(float(slot.term_sums.get(name, 0)))
                if progress:
                    done = repetition * self.num_envs + first + count
                    progress(done, repetitions * self.num_envs, time.perf_counter() - started)
            repeats.append({
                "repetition": repetition + 1, "environment_count": self.num_envs,
                "mean_episode_reward": float(np.mean(rep_rewards)), "std_episode_reward": float(np.std(rep_rewards)),
                "fall_rate_percent": float(np.mean(rep_falls) * 100),
            })
            all_rewards.extend(rep_rewards); all_falls.extend(rep_falls)
        elapsed = time.perf_counter() - started
        result = {
            "engine": "NCRC_LOCAL_HIGH_FIDELITY_ENGINE", "engine_version": ENGINE_VERSION,
            "physics_backend": "MUJOCO_CPU_HEADLESS", "reference_mechanism": "ISAAC_SIM_5_1_ISAAC_LAB_2_3_REWARD_MANAGER",
            "is_official_isaac_sim": False, "rendering": False, "formula_simplified": False,
            "background_lock": self.validation, "num_envs_per_repetition": self.num_envs,
            "repetitions": repetitions, "total_environment_episodes": self.num_envs * repetitions,
            "physics_dt": self.model.opt.timestep, "decimation": self.decimation, "environment_dt": self.control_dt,
            "control_steps_per_episode": steps, "elapsed_seconds": elapsed,
            "results": {
                "mean_episode_reward": float(np.mean(all_rewards)), "median_episode_reward": float(np.median(all_rewards)),
                "std_episode_reward": float(np.std(all_rewards)), "min_episode_reward": float(np.min(all_rewards)),
                "max_episode_reward": float(np.max(all_rewards)), "fall_rate_percent": float(np.mean(all_falls) * 100),
                "term_contributions": {name: {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values))} for name, values in all_terms.items()},
                "per_repetition": repeats,
            },
            "compatibility": _compatibility(self.cfg),
            "artifacts": {"env": str(self.validation["candidate_path"]), "env_sha256": sha256_file(Path(self.validation["candidate_path"])), "policy": str(self.policy_source), "policy_sha256": sha256_file(self.policy_source)},
            "limitations": [
                "MuJoCo CPU contact/solver differs from PhysX GPU, so numeric parity is not guaranteed.",
                "The current terrain collision backend is an initial headless ground implementation; terrain generator fidelity is PARTIAL.",
                "NCRC official scoring/training code is unavailable; this is local policy evaluation, not an official score.",
            ],
        }
        destination = path_for(f"generated/local_engine_result_{int(time.time())}.json")
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["result_file"] = str(destination)
        return result


def run_local_experiment(env_path: Path, policy_path: Path, repetitions: int, chunk_size: int = 32, progress: Any = None, max_control_steps: int | None = None) -> dict[str, Any]:
    return HeadlessH1Engine(env_path, policy_path, chunk_size).run(repetitions, progress, max_control_steps)
