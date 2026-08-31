from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import struct
import sys
import time
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import ensure_layout, load_settings, path_for, save_settings
from .database import Database
from .doctor import collect_system_profile, write_system_profile
from .environment import load_background_defaults, snapshot_background_defaults, validate_reward_only_env
from .eta import estimate, precompute_plan
from .experiment.analysis import experiment_diff, lineage
from .experiment.importer import import_run
from .recommendation.engine import current_best, pareto_front, recommend, repeat_plan
from .reward.catalog import build_catalog, write_catalog
from .reward.nvidia_reference import sync_nvidia_reward_reference
from .rules.scanner import write_rules_manifest
from .source.scanner import scan_source, write_manifest
from .submission.checklist import checklist
from .submission.patch import generate_full_patch, reward_weights_snippet


def _configure_console_encoding() -> None:
    """Keep the numbered Korean UI readable in Windows console hosts."""
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except (AttributeError, OSError):
            pass
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def emit(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def active_robot() -> str:
    return str(load_settings().get("active_robot", "H1"))


def command_doctor(_: argparse.Namespace) -> int:
    path = write_system_profile()
    profile = json.loads(path.read_text(encoding="utf-8"))
    db = Database()
    try:
        db.initialize(); db_status = "OK"
    except sqlite3.Error as exc:
        db_status = f"ERROR: {exc}"
    emit({
        "SYSTEM": profile, "NCRC_SOURCE": "FOUND" if any(path_for("ncrc_source").iterdir()) else "NOT_AVAILABLE",
        "DATABASE": db_status, "ARTIFACT_STORAGE": str(path_for("runs")),
        "PARSER": "AVAILABLE", "PYTORCH": profile["pytorch"],
        "ONNX": profile["onnx_runtime"], "VIDEO": "METADATA_ONLY" if not shutil.which("ffprobe") else "FFPROBE_AVAILABLE",
        "DISK": profile["storage"], "MEMORY": profile["memory"], "ETA_MODEL": "FIRST_RUN_LOW_CONFIDENCE",
        "profile_path": str(path),
    })
    return 0


def _native_executable() -> Path:
    candidates = [
        path_for("native/ncrc_physics.exe"),
        path_for("native_runtime/build/Release/ncrc_physics.exe"),
    ]
    for candidate in candidates:
        if candidate.exists(): return candidate
    raise FileNotFoundError("native PhysX runtime executable is missing")


def command_native_test(_: argparse.Namespace) -> int:
    executable = _native_executable()
    proc = subprocess.run([str(executable), "test"], capture_output=True, text=True, timeout=60)
    if proc.stdout: print(proc.stdout.rstrip())
    if proc.stderr: print(proc.stderr.rstrip())
    return proc.returncode


def command_cross_validate(_: argparse.Namespace) -> int:
    from .validation import cross_validate_engines
    print(precompute_plan("Engine cross-validation", ["PhysX repeat x2", "semi-implicit analytic reference", "MuJoCo matched rigid-body scene", "error metrics/report"], estimate(3, [20, 20, 20]), peak_ram="<1 GB", cpu_threads=os.cpu_count() or 1))
    result = cross_validate_engines(); emit(result)
    return 0 if result["passed"] else 2


def command_validate_env(args: argparse.Namespace) -> int:
    result = validate_reward_only_env(Path(args.env), args.robot or active_robot())
    emit(result)
    return 0 if result["valid"] else 2


def command_local_evaluate(args: argparse.Namespace) -> int:
    from .experiment.parsers import parse_env
    from .local_engine import run_local_experiment
    env = Path(args.env); policy = Path(args.policy)
    parsed = parse_env(env)
    if parsed.get("status") != "PARSED": raise ValueError("env.yaml parse failed")
    num_envs = int(parsed["data"]["scene"]["num_envs"])
    dt = float(parsed["data"]["sim"]["dt"]) * int(parsed["data"]["decimation"])
    steps = int(round(float(parsed["data"]["episode_length_s"]) / dt))
    repetitions = int(args.repetitions)
    # Measured on this machine with the same H1/ONNX loop. It is deliberately
    # a range, not false precision; completed jobs can replace this baseline.
    throughput_low, throughput_high = 700.0, 1400.0
    workload = num_envs * repetitions * steps
    print(precompute_plan(
        "Headless H1 policy evaluation",
        ["strict background lock", "256D observation", "ONNX policy", "4 physics substeps/action", "ordered reward manager", "episode aggregation"],
        estimate(workload, [throughput_low, throughput_high]),
        planned_repetitions=str(repetitions), recommended_repetitions="3 for a promising candidate",
        peak_ram="<4 GB with chunked environments", cpu_threads=os.cpu_count() or 1,
    ))
    print(f"환경/반복: {num_envs} × {repetitions} = {num_envs * repetitions:,} episodes")
    print("[ENGINE] MUJOCO CPU headless validation backend; official Isaac score가 아닙니다.")
    last_print = 0.0
    def progress(done: int, total: int, elapsed: float) -> None:
        nonlocal last_print
        if elapsed - last_print >= 5 or done == total:
            rate = done / elapsed if elapsed else 0
            eta = (total - done) / rate if rate else None
            print(f"진행 {done:,}/{total:,} ({done/total*100:.1f}%) | ETA {eta:.0f}s" if eta is not None else f"진행 {done:,}/{total:,}")
            last_print = elapsed
    result = run_local_experiment(env, policy, repetitions, int(args.chunk_size), progress, getattr(args, "max_control_steps", None))
    emit(result)
    return 0


def command_analyze_source(_: argparse.Namespace) -> int:
    manifest = write_manifest(); rules = write_rules_manifest()
    for robot in ("H1", "Go2"):
        write_catalog(robot)
    emit({"manifest": str(manifest), "rules_manifest": str(rules), "status": scan_source()["status"]})
    return 0


def command_sync_isaaclab(_: argparse.Namespace) -> int:
    print(precompute_plan(
        "NVIDIA Isaac Lab reward reference sync",
        ["resolve official main commit", "download source/**/rewards.py", "AST symbol catalog", "SHA256 manifest"],
        estimate(1, None), planned_repetitions="1", recommended_repetitions="1 per selected upstream revision",
        peak_ram="<256 MB", cpu_threads=4,
    ))
    emit(sync_nvidia_reward_reference(ref=getattr(_, "ref", "v2.3.2"), progress=print))
    return 0


def command_compatibility(_: argparse.Namespace) -> int:
    manifest = scan_source()
    result = {
        "official_source_available": manifest["source_available"],
        "compatibility": "UNKNOWN" if not manifest["source_available"] else "REQUIRES_MANUAL_VERSION_CROSS_CHECK",
        "submission_eligible_local_training": False,
    }
    target = path_for("generated/compatibility.json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8"); emit(result); return 0


def command_environment(args: argparse.Namespace) -> int:
    if args.environment_action == "set-default":
        emit(snapshot_background_defaults(Path(args.env), args.robot or active_robot()))
    else:
        emit(load_background_defaults(args.robot or active_robot()))
    return 0


def command_robot(args: argparse.Namespace) -> int:
    settings = load_settings(); settings["active_robot"] = args.robot; save_settings(settings)
    emit({"active_robot": args.robot}); return 0


def command_rewards(args: argparse.Namespace) -> int:
    catalog = build_catalog(active_robot())
    rows = catalog["rewards"]
    if args.reward_action == "list": emit(catalog)
    elif args.reward_action == "functions":
        emit({"classification": "ISAACLAB_REFERENCE", "ncrc_version_match": "UNKNOWN", "functions": catalog.get("isaaclab_reference_functions", [])})
    elif args.reward_action == "inspect":
        item = next((x for x in rows if x.get("name") == args.name), None)
        if item is None:
            item = next((x for x in catalog.get("isaaclab_reference_functions", []) if x.get("name") == args.name or x.get("qualified_name") == args.name), None)
        emit(item or {"name": args.name, "classification": "UNKNOWN", "status": "NOT_FOUND"})
    else:
        db = Database(); db.initialize(); values = []
        for row in db.list_experiments(active_robot()):
            item = db.get_experiment(row["experiment_id"])
            if item and args.name in item["reward_weights"]:
                values.append({"experiment_id": item["experiment_id"], "value": item["reward_weights"][args.name], "metrics": item["metrics"]})
        emit({"reward": args.name, "tested_values": values})
    return 0


def command_import(args: argparse.Namespace) -> int:
    folder = Path(args.path)
    total = sum(p.stat().st_size for p in folder.iterdir() if p.is_file()) if folder.is_dir() else 0
    prediction = estimate(total, None)
    print(precompute_plan("Artifact import", ["discover artifacts", "chunked SHA256", "parse env/report", "SQLite transaction"], prediction, peak_ram="<256 MB", cpu_threads=1))
    emit(import_run(folder, args.robot or active_robot(), args.parent, args.official_server)); return 0


def command_experiments(args: argparse.Namespace) -> int:
    db = Database(); db.initialize()
    if args.experiment_action == "list": emit(db.list_experiments(args.robot or active_robot()))
    elif args.experiment_action == "show": emit(db.get_experiment(args.id) or {"status": "NOT_FOUND"})
    elif args.experiment_action == "diff": emit(experiment_diff(args.a, args.b, db))
    elif args.experiment_action == "lineage": emit({"lineage": lineage(args.id, db)})
    elif args.experiment_action == "add-manual":
        weights = json.loads(args.rewards or "{}")
        metrics = json.loads(args.metrics or "{}")
        eid = db.next_experiment_id(args.robot or active_robot())
        db.insert_experiment({"experiment_id": eid, "robot": args.robot or active_robot(), "run_type": "MANUAL_HISTORICAL", "reward_weights": weights, "applied_rewards": weights, "metrics": metrics, "verdict": "INCONCLUSIVE", "confidence": "LOW", "notes": "ARTIFACTS_MISSING"})
        emit({"status": "CREATED", "experiment_id": eid, "artifacts": "MISSING"})
    return 0


def command_best(_: argparse.Namespace) -> int: emit(current_best(active_robot())); return 0


def command_pareto(_: argparse.Namespace) -> int:
    db = Database(); items = [db.get_experiment(x["experiment_id"]) for x in db.list_experiments(active_robot())]
    emit([x["experiment_id"] for x in pareto_front([x for x in items if x], {"fall": "min", "terrain": "max"})]); return 0


def command_recommend(args: argparse.Namespace) -> int:
    print(precompute_plan("Candidate recommendation", ["load experiments", "validate comparability", "rank evidence and information value"], estimate(1, [0.2]), planned_repetitions="N/A", recommended_repetitions="N/A", peak_ram="<1 GB", cpu_threads=1))
    emit(recommend(active_robot(), args.mode)); return 0


def command_patch(args: argparse.Namespace) -> int:
    if args.target == "rollback":
        if len(args.values) != 2:
            raise ValueError("patch rollback requires BACKUP and DESTINATION")
        from .submission.patch import rollback
        rollback(Path(args.values[0]), Path(args.values[1])); emit({"status": "ROLLED_BACK"}); return 0
    if args.values:
        raise ValueError("unexpected patch arguments")
    experiment_id = args.target
    db = Database(); item = db.get_experiment(experiment_id)
    if not item: emit({"status": "NOT_FOUND"}); return 2
    snippet = reward_weights_snippet(item["reward_weights"])
    snippet_path = path_for(f"generated/{experiment_id}_REWARD_WEIGHTS.py")
    snippet_path.write_text(snippet + "\n", encoding="utf-8")
    result: dict[str, Any] = {"snippet": str(snippet_path), "server_ready_full_patch": False}
    if args.source:
        result["full_patch"] = generate_full_patch(Path(args.source), item["reward_weights"])
        result["server_ready_full_patch"] = result["full_patch"]["server_ready"]
    emit(result); return 0


def command_checklist(args: argparse.Namespace) -> int: emit(checklist(args.id)); return 0


def command_budget(_: argparse.Namespace) -> int:
    db = Database(); db.initialize()
    with db.connect() as connection: rows = [dict(x) for x in connection.execute("SELECT * FROM budgets")]
    emit({"budgets": rows, "status": "UNKNOWN" if not rows else "USER_OR_RULE_SUPPLIED"}); return 0


def inspect_model(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    result = {"path": str(path), "size_bytes": path.stat().st_size, "format": suffix, "executed": False}
    if suffix in {".pt", ".pth"}:
        result.update({"status": "METADATA_ONLY", "warning": "Pickle/PT content was not executed or deserialized."})
    elif suffix == ".onnx": result["status"] = "METADATA_ONLY"
    else: result["status"] = "UNSUPPORTED"
    return result


def inspect_video(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size, "streamed": True}
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        result.update({"status": "FFPROBE_NOT_AVAILABLE", "duration": "UNKNOWN", "frames": "UNKNOWN"})
        return result
    import subprocess
    proc = subprocess.run([ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)], capture_output=True, text=True, timeout=30)
    try: result.update({"status": "PARSED" if proc.returncode == 0 else "CORRUPT", "metadata": json.loads(proc.stdout or "{}")})
    except json.JSONDecodeError: result.update({"status": "CORRUPT", "metadata": {}})
    return result


def command_model(args: argparse.Namespace) -> int: emit(inspect_model(Path(args.file))); return 0
def command_video(args: argparse.Namespace) -> int: emit(inspect_video(Path(args.file))); return 0


def command_benchmark(_: argparse.Namespace) -> int:
    candidates = [1, 2, 4, 6, 8, 10, 12]
    print(precompute_plan("CPU thread calibration", ["cold/warm short integer workload", "candidate thread comparison", "record provenance"], estimate(7, [2]), planned_repetitions="1", recommended_repetitions="3 sustained checks before production tuning", peak_ram="<256 MB", cpu_threads=12))
    import concurrent.futures, hashlib
    results = []
    payload = b"ncrc-reward-lab" * 8192
    for workers in candidates:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda _: hashlib.sha256(payload).digest(), range(workers * 25)))
        results.append({"threads": workers, "seconds": time.perf_counter() - started, "scope": "hash_calibration_not_training"})
    target = path_for(f"benchmark_profiles/thread_benchmark_{int(time.time())}.json")
    target.write_text(json.dumps(results, indent=2), encoding="utf-8"); emit({"results": results, "path": str(target)}); return 0


def command_eta(_: argparse.Namespace) -> int: emit(estimate(None).as_dict()); return 0


def command_submission(args: argparse.Namespace) -> int:
    item = Database().get_experiment(args.id)
    if not item: emit({"status": "NOT_FOUND"}); return 2
    errors = []
    if not item["official_server"]: errors.append("LOCAL_ORIGIN_NOT_OFFICIAL_SERVER")
    if not item["submission_eligible"]: errors.append("SUBMISSION_ELIGIBLE_FALSE")
    if item["skipped_rewards"]: errors.append("SKIPPED_REWARDS_PRESENT")
    emit({"experiment_id": args.id, "valid": not errors, "errors": errors}); return 0 if not errors else 1


def command_placeholder(args: argparse.Namespace) -> int:
    emit({"status": "NOT_AVAILABLE_IN_CORE_MVP", "command": args.command, "evidence_status": "UNKNOWN", "note": "No result was fabricated."}); return 3


def _input_path(prompt: str) -> str:
    """Accept pasted or Explorer drag-and-drop paths, with optional quotes."""
    return input(prompt).strip().strip('"').strip("'")


def _choose_robot() -> str:
    current = active_robot()
    value = input(f"로봇 [현재 {current}, H1/Go2]: ").strip()
    if not value:
        return current
    normalized = "Go2" if value.lower() == "go2" else "H1" if value.lower() == "h1" else ""
    if not normalized:
        print("H1 또는 Go2만 입력할 수 있습니다.")
        return current
    return normalized


def _experiment_menu() -> None:
    print("\n1. 실험 목록  2. 실험 상세  3. 두 실험 비교  4. 계보  0. 뒤로")
    choice = input("선택: ").strip()
    if choice == "1": command_experiments(SimpleNamespace(experiment_action="list", robot=None))
    elif choice == "2": command_experiments(SimpleNamespace(experiment_action="show", id=input("실험 ID: ").strip()))
    elif choice == "3": command_experiments(SimpleNamespace(experiment_action="diff", a=input("기준 실험 ID: ").strip(), b=input("비교 실험 ID: ").strip()))
    elif choice == "4": command_experiments(SimpleNamespace(experiment_action="lineage", id=input("실험 ID: ").strip()))


def _reward_menu() -> None:
    print("\n1. 현재 reward term  2. reward 상세  3. 실험 이력  4. Isaac Lab 함수 전체  0. 뒤로")
    choice = input("선택: ").strip()
    if choice == "1": command_rewards(SimpleNamespace(reward_action="list"))
    elif choice == "2": command_rewards(SimpleNamespace(reward_action="inspect", name=input("Reward 이름: ").strip()))
    elif choice == "3": command_rewards(SimpleNamespace(reward_action="history", name=input("Reward 이름: ").strip()))
    elif choice == "4": command_rewards(SimpleNamespace(reward_action="functions"))


def _server_menu() -> None:
    print("\n1. REWARD_WEIGHTS 생성  2. 서버 체크리스트  3. 제출 검증  0. 뒤로")
    choice = input("선택: ").strip()
    experiment_id = input("실험 ID: ").strip() if choice in {"1", "2", "3"} else ""
    if choice == "1":
        source = _input_path("원본 rewards.py 경로 (snippet만 만들려면 Enter): ")
        command_patch(SimpleNamespace(target=experiment_id, values=[], source=source or None))
    elif choice == "2": command_checklist(SimpleNamespace(id=experiment_id))
    elif choice == "3": command_submission(SimpleNamespace(id=experiment_id))


def interactive_menu() -> int:
    """Beginner-friendly numbered workflow. Existing CLI commands remain available."""
    ensure_layout(); Database().initialize()
    while True:
        print("\n" + "=" * 58)
        print("NCRC REWARD LAB")
        print(f"현재 로봇: {active_robot()}")
        print("=" * 58)
        print("1. PhysX 5.6.1 native 엔진 실제 점검")
        print("2. 엔진 교차검증 (PhysX/해석해/MuJoCo)")
        print("3. H1 로컬 정책 평가 (경로 입력)")
        print("4. env.yaml 배경 잠금 검증")
        print("5. 시스템 점검")
        print("6. 학습 결과 폴더 가져오기")
        print("7. 배경 기본 env.yaml 확인")
        print("8. 실험·Reward 조회")
        print("9. Best / 추천 / 서버 설정")
        print("0. 종료")
        choice = input("선택 번호: ").strip()
        try:
            if choice == "0":
                print("종료합니다.")
                return 0
            if choice == "1": command_native_test(SimpleNamespace())
            elif choice == "2": command_cross_validate(SimpleNamespace())
            elif choice == "3":
                env = _input_path("env.yaml 파일 경로: ")
                policy = _input_path("policy.onnx 파일 경로: ")
                print("반복 횟수: 1. 1회  2. 3회  3. 5회  4. 직접 입력")
                repeat_choice = input("선택: ").strip()
                repetitions = {"1": 1, "2": 3, "3": 5}.get(repeat_choice)
                if repetitions is None: repetitions = int(input("반복 횟수: ").strip())
                command_local_evaluate(SimpleNamespace(env=env, policy=policy, repetitions=repetitions, chunk_size=32, max_control_steps=None))
            elif choice == "4":
                command_validate_env(SimpleNamespace(env=_input_path("env.yaml 파일 경로: "), robot=active_robot()))
            elif choice == "5": command_doctor(SimpleNamespace())
            elif choice == "6":
                folder = _input_path("학습 결과 폴더 경로: ")
                robot = _choose_robot()
                parent = input("부모 실험 ID (없으면 Enter): ").strip() or None
                official = input("공식 NCRC 서버 생성 자료입니까? [y/N]: ").strip().lower() == "y"
                command_import(SimpleNamespace(path=folder, robot=robot, parent=parent, official_server=official))
            elif choice == "7": command_environment(SimpleNamespace(environment_action="show-default", robot=active_robot()))
            elif choice == "8":
                print("1. 실험  2. Reward  0. 뒤로")
                selected = input("선택: ").strip()
                if selected == "1": _experiment_menu()
                elif selected == "2": _reward_menu()
            elif choice == "9":
                print("1. Best/Pareto  2. 다음 실험 추천  0. 뒤로")
                subchoice = input("선택: ").strip()
                if subchoice == "1": command_best(SimpleNamespace())
                elif subchoice == "2":
                    mode = input("모드 [balanced/isolation/aggressive/finalist, 기본 balanced]: ").strip() or "balanced"
                    if mode not in {"isolation", "conservative", "balanced", "aggressive", "ablation", "replication", "finalist"}:
                        print("지원하지 않는 모드이므로 balanced를 사용합니다."); mode = "balanced"
                    command_recommend(SimpleNamespace(mode=mode))
            else: print("메뉴에 있는 번호를 입력하세요.")
        except (FileNotFoundError, ValueError, KeyError, sqlite3.Error, OSError) as exc:
            print(f"[ERROR] {type(exc).__name__}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ncrc-lab")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor").set_defaults(func=command_doctor)
    sub.add_parser("native-test").set_defaults(func=command_native_test)
    sub.add_parser("cross-validate").set_defaults(func=command_cross_validate)
    p=sub.add_parser("validate-env"); p.add_argument("env"); p.add_argument("--robot", choices=["H1","Go2"]); p.set_defaults(func=command_validate_env)
    p=sub.add_parser("evaluate-local"); p.add_argument("env"); p.add_argument("policy"); p.add_argument("--repetitions", type=int, default=1); p.add_argument("--chunk-size", type=int, default=32); p.add_argument("--max-control-steps", type=int); p.set_defaults(func=command_local_evaluate)
    sub.add_parser("analyze-source").set_defaults(func=command_analyze_source)
    p = sub.add_parser("sync-isaaclab-reference"); p.add_argument("--ref", default="v2.3.2"); p.set_defaults(func=command_sync_isaaclab)
    sub.add_parser("compatibility").set_defaults(func=command_compatibility)
    p = sub.add_parser("robot"); p.add_argument("robot", choices=["H1", "Go2"]); p.set_defaults(func=command_robot)
    p = sub.add_parser("environment"); es = p.add_subparsers(dest="environment_action", required=True)
    q = es.add_parser("set-default"); q.add_argument("env"); q.add_argument("--robot", choices=["H1", "Go2"])
    q = es.add_parser("show-default"); q.add_argument("--robot", choices=["H1", "Go2"])
    p.set_defaults(func=command_environment)
    p = sub.add_parser("rewards"); rs = p.add_subparsers(dest="reward_action", required=True)
    rs.add_parser("list"); rs.add_parser("functions"); q=rs.add_parser("inspect"); q.add_argument("name"); q=rs.add_parser("history"); q.add_argument("name"); p.set_defaults(func=command_rewards)
    p=sub.add_parser("import-run"); p.add_argument("path"); p.add_argument("--robot", choices=["H1","Go2"]); p.add_argument("--parent"); p.add_argument("--official-server", action="store_true"); p.set_defaults(func=command_import)
    p=sub.add_parser("experiments"); es=p.add_subparsers(dest="experiment_action", required=True)
    q=es.add_parser("list"); q.add_argument("--robot", choices=["H1","Go2"])
    q=es.add_parser("show"); q.add_argument("id")
    q=es.add_parser("diff"); q.add_argument("a"); q.add_argument("b")
    q=es.add_parser("lineage"); q.add_argument("id")
    q=es.add_parser("add-manual"); q.add_argument("--robot", choices=["H1","Go2"]); q.add_argument("--rewards"); q.add_argument("--metrics")
    p.set_defaults(func=command_experiments)
    sub.add_parser("best").set_defaults(func=command_best); sub.add_parser("pareto").set_defaults(func=command_pareto)
    p=sub.add_parser("recommend"); p.add_argument("--mode", default="balanced", choices=["isolation","conservative","balanced","aggressive","ablation","replication","finalist"]); p.set_defaults(func=command_recommend)
    p=sub.add_parser("patch"); p.add_argument("target"); p.add_argument("values", nargs="*"); p.add_argument("--source"); p.set_defaults(func=command_patch)
    p=sub.add_parser("checklist"); p.add_argument("id"); p.set_defaults(func=command_checklist)
    sub.add_parser("budget").set_defaults(func=command_budget)
    p=sub.add_parser("model"); ms=p.add_subparsers(dest="model_action", required=True); q=ms.add_parser("inspect"); q.add_argument("file"); p.set_defaults(func=command_model)
    p=sub.add_parser("video"); vs=p.add_subparsers(dest="video_action", required=True); q=vs.add_parser("inspect"); q.add_argument("file"); p.set_defaults(func=command_video)
    sub.add_parser("benchmark").set_defaults(func=command_benchmark); sub.add_parser("eta").set_defaults(func=command_eta)
    p=sub.add_parser("submission"); ss=p.add_subparsers(dest="submission_action", required=True); q=ss.add_parser("validate"); q.add_argument("id"); p.set_defaults(func=command_submission)
    for name in ("watch", "evidence", "queue", "report", "parity"):
        p=sub.add_parser(name); p.add_argument("args", nargs="*"); p.set_defaults(func=command_placeholder)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_console_encoding()
    ensure_layout(); Database().initialize()
    parser = build_parser(); args = parser.parse_args(argv)
    if args.command is None:
        return interactive_menu()
    try: return int(args.func(args))
    except (FileNotFoundError, ValueError, KeyError, sqlite3.Error) as exc:
        emit({"status": "ERROR", "type": type(exc).__name__, "message": str(exc)}); return 2


if __name__ == "__main__":
    raise SystemExit(main())
