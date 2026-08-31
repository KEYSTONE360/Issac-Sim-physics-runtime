from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ncrc_lab.database import Database
from ncrc_lab.environment import _remove_reward_sections
from ncrc_lab.eta import EtaEstimate, estimate
from ncrc_lab.experiment.analysis import confounding_classification, experiment_diff, lineage
from ncrc_lab.experiment.importer import import_run
from ncrc_lab.experiment.parsers import parse_env, parse_report, parse_reward_application_log
from ncrc_lab.recommendation.engine import current_best, pareto_front, repeat_plan
from ncrc_lab.reward.catalog import build_catalog
from ncrc_lab.source.scanner import scan_source, sha256_file
from ncrc_lab.submission.checklist import checklist
from ncrc_lab.submission.patch import patch_existing_values, validate_numeric_only


class NcrcCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = Database(self.root / "database.sqlite")
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_experiment(self, eid: str, rewards: dict, metrics: dict, parent: str | None = None,
                       official: bool = False) -> None:
        self.db.insert_experiment({
            "experiment_id": eid, "robot": "H1", "parent_experiment": parent,
            "reward_weights": rewards, "applied_rewards": rewards, "metrics": metrics,
            "official_server": int(official), "submission_eligible": int(official),
        })

    def test_system_profile_schema(self) -> None:
        from ncrc_lab.doctor import collect_system_profile
        profile = collect_system_profile()
        self.assertIn("cpu", profile); self.assertIn("memory", profile)
        self.assertFalse(profile["memory"]["pagefile_counted_as_ram"])

    def test_source_parser_ast(self) -> None:
        source = self.root / "source"; source.mkdir()
        (source / "rewards.py").write_text("import math\nclass A(B):\n def reward(self): return 1\n", encoding="utf-8")
        result = scan_source(source)
        self.assertTrue(result["source_available"])
        self.assertIn("reward", result["files"][0]["python"]["symbols"])

    def test_rule_parser_empty_is_unknown(self) -> None:
        from ncrc_lab.rules.scanner import scan_rules
        rules = self.root / "rules"; rules.mkdir()
        self.assertEqual(str(scan_rules(rules)["status"]), "UNKNOWN")

    def test_reward_catalog_from_server_env(self) -> None:
        env = self.root / "env.yaml"
        env.write_text("rewards:\n  active_term:\n    func: pkg:fn\n    weight: 1.5\n  inactive_term:\n    func: pkg:off\n    weight: null\n", encoding="utf-8")
        catalog = build_catalog("H1", self.root / "missing", [env])
        by_name = {x["name"]: x for x in catalog["rewards"]}
        self.assertEqual(by_name["active_term"]["classification"], "NCRC_ACTIVE")
        self.assertEqual(by_name["inactive_term"]["classification"], "NCRC_INACTIVE")

    def test_reward_application_warn_never_applied(self) -> None:
        log = self.root / "train.log"
        log.write_text("[OK] feet_slide = -0.4\n[OK] ghost = 1\n[WARN] skip: ghost\n", encoding="utf-8")
        result = parse_reward_application_log(log)
        self.assertIn("feet_slide", result["applied_rewards"])
        self.assertNotIn("ghost", result["applied_rewards"])
        self.assertIn("ghost", result["skipped_rewards"])

    def test_experiment_import_and_duplicate_detection(self) -> None:
        incoming = self.root / "incoming"; incoming.mkdir()
        (incoming / "env (3).yaml").write_text("rewards:\n  r:\n    func: pkg:r\n    weight: 1\n", encoding="utf-8")
        (incoming / "report (5).html").write_text("<p>fall: 20%</p><p>terrain: 5.2</p>", encoding="utf-8")
        (incoming / "policy (5).pt").write_bytes(b"safe-test")
        runs = self.root / "runs"
        with patch("ncrc_lab.experiment.importer.path_for", lambda p: self.root / p):
            first = import_run(incoming, "H1", database=self.db)
            second = import_run(incoming, "H1", database=self.db)
        self.assertEqual(first["status"], "IMPORTED")
        self.assertEqual(second["status"], "DUPLICATE_BACKUP")
        self.assertEqual(self.db.get_experiment(first["experiment_id"])["applied_rewards"], {})

    def test_experiment_diff_and_lineage(self) -> None:
        self.add_experiment("EXP-H1-0001", {"a": 1}, {"fall": 20})
        self.add_experiment("EXP-H1-0002", {"a": 2}, {"fall": 18}, "EXP-H1-0001")
        result = experiment_diff("EXP-H1-0001", "EXP-H1-0002", self.db)
        self.assertEqual(result["confounding"]["classification"], "CLEAN_SINGLE_VARIABLE")
        self.assertEqual(lineage("EXP-H1-0002", self.db), ["EXP-H1-0002", "EXP-H1-0001"])

    def test_confounding_levels(self) -> None:
        self.assertEqual(confounding_classification(0)["classification"], "REPLICATION")
        self.assertEqual(confounding_classification(2)["severity"], "MEDIUM")
        self.assertEqual(confounding_classification(5)["severity"], "VERY_HIGH")

    def test_replication_repeat_planner(self) -> None:
        self.assertEqual(repeat_plan("exploration").planned, 1)
        self.assertEqual(repeat_plan("promising").recommended, 3)
        self.assertEqual(repeat_plan(confounded=True).stage, "ISOLATION_REQUIRED")

    def test_statistics_pareto(self) -> None:
        values = [
            {"experiment_id": "a", "metrics": {"fall": 20, "terrain": 5}},
            {"experiment_id": "b", "metrics": {"fall": 19, "terrain": 6}},
        ]
        self.assertEqual([x["experiment_id"] for x in pareto_front(values, {"fall": "min", "terrain": "max"})], ["b"])

    def test_eta_model_first_run_and_history(self) -> None:
        first = estimate(None); self.assertEqual(first.confidence, "UNKNOWN")
        measured = estimate(100, [10, 11, 9, 10, 10])
        self.assertEqual(measured.confidence, "HIGH")
        for value in (measured.low_seconds, measured.high_seconds, measured.best_seconds):
            self.assertTrue(value is not None and math.isfinite(value) and value >= 0)
        with self.assertRaises(ValueError): EtaEstimate(float("nan"), 1, 1, "LOW", "bad")

    def test_budget_schema(self) -> None:
        with self.db.connect() as connection:
            connection.execute("INSERT INTO budgets(scope,total,used,reserved) VALUES('H1',10,2,1)")
            row = connection.execute("SELECT total-used-reserved AS remaining FROM budgets").fetchone()
        self.assertEqual(row["remaining"], 7)

    def test_candidate_ranking_requires_multiple_metrics(self) -> None:
        self.add_experiment("EXP-H1-0001", {"a": 1}, {"fall": 20})
        result = current_best("H1", self.db)
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_patch_numeric_only(self) -> None:
        original = "# keep\nREWARD_WEIGHTS = {'a': -1.0, \"b\": 2}\n"
        modified, missing = patch_existing_values(original, {"a": -2.5})
        self.assertFalse(missing)
        self.assertTrue(validate_numeric_only(original, modified)["ONLY_ALLOWED_VALUES_CHANGED"])
        self.assertFalse(validate_numeric_only(original, modified + "# changed\n")["ONLY_ALLOWED_VALUES_CHANGED"])

    def test_background_defaults_exclude_reward_only(self) -> None:
        source = {"sim": {"gravity": [0, 0, -9.81]}, "rewards": {"r": 1}, "scene": {"x": 2}}
        result = _remove_reward_sections(source)
        self.assertNotIn("rewards", result)
        self.assertEqual(result["sim"]["gravity"], [0, 0, -9.81])

    def test_yaml_python_tags_safe(self) -> None:
        env = self.root / "env.yaml"
        env.write_text("gravity: !!python/tuple [0.0, 0.0, -9.81]\nids: !!python/object/apply:builtins.slice [0, 2, null]\n", encoding="utf-8")
        result = parse_env(env)
        self.assertEqual(result["data"]["gravity"], [0.0, 0.0, -9.81])
        self.assertEqual(result["data"]["ids"]["__python_slice__"], [0, 2, None])

    def test_corrupt_report_does_not_crash(self) -> None:
        report = self.root / "broken.html"; report.write_bytes(b"\xff\xfe<html><p>fall: 12")
        result = parse_report(report)
        self.assertIn(result["status"], {"PARSED", "CORRUPT"})

    def test_ncrc_report_stat_cards(self) -> None:
        report = self.root / "report.html"
        report.write_text('<div class=stat><div class=sv>25.0%</div><div class=sk>낙상률 (마지막)</div></div><div class=stat><div class=sv>5.76</div><div class=sk>지형 난이도 (마지막)</div></div>', encoding="utf-8")
        metrics = parse_report(report)["metrics"]
        self.assertEqual(metrics["fall"]["value"], 25.0)
        self.assertEqual(metrics["terrain"]["value"], 5.76)

    def test_cli_patch_and_reward_routes(self) -> None:
        from ncrc_lab.cli import build_parser
        parsed = build_parser().parse_args(["patch", "EXP-H1-0001"])
        self.assertEqual(parsed.target, "EXP-H1-0001")
        parsed = build_parser().parse_args(["rewards", "functions"])
        self.assertEqual(parsed.reward_action, "functions")
        self.assertIsNone(build_parser().parse_args([]).command)

    def test_menu_path_input_removes_quotes(self) -> None:
        from ncrc_lab.cli import _input_path
        with patch("builtins.input", return_value='"F:\\folder with spaces\\env.yaml"'):
            self.assertEqual(_input_path("path: "), "F:\\folder with spaces\\env.yaml")

    def test_model_inspector_never_executes_pt(self) -> None:
        from ncrc_lab.cli import inspect_model
        model = self.root / "evil.pt"; model.write_bytes(b"not a pickle")
        result = inspect_model(model)
        self.assertFalse(result["executed"])
        self.assertEqual(result["status"], "METADATA_ONLY")

    def test_video_metadata_corrupt_safe(self) -> None:
        from ncrc_lab.cli import inspect_video
        video = self.root / "broken.mp4"; video.write_bytes(b"bad")
        self.assertIn(inspect_video(video)["status"], {"CORRUPT", "FFPROBE_NOT_AVAILABLE"})

    def test_submission_checklist_local_is_ineligible(self) -> None:
        self.add_experiment("EXP-H1-0001", {"a": 1}, {"fall": 20}, official=False)
        result = checklist("EXP-H1-0001", self.db)
        self.assertFalse(result["submission_eligible"])

    def test_database_backup(self) -> None:
        with patch("ncrc_lab.database.path_for", lambda p: self.root / p):
            target = self.db.backup()
        self.assertTrue(target.exists())
        self.assertGreater(target.stat().st_size, 0)

    def test_chunked_sha256(self) -> None:
        import hashlib
        path = self.root / "large.bin"; path.write_bytes(b"abc" * 500_000)
        self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
